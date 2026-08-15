"""单炮 BC 推理 + phase 切片.

加载 best_val.pt, 对指定 shot 整炮 forward, 反归一化到 Ampere, 同时
返回 phase slices 供下游 FreeGS 对比用.

Usage (on compute node with torch):
    from bc.infer_one_shot import infer_one_shot, pick_representative_steps

    out = infer_one_shot(shot=142857,
                        ckpt_path="results/bc_v1/run1/checkpoints/best_val.pt",
                        device="cuda:0",
                        dataset_root="/path/to/PF_ATIME_dataset")  # 可选
    picks = pick_representative_steps(out["phase_slices"])
    # picks 可能只含 'ramp_up'/'flat_top' (若无 ramp_down 段)

Returns a dict with:
    shot, time (T,), pred_A (T,12), target_A (T,12), Ip_A (T,),
    phase_ids (T,), phase_slices, state_raw_denorm (T,19)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from bc.common.constants import PCPF_NAMES, T_MAX
from bc.data.dataset import build_sample_arrays, load_norm_stats
from bc.data.phases import detect_phase_slices, phase_ids_per_step


def _import_torch():
    import torch
    return torch


def infer_one_shot(
    shot: int,
    ckpt_path: str | Path,
    device: str | None = None,
    dataset_root: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load ckpt + forward one shot. Returns denormalized arrays in A.

    ``dataset_root`` 覆盖 ckpt 内 cfg 的路径。
    若 ``{dataset_root}/{shot}.h5`` 不存在：当传入 ``mds_server=`` 或环境变量 ``MDS_HOSTNAME`` 时，
    经 ``mds_pf_atime.build_sample_arrays_from_mds`` 从 MDS 建样本（与 ``run_freegsnke_eval --mds-server`` 一致）。
    """
    torch = _import_torch()
    from bc.models.model import CausalTransformer  # 延迟导入，避免 import 本模块即加载 torch

    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["extra"]["cfg"]
    model = CausalTransformer(**cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ns = load_norm_stats(cfg["data"]["norm_stats"])
    ds_root = Path(cfg["data"]["dataset_root"]) if dataset_root is None else Path(dataset_root)
    h5_path = ds_root / f"{int(shot)}.h5"
    mds_server = kwargs.get("mds_server") or os.environ.get("MDS_HOSTNAME")

    if h5_path.is_file():
        arrs = build_sample_arrays(
            h5_path,
            ns["state_mean"],
            ns["state_std"],
            ns["action_mean"],
            ns["action_std"],
            T_max=T_MAX,
        )
    else:
        if not (mds_server and str(mds_server).strip()):
            raise FileNotFoundError(
                f"缺少本地 ATIME 数据集文件: {h5_path}\n"
                "请任选其一："
                f"(1) 将含该炮的 PF_ATIME_dataset 挂到该路径或把 --dataset-root 指到有 {int(shot)}.h5 的目录；"
                "(2) 设置 MDS_HOSTNAME / 传入 mds_server= 或 run_freegsnke_eval 的 --mds-server，从 MDS 拉 efit_east+pcs_east。"
            )
        from bc.gs_forward.mds_pf_atime import build_sample_arrays_from_mds

        arrs = build_sample_arrays_from_mds(
            int(shot),
            ns["state_mean"],
            ns["state_std"],
            ns["action_mean"],
            ns["action_std"],
            T_max=T_MAX,
            mds_server=str(mds_server).strip(),
        )

    T_eff = int(arrs["T"])
    state = torch.from_numpy(arrs["state"]).unsqueeze(0).to(device)
    time_phys = torch.from_numpy(arrs["time_phys"]).unsqueeze(0).to(device)
    token_mask = torch.from_numpy(arrs["token_mask"]).unsqueeze(0).to(device)
    target = torch.from_numpy(arrs["action"]).to(device)

    with torch.no_grad():
        pred = model(state, time_phys, token_mask)  # (1, T_max, 12) normalized

    a_mean = torch.as_tensor(ns["action_mean"], device=device)
    a_std = torch.as_tensor(ns["action_std"], device=device)
    pred_A = ((pred[0, :T_eff] * a_std + a_mean)).cpu().numpy().astype(np.float64)  # (T, 12)
    target_A = ((target[:T_eff] * a_std + a_mean)).cpu().numpy().astype(np.float64)

    # Ip from state ch18 (PCRL01), state 已归一化
    s_mean = ns["state_mean"].astype(np.float64)
    s_std = ns["state_std"].astype(np.float64)
    state_np = state[0, :T_eff].cpu().numpy().astype(np.float64)
    state_denorm = state_np * s_std + s_mean
    Ip_A = state_denorm[:, 18].copy()

    time_s = time_phys[0, :T_eff].cpu().numpy().astype(np.float64)
    phase_ids = phase_ids_per_step(time_s, Ip_A, valid_len=T_eff)
    phase_slices = detect_phase_slices(time_s, Ip_A)

    return {
        "shot": int(shot),
        "time": time_s,  # (T,) seconds
        "pred_A": pred_A,  # (T, 12) A, order = PCPF_NAMES
        "target_A": target_A,  # (T, 12) A
        "Ip_A": Ip_A,  # (T,) A
        "phase_ids": phase_ids,
        "phase_slices": phase_slices,
        "state_denorm": state_denorm,  # (T, 19) 反归一化后 state
        "pcpf_names": list(PCPF_NAMES),
        "cfg": cfg,
    }


def pick_representative_steps(
    phase_slices: dict[str, slice],
    policy: str = "middle",
) -> dict[str, int]:
    """Per phase select one representative step index.
    policy='middle' -> (start + stop - 1) // 2 (phase 中点)
    """
    picks: dict[str, int] = {}
    for name, sl in phase_slices.items():
        if sl.stop > sl.start:
            if policy == "middle":
                picks[name] = (sl.start + sl.stop - 1) // 2
            elif policy == "start":
                picks[name] = sl.start
            elif policy == "end":
                picks[name] = sl.stop - 1
            else:
                raise ValueError(policy)
    return picks
