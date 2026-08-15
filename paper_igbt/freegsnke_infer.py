"""infer_one_shot 的最优模型版 (build_model + ablation 19 维 dataset), 返回契约与 bc 一致。

bc/gs_forward/infer_one_shot.py:63 用 CausalTransformer(**cfg) + bc.build_sample_arrays(21维),
对 s44 会崩; 这里换 build_model + ablation.build_sample_arrays(19维)。其余 (pred_A/target_A/
Ip_A/phase_slices) 逻辑不变, 模型无关。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.common.constants import T_MAX  # noqa: E402 (19 维)
from ablation.data.dataset import build_sample_arrays, load_norm_stats  # noqa: E402
from bc.common.constants import PCPF_NAMES  # noqa: E402 (模型无关)
from bc.data.phases import detect_phase_slices, phase_ids_per_step  # noqa: E402 (模型无关)


def infer_one_shot_best(
    shot: int,
    ckpt_path: str | Path,
    device: str | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load s44 + forward one shot. Returns denormalized arrays in Amperes."""
    import torch

    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"
    from ablation.models.factory import build_model  # 延迟, 避免无 torch 环境报错

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["extra"]["cfg"]
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ns = load_norm_stats(cfg["data"]["norm_stats"])
    ds_root = Path(cfg["data"]["dataset_root"]) if dataset_root is None else Path(dataset_root)
    h5_path = ds_root / f"{int(shot)}.h5"
    assert h5_path.is_file(), f"缺少 {h5_path} (134925 应在 PF_ATIME_dataset)"

    arrs = build_sample_arrays(
        h5_path, ns["state_mean"], ns["state_std"], ns["action_mean"], ns["action_std"], T_max=T_MAX
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
    pred_A = ((pred[0, :T_eff] * a_std + a_mean)).cpu().numpy().astype(np.float64)  # (T,12) A
    target_A = ((target[:T_eff] * a_std + a_mean)).cpu().numpy().astype(np.float64)

    s_mean = ns["state_mean"].astype(np.float64)
    s_std = ns["state_std"].astype(np.float64)
    state_denorm = state[0, :T_eff].cpu().numpy().astype(np.float64) * s_std + s_mean
    Ip_A = state_denorm[:, 18].copy()  # PCRL01 = idx 18

    time_s = time_phys[0, :T_eff].cpu().numpy().astype(np.float64)
    phase_ids = phase_ids_per_step(time_s, Ip_A, valid_len=T_eff)
    phase_slices = detect_phase_slices(time_s, Ip_A)

    return {
        "shot": int(shot), "time": time_s, "pred_A": pred_A, "target_A": target_A,
        "Ip_A": Ip_A, "phase_ids": phase_ids, "phase_slices": phase_slices,
        "state_denorm": state_denorm, "pcpf_names": list(PCPF_NAMES), "cfg": cfg,
    }
