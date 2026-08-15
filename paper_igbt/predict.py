"""用最优模型对 test 集推理一次, 存 padded per_shot_preds.npz (完整字段)。

后续 eval_test / constraints_fig / plot_figures 全部读这一个 npz, 不再推理。
字段 (padded to T_max=128): shots(N,) T(N,) time/dt(N,128) pred_kA/target_kA(N,128,12)
action_mask(N,128,12) Ip_A(N,128) phase_ids(N,128) step_phase_ids(N,127)。

用法(计算节点): python -m paper_igbt.predict [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_igbt.model import BEST_CKPT, load_best_model  # noqa: E402
from ablation.data.dataset import PFDataset, load_norm_stats, read_shots_txt  # noqa: E402
from ablation.data.phases import phase_step_ids  # noqa: E402 (model-agnostic, 与 bc 一致)
from ablation.common.constants import T_MAX  # noqa: E402
from bc.training.utils import deep_get  # noqa: E402 (model-agnostic util)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(BEST_CKPT))
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/paper_igbt/predictions/per_shot_preds.npz")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg, extra = load_best_model(args.ckpt, device)
    print(f"loaded {args.ckpt}  val_loss={extra.get('val_loss')}  model={cfg['model']}")

    shots = read_shots_txt(deep_get(cfg, f"data.{args.split}_shots"))
    if args.limit:
        shots = shots[: args.limit]
    T_max = int(deep_get(cfg, "data.T_max", T_MAX))
    ds = PFDataset(
        shots,
        dataset_root=deep_get(cfg, "data.dataset_root"),
        norm_stats_path=deep_get(cfg, "data.norm_stats"),
        T_max=T_max,
    )
    ns = load_norm_stats(deep_get(cfg, "data.norm_stats"))
    a_mean = torch.as_tensor(ns["action_mean"], device=device)
    a_std = torch.as_tensor(ns["action_std"], device=device)
    ip_mean = float(ns["state_mean"][18])  # PCRL01 = idx 18 in 19-dim state
    ip_std = float(ns["state_std"][18])

    N = len(shots)
    time_p = np.zeros((N, T_max), np.float32)
    dt_p = np.zeros((N, T_max), np.float32)
    pred_p = np.zeros((N, T_max, 12), np.float32)
    tgt_p = np.zeros((N, T_max, 12), np.float32)
    mask_p = np.zeros((N, T_max, 12), bool)
    ip_p = np.zeros((N, T_max), np.float32)
    pid_p = np.full((N, T_max), -1, np.int8)
    stepid_p = np.full((N, T_max - 1), -1, np.int8)
    shots_arr = np.zeros(N, np.int64)
    T_arr = np.zeros(N, np.int64)

    t0 = time.monotonic()
    with torch.no_grad():
        for i, shot in enumerate(shots):
            s = ds[i]
            state = s["state"].unsqueeze(0).to(device)
            tphys = s["time_phys"].unsqueeze(0).to(device)
            tmask = s["token_mask"].unsqueeze(0).to(device)
            action = s["action"].unsqueeze(0).to(device)  # (1,128,12) ← 必须 unsqueeze, 否则 action[0] 只取第1步
            amask = s["action_mask"]
            pred = model(state, tphys, tmask)  # (1,T,12) normalized
            pred_kA = (pred[0] * a_std + a_mean).cpu().numpy().astype(np.float32) / 1e3
            tgt_kA = (action[0] * a_std + a_mean).cpu().numpy().astype(np.float32) / 1e3
            T = int(s["T"])
            shots_arr[i] = shot
            T_arr[i] = T
            time_p[i, :T] = s["time_phys"][:T].numpy()
            dt_p[i, :T] = s["dt_phys"][:T].numpy()
            pred_p[i, :T] = pred_kA[:T]
            tgt_p[i, :T] = tgt_kA[:T]
            mask_p[i, :T] = amask[:T].numpy()
            ip_norm = s["state"][:T, 18].numpy().astype(np.float64)
            ip_p[i, :T] = (ip_norm * ip_std + ip_mean).astype(np.float32)
            pid = s["phase_ids"][:T].numpy().astype(np.int8)
            pid_p[i, :T] = pid
            if T > 1:
                stepid_p[i, : T - 1] = phase_step_ids(pid).astype(np.int8)
            if (i + 1) % 200 == 0:
                print(f"  predicted {i + 1}/{N}  ({time.monotonic()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        shots=shots_arr, T=T_arr, T_max=np.array(T_max),
        time=time_p, dt=dt_p, pred_kA=pred_p, target_kA=tgt_p, action_mask=mask_p,
        Ip_A=ip_p, phase_ids=pid_p, step_phase_ids=stepid_p,
        val_loss=np.array(float(extra.get("val_loss", float("nan")))),
    )
    print(f"saved {out} ({N} shots, {time.monotonic()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
