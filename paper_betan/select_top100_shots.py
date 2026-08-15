#!/usr/bin/env python3
"""从 per_shot_preds.npz 选「PF 电流预测误差最小」的 N 炮, 写成 FreeGSNKE 评估用的炮清单。

目的: FreeGSNKE 前向求解昂贵, 只挑预测最准的 N 炮来做平衡一致性统计 (老师要求)。
口径: 每炮在 action_mask 内, 对 12 通道×有效步求整体 RMSE (kA, 越低越好), 升序取前 N。

读:  results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz
写:  meta/split_by_order_betan/freegsnke_100best.txt  (一行一炮号)

⚠️ 透明度: 按"电流误差最小"挑炮存在选择偏差 (这些炮的平衡边界误差也倾向偏小)。
   图注/正文必须声明 "on the N best-predicted test discharges"; 全集回归精度由 Table 1/Fig 3 承担。

用法:
  python -m paper_betan.select_top100_shots --n 100
  python -m paper_betan.select_top100_shots --n 100 --metric mae
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_NPZ = Path(
    "results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
DEFAULT_OUT = Path("meta/split_by_order_betan/freegsnke_100best.txt")


def per_shot_error(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray, metric: str) -> np.ndarray:
    """每炮标量误差 (n_shots,). pred/tgt: (n, T, 12) kA; mask: (n, T, 12) bool."""
    diff2 = (pred - tgt) ** 2 * mask                      # (n, T, 12)
    n_eff = mask.reshape(mask.shape[0], -1).sum(axis=1)    # (n,) 有效元素数
    n_eff = np.maximum(n_eff, 1)
    if metric == "rmse":
        s = diff2.reshape(diff2.shape[0], -1).sum(axis=1)
        return np.sqrt(s / n_eff)
    if metric == "mae":
        ab = (np.abs(pred - tgt) * mask).reshape(mask.shape[0], -1).sum(axis=1)
        return ab / n_eff
    raise ValueError(f"unknown metric: {metric}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=100, help="选误差最小的 N 炮")
    ap.add_argument("--metric", choices=["rmse", "mae"], default="rmse")
    args = ap.parse_args()

    npz = Path(args.npz)
    if not npz.is_file():
        sys.exit(f"npz not found: {npz}")
    d = np.load(npz, allow_pickle=False)
    shots = np.asarray(d["shots"]).ravel()
    pred = np.asarray(d["pred_kA"], float)
    tgt = np.asarray(d["target_kA"], float)
    mask = np.asarray(d["action_mask"], bool)
    d.close()
    assert pred.shape == tgt.shape == mask.shape, f"shape mismatch {pred.shape} {tgt.shape} {mask.shape}"

    err = per_shot_error(pred, tgt, mask, args.metric)
    order = np.argsort(err, kind="stable")          # 升序, 误差小在前
    n = min(args.n, len(shots))
    pick = order[:n]
    pick_shots = shots[pick]
    pick_err = err[pick]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(str(int(s)) for s in pick_shots) + "\n")

    # ---- 诊断打印 (供图注/正文引用选择区间) ----
    print(f"[select] metric={args.metric}  n_picked={n} / {len(shots)} test shots")
    print(f"[select] full-set {args.metric}: median={np.median(err):.4f} "
          f"p25={np.percentile(err,25):.4f} p75={np.percentile(err,75):.4f} "
          f"min={err.min():.4f} max={err.max():.4f} kA")
    print(f"[select] picked-{n} {args.metric}: max={pick_err.max():.4f} "
          f"median={np.median(pick_err):.4f} kA  (即全集中前 {n/len(shots):.1%})")
    print(f"[select] shot# range: {int(pick_shots.min())}..{int(pick_shots.max())}")
    print(f"[select] wrote {args.out}")
    print(f"[select] first 10: {pick_shots[:10].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
