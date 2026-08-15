#!/usr/bin/env python
"""画单炮 12 路 PF 线圈: pred vs truth (paper Figure 1 style, 无主标题).
4×3 网格, truth=C0(蓝) pred=C3(红,alpha0.85), 每通道标题带 R²+RMSE.
用法: SHOT=156715 python plot_pf_timeseries_shot.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz"
OUTDIR = ROOT / "results/betan_ablation/figures"
SHOT = int(os.environ["SHOT"])
PF_NAMES = ['PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6',
            'PF7', 'PF8', 'PF11', 'PF12', 'PF13', 'PF14']

d = np.load(NPZ, allow_pickle=True)
shots = [int(s) for s in d['shots']]
if SHOT not in shots:
    raise SystemExit(f"shot {SHOT} not in {NPZ}")
idx = shots.index(SHOT)
T = int(d['T'][idx])
t = d['time'][idx][:T].astype(float)
pred = d['pred_kA'][idx][:T]
truth = d['target_kA'][idx][:T]
mask = d['action_mask'][idx][:T]


def r2_1d(p, tr):
    m = np.isfinite(p) & np.isfinite(tr)
    p, tr = p[m], tr[m]
    if p.size < 2:
        return float('nan')
    ss_res = float(np.sum((p - tr) ** 2))
    ss_tot = float(np.sum((tr - tr.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def rmse_1d(p, tr):
    m = np.isfinite(p) & np.isfinite(tr)
    p, tr = p[m], tr[m]
    if p.size < 1:
        return float('nan')
    return float(np.sqrt(np.mean((p - tr) ** 2)))


fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True)
for ch in range(12):
    ax = axes[ch // 3, ch % 3]
    ax.plot(t, truth[:, ch], "C0", lw=1.0, label="truth")
    ax.plot(t, pred[:, ch], "C3", lw=0.9, alpha=0.85, label="pred")
    m = mask[:, ch]
    pch, tch = pred[:, ch][m], truth[:, ch][m]
    r2 = r2_1d(pch, tch)
    rmse = rmse_1d(pch, tch)
    ax.set_title(f"{PF_NAMES[ch]}  ($R^2$={r2:.3f}, RMSE={rmse:.2f} kA)", fontsize=9)
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("t (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("Current (kA)")
axes[0, 0].legend(fontsize=8, loc="upper right")
plt.tight_layout()
OUTDIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUTDIR / f"shot_timeseries_{SHOT}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {OUTDIR}/shot_timeseries_{SHOT}.{{png,pdf}}")
