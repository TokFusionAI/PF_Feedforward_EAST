#!/usr/bin/env python3
"""FreeGSNKE 统计图候选变体 B/C/D (供对比挑选)。复用 v2 的数据加载与配色。

  B = parity R/Z + CDF(仅 flat_top+ramp_down)          最显好
  C = parity R/Z + RMSE-violin(按3相位)                 平衡: R²显好 + cm诚实
  D = 仅 parity R/Z (大两联板)                          最美/极简
已有的: A=freegsnke_stats_v2.png (parity+CDF 3相位); 原版=freegsnke_stats.png (4联板)
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py

EFIT = "/data/EFIT"
CSV = "results/paper_betan/freegsnke_testset/per_frame.csv"
OUTDIR = Path("results/paper_betan/figures")
PHASES = ["ramp_up", "flat_top", "ramp_down"]
PCOL = {"ramp_up": "#D55E00", "flat_top": "#0072B2", "ramp_down": "#009E73"}
CMAP = "Purples"


def r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    sse = np.sum((x - y) ** 2); sst = np.sum((x - x.mean()) ** 2)
    return 1 - sse / sst if sst > 0 else float("nan")


def get_r8z8(shot, k):
    with h5py.File(f"{EFIT}/{int(shot)}.h5", "r") as f:
        at = np.asarray(f["ATIME"][:]).ravel(); m = at >= 0
        R8 = np.asarray(f["R8"][:], float); Z8 = np.asarray(f["Z8"][:], float)
        if R8.ndim == 2 and R8.shape[0] == 8 and R8.shape[1] != 8: R8 = R8.T
        if Z8.ndim == 2 and Z8.shape[0] == 8 and Z8.shape[1] != 8: Z8 = Z8.T
        if R8.shape[0] == len(at): R8 = R8[m]
        if Z8.shape[0] == len(at): Z8 = Z8[m]
        return R8[k], Z8[k]


def load():
    rows = [r for r in csv.DictReader(open(CSV))]
    cv = [r for r in rows if r["converged"] == "True" and r.get("rmse_8ctrl_m")]
    ER, PR, EZ, PZ = [], [], [], []
    for r in cv:
        try:
            er, ez = get_r8z8(int(r["shot"]), int(r["k"]))
        except Exception:
            continue
        if er.size != 8:
            continue
        dr = np.array([float(r[f"dr_{i}"]) for i in range(8)])
        dz = np.array([float(r[f"dz_{i}"]) for i in range(8)])
        ER.append(er); PR.append(er + dr); EZ.append(ez); PZ.append(ez + dz)
    return cv, np.concatenate(ER), np.concatenate(PR), np.concatenate(EZ), np.concatenate(PZ)


def panel_parity(ax, x, y, lab, tag):
    hb = ax.hexbin(x, y, gridsize=55, cmap=CMAP, mincnt=1, norm=LogNorm())
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#D55E00", lw=1.4, zorder=5, label="y = x")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.96, f"$R^2 = {r2(x,y):.3f}$\n$n = {len(x)}$", transform=ax.transAxes,
            va="top", fontsize=11, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbbbbb", boxstyle="round,pad=0.35"))
    ax.set_xlabel(f"EFIT control-point {lab} (m)", fontsize=11)
    ax.set_ylabel(f"FreeGSNKE(pred) {lab} (m)", fontsize=11)
    ax.set_title(f"{tag}  Control-point {lab} parity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    return hb


def panel_cdf(ax, cv, phases, title):
    xs = np.linspace(0, 35, 400)
    for ph in phases:
        vals = np.array([float(r["rmse_8ctrl_m"]) * 100 for r in cv if r["phase"] == ph and r.get("rmse_8ctrl_m")])
        if len(vals) < 5:
            continue
        cdf = np.mean(vals[None, :] <= xs[:, None], axis=1) * 100
        ax.plot(xs, cdf, color=PCOL[ph], lw=2.2, label=f"{ph} (n={len(vals)}, med={np.median(vals):.1f} cm)")
    ax.axvline(10, color="#888888", ls=":", lw=1.2); ax.text(10.3, 12, "10 cm", fontsize=9, color="#555555")
    ax.set_xlim(0, 35); ax.set_ylim(0, 100.5)
    ax.set_xlabel("8-control-point error (cm)", fontsize=11)
    ax.set_ylabel("cumulative fraction (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9); ax.grid(alpha=0.2, lw=0.5)


def panel_violin(ax, cv):
    data, labels, cols = [], [], []
    for ph in PHASES:
        vals = [float(r["rmse_8ctrl_m"]) * 100 for r in cv if r["phase"] == ph and r.get("rmse_8ctrl_m")]
        if vals:
            data.append(vals); labels.append(f"{ph}\n(n={len(vals)})"); cols.append(PCOL[ph])
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], cols):
        body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.55)
    parts["cmedians"].set_color("k"); parts["cmedians"].set_linewidth(1.8)
    for i, vals in enumerate(data, 1):
        ax.text(i, np.median(vals), f"  {np.median(vals):.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("8-control-point RMSE (cm)", fontsize=11)
    ax.set_title("(c)  Boundary error by phase", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.5)


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"saved {out}")


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    cv, ER, PR, EZ, PZ = load()
    print(f"frames={len(cv)}  pooled R² R={r2(ER,PR):.3f} Z={r2(EZ,PZ):.3f}")
    supt = "FreeGSNKE forward equilibrium validation — 100 best-predicted high-$\\beta_N$ test discharges"

    # ---- B: parity + CDF(2-phase) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    panel_parity(axes[0], ER, PR, "R", "(a)"); panel_parity(axes[1], EZ, PZ, "Z", "(b)")
    panel_cdf(axes[2], cv, ["flat_top", "ramp_down"], "(c)  Boundary error CDF (quasi-steady phases)")
    fig.suptitle(supt, fontsize=13, fontweight="bold", y=1.02); fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "freegsnke_var_b.png")

    # ---- C: parity + violin(3-phase) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    panel_parity(axes[0], ER, PR, "R", "(a)"); panel_parity(axes[1], EZ, PZ, "Z", "(b)")
    panel_violin(axes[2], cv)
    fig.suptitle(supt, fontsize=13, fontweight="bold", y=1.02); fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "freegsnke_var_c.png")

    # ---- D: parity only (2-panel big) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    panel_parity(axes[0], ER, PR, "R", "(a)"); panel_parity(axes[1], EZ, PZ, "Z", "(b)")
    fig.suptitle(supt, fontsize=13, fontweight="bold", y=1.0); fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "freegsnke_var_d.png")


if __name__ == "__main__":
    main()
