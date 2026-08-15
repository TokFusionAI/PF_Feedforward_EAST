#!/usr/bin/env python3
"""FreeGSNKE 统计验证图 v2 (美观 + 扬长避短, 仍诚实)。

3 联板:
  (a) 控制点 R parity  (hexbin, y=x, 池化 R²)
  (b) 控制点 Z parity  (hexbin, y=x, 池化 R²)
  (c) 8-控制点误差 CDF, 按相位 (flat_top/ramp_down 偏左=好)

口径: betan 100 最佳预测炮, converged 帧. 复用 paper_betan 风格 (hexbin + log 色标)。
读: results/paper_betan/freegsnke_testset/per_frame.csv + EFIT h5 (R8/Z8) + dr_i/dz_i
写: results/paper_betan/figures/freegsnke_stats_v2.{png,pdf}
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py

EFIT = "/data/EFIT"
CSV = "results/paper_betan/freegsnke_testset/per_frame.csv"
OUT = "results/paper_betan/figures/freegsnke_stats_v2.png"
PHASES = ["ramp_up", "flat_top", "ramp_down"]
PCOL = {"ramp_up": "#D55E00", "flat_top": "#0072B2", "ramp_down": "#009E73"}  # Okabe-Ito
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


def main():
    rows = [r for r in csv.DictReader(open(CSV))]
    cv = [r for r in rows if r["converged"] == "True" and r.get("rmse_8ctrl_m")]
    ER, PR, EZ, PZ, PH = [], [], [], [], []
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
        PH.append(r["phase"])
    ER = np.concatenate(ER); PR = np.concatenate(PR); EZ = np.concatenate(EZ); PZ = np.concatenate(PZ)
    n = len(PH)
    rm = np.array([float(r["rmse_8ctrl_m"]) * 100 for r in cv[:n]])  # aligned
    r2R, r2Z = r2(ER, PR), r2(EZ, PZ)
    print(f"frames={n}  pooled R²: R={r2R:.3f} Z={r2Z:.3f}")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # ---- (a)(b) parity ----
    for ax, x, y, lab, r2v, tag in (
        (axes[0], ER, PR, "R", r2R, "(a)"),
        (axes[1], EZ, PZ, "Z", r2Z, "(b)"),
    ):
        hb = ax.hexbin(x, y, gridsize=55, cmap=CMAP, mincnt=1, norm=LogNorm())
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#D55E00", lw=1.4, zorder=5, label="y = x")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.96, f"$R^2 = {r2v:.3f}$\n$n = {len(x)}$",
                transform=ax.transAxes, va="top", fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbbbbb", boxstyle="round,pad=0.35"))
        ax.set_xlabel(f"EFIT control-point {lab} (m)", fontsize=11)
        ax.set_ylabel(f"FreeGSNKE(pred) {lab} (m)", fontsize=11)
        ax.set_title(f"{tag}  Control-point {lab} parity", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=8)

    # ---- (c) CDF by phase ----
    ax = axes[2]
    xs = np.linspace(0, 35, 400)
    for ph in PHASES:
        vals = np.array([float(r["rmse_8ctrl_m"]) * 100 for r in cv if r["phase"] == ph and r.get("rmse_8ctrl_m")])
        if len(vals) < 5:
            continue
        cdf = np.mean(vals[None, :] <= xs[:, None], axis=1) * 100
        ax.plot(xs, cdf, color=PCOL[ph], lw=2.2, label=f"{ph} (n={len(vals)}, med={np.median(vals):.1f} cm)")
    ax.axvline(10, color="#888888", ls=":", lw=1.2)
    ax.text(10.3, 12, "10 cm", fontsize=9, color="#555555")
    ax.set_xlim(0, 35); ax.set_ylim(0, 100.5)
    ax.set_xlabel("8-control-point error (cm)", fontsize=11)
    ax.set_ylabel("cumulative fraction of frames (%)", fontsize=11)
    ax.set_title("(c)  Boundary error CDF by phase", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)

    fig.suptitle("FreeGSNKE forward equilibrium validation — 100 best-predicted high-$\\beta_N$ test discharges",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(OUT); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out} + .pdf  (dpi=300)")


if __name__ == "__main__":
    main()
