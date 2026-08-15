#!/usr/bin/env python3
"""探索图: 8 控制点 pred-vs-EFIT parity (R, Z) + 逐点 R²。看 R² 表征长什么样。

  (a) R parity: pred R vs EFIT R (hexbin, y=x), 池化 R² + 逐点中位 R²
  (b) Z parity: 同上
  (c) 逐控制点 R² 柱状 (R 蓝 / Z 橙)

读: results/paper_betan/freegsnke_testset/per_frame.csv + EFIT h5 (R8/Z8) + dr_i/dz_i
写: results/paper_betan/figures/ctrlpt_parity.{png,pdf}
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

EFIT = "/data/EFIT"
CSV = "results/paper_betan/freegsnke_testset/per_frame.csv"
OUT = "results/paper_betan/figures/ctrlpt_parity.png"


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
    ER, PR, EZ, PZ = [], [], [], []
    per_pt = {i: {"er": [], "pr": [], "ez": [], "pz": []} for i in range(8)}  # per-point pools
    for r in cv:
        try:
            er, ez = get_r8z8(int(r["shot"]), int(r["k"]))
        except Exception:
            continue
        if er.size != 8:
            continue
        dr = np.array([float(r[f"dr_{i}"]) for i in range(8)])
        dz = np.array([float(r[f"dz_{i}"]) for i in range(8)])
        pr = er + dr; pz = ez + dz
        ER.append(er); PR.append(pr); EZ.append(ez); PZ.append(pz)
        for i in range(8):
            per_pt[i]["er"].append(er[i]); per_pt[i]["pr"].append(pr[i])
            per_pt[i]["ez"].append(ez[i]); per_pt[i]["pz"].append(pz[i])
    ER = np.concatenate(ER); PR = np.concatenate(PR); EZ = np.concatenate(EZ); PZ = np.concatenate(PZ)
    n = len(per_pt[0]["er"])
    print(f"frames={n}, points={ER.size}")

    r2R_pool, r2Z_pool = r2(ER, PR), r2(EZ, PZ)
    r2R_pt = np.array([r2(per_pt[i]["er"], per_pt[i]["pr"]) for i in range(8)])
    r2Z_pt = np.array([r2(per_pt[i]["ez"], per_pt[i]["pz"]) for i in range(8)])
    print(f"pooled R²: R={r2R_pool:.3f}  Z={r2Z_pool:.3f}")
    print(f"per-point median R²: R={np.median(r2R_pt):.3f}  Z={np.median(r2Z_pt):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    CMAP = "Purples"
    for ax, x, y, name, rpool, rpt in (
        (axes[0], ER, PR, "R (m)", r2R_pool, r2Z_pt),
        (axes[1], EZ, PZ, "Z (m)", r2Z_pool, r2Z_pt),
    ):
        hb = ax.hexbin(x, y, gridsize=50, cmap=CMAP, mincnt=1, norm=matplotlib.colors.LogNorm())
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#D55E00", lw=1.3, label="y = x")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        which_R2 = r2R_pool if name.startswith("R") else r2Z_pool
        med = np.median(r2R_pt if name.startswith("R") else rpt)
        ax.text(0.03, 0.97, f"EFIT vs FreeGSNKE(pred)  {name}\n"
                f"pooled R² = {which_R2:.3f}\nper-point median R² = {med:.3f}\nn = {len(x)}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.3"))
        ax.set_xlabel(f"EFIT {name}", fontsize=10); ax.set_ylabel(f"FreeGSNKE(pred) {name}", fontsize=10)
        ax.set_title(f"{'(a)' if name.startswith('R') else '(b)'} Control-point {name[0]} parity", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2, lw=0.5); ax.legend(fontsize=8, loc="lower right")
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04); cb.set_label("# frames", fontsize=8)

    ax = axes[2]
    w = 0.38; x = np.arange(8)
    ax.bar(x - w / 2, r2R_pt, w, color="#0072B2", label="R")
    ax.bar(x + w / 2, r2Z_pt, w, color="#D55E00", label="Z")
    ax.axhline(np.median(r2R_pt), color="#0072B2", ls=":", lw=1, alpha=0.6)
    ax.axhline(np.median(r2Z_pt), color="#D55E00", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in range(8)])
    ax.set_xlabel("control point index", fontsize=10)
    ax.set_ylabel("$R^2$ (pred vs EFIT, per point)", fontsize=10)
    ax.set_title(f"(c) Per-control-point $R^2$\n(median R={np.median(r2R_pt):.3f}, Z={np.median(r2Z_pt):.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(min(0, np.nanmin(r2R_pt) - 0.05), 1.02); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=9)

    fig.suptitle("Control-point parity: FreeGSNKE(pred currents) vs EFIT — 100 best-predicted high-$\\beta_N$ shots",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(OUT); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out} + .pdf")


if __name__ == "__main__":
    main()
