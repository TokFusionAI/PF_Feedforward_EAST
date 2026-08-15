#!/usr/bin/env python3
"""单图: FreeGSNKE 8-控制点边界【形状】R², 按三阶段 (ramp_up/flat_top/ramp_down) 统计。

是 plot_freegsnke_phase_stats.py (RMSE 版) 的 R² 兄弟图。
  - R² 口径 = centered shape R² (逐帧): 每帧去掉质心(平移)后, 8 控制点形状方差被解释的比例.
    num  = Σ_i (dr_i - dr̄)² + (dz_i - dz̄)²          # 去平移后的形状畸变误差
    den  = Σ_i (R8e_i - R̄_e)² + (Z8e_i - Z̄_e)²      # EFIT 目标去质心形状方差
    R²   = 1 - num/den
    (去质心 = 移除 FreeGSNKE 系统平移偏移; 衡量纯形状吻合. 分母是空间跨度, 但分子已去平移
     所以不退化成 ≈1 —— 与"逐帧位置 R²"不同, 那个分母被空间跨度主导会退化.)
  - EFIT 目标 R8/Z8 从 h5 按 CSV 的 k 索引读回 (CSV 只存了 dr_i/dz_i 偏移).
口径: betan 100 最佳预测炮, converged 帧. 半小提琴+内嵌箱线+抖动散点 (raincloud-lite).
读: results/paper_betan/freegsnke_testset/per_frame.csv
写: results/paper_betan/figures/freegsnke_phase_r2.{png,pdf}
"""
from __future__ import annotations
import csv
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/paper_betan/freegsnke_testset/per_frame.csv"
EFIT_DIR = Path("/data/EFIT")
OUT = Path("results/paper_betan/figures/freegsnke_phase_r2.png")
PHASES = ["ramp_up", "flat_top", "ramp_down"]
PLABEL = {"ramp_up": "Ramp-up", "flat_top": "Flat-top", "ramp_down": "Ramp-down"}
PCOL = {"ramp_up": "#D55E00", "flat_top": "#0072B2", "ramp_down": "#009E73"}  # Okabe-Ito, CVD-safe


def load_r8z8(shot: int):
    """复现 freegsnke_testset_eval.bundle_from_efit_h5 的 R8/Z8 读取 (去 ATIME<0, 保 (T,8))."""
    fp = EFIT_DIR / f"{int(shot)}.h5"
    with h5py.File(fp, "r") as f:
        atime = np.asarray(f["ATIME"][:], dtype=np.float64).ravel()
        m = atime >= 0.0
        T = int(m.sum())

        def take(k):
            a = np.asarray(f[k][:], dtype=np.float64)
            if a.shape[0] == atime.size:
                return a[m]
            if a.shape[0] == T:
                return a
            raise ValueError(f"{k} shape {a.shape} vs ATIME {atime.size}/{T}")

        R8 = take("R8"); Z8 = take("Z8")
        if R8.ndim == 2 and R8.shape[1] != 8 and R8.shape[0] == 8:
            R8 = R8.T; Z8 = Z8.T
    return R8, Z8


def centered_shape_r2(r8e, z8e, dr, dz) -> float:
    """去质心形状 R²: 平移无关, 衡量纯形状吻合."""
    num = float(np.sum((dr - dr.mean()) ** 2 + (dz - dz.mean()) ** 2))
    tcR = r8e - r8e.mean(); tcZ = z8e - z8e.mean()
    den = float(np.sum(tcR ** 2 + tcZ ** 2))
    return 1.0 - num / den if den > 0 else float("nan")


def main():
    rows = [r for r in csv.DictReader(open(CSV))]
    cv = [r for r in rows if r["converged"] == "True" and r.get("dr_0") and r.get("dz_0")]
    cache: dict[int, tuple] = {}
    data = {ph: [] for ph in PHASES}
    n_skip = 0
    for r in cv:
        shot, k, ph = int(r["shot"]), int(r["k"]), r["phase"]
        if shot not in cache:
            cache[shot] = load_r8z8(shot)
        R8, Z8 = cache[shot]
        if not (0 <= k < R8.shape[0]):
            n_skip += 1; continue
        dr = np.array([float(r[f"dr_{i}"]) for i in range(8)])
        dz = np.array([float(r[f"dz_{i}"]) for i in range(8)])
        r2 = centered_shape_r2(R8[k], Z8[k], dr, dz)
        if np.isfinite(r2):
            data[ph].append(r2)
    data = {ph: np.array(data[ph]) for ph in PHASES}

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444", "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    pos = np.arange(1, len(PHASES) + 1)

    vparts = ax.violinplot([data[ph] for ph in PHASES], positions=pos, widths=0.75,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, ph in zip(vparts["bodies"], PHASES):
        body.set_facecolor(PCOL[ph]); body.set_edgecolor(PCOL[ph]); body.set_alpha(0.45); body.set_linewidth(1.0)

    for i, ph in enumerate(PHASES, 1):
        v = data[ph]
        bp = ax.boxplot([v], positions=[i], widths=0.18, patch_artist=True,
                        showfliers=False, whis=[10, 90], zorder=4)
        bp["boxes"][0].set_facecolor("white"); bp["boxes"][0].set_edgecolor("#333"); bp["boxes"][0].set_linewidth(1.0)
        for mm in bp["medians"]: mm.set_color(PCOL[ph]); mm.set_linewidth(2.2)
        jit = np.random.default_rng(7).uniform(-0.13, 0.13, size=len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=10, color=PCOL[ph], alpha=0.35, edgecolors="none", zorder=3)

    # R²=1 参考线 (完美) + 0.9 (形状吻合良好)
    ax.axhline(1.0, color="#999999", ls="--", lw=1.0, zorder=1)
    ax.axhline(0.9, color="#bbbbbb", ls=":", lw=0.9, zorder=1)
    ax.text(len(PHASES) + 0.55, 1.0, " 1.0", va="center", color="#666", fontsize=10)

    yhi = 1.18
    for i, ph in enumerate(PHASES, 1):
        v = data[ph]; med = float(np.median(v)); pct = float(np.mean(v >= 0.9) * 100)
        ax.text(i, yhi - 0.02, f"median {med:.2f}\n{pct:.0f}% ≥ 0.90",
                ha="center", va="top", fontsize=9.5,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc", boxstyle="round,pad=0.3"))

    ax.set_ylim(0.0, yhi)
    ax.set_xticks(pos)
    ax.set_xticklabels([PLABEL[ph] for ph in PHASES], fontsize=12)
    ax.set_ylabel("8-control-point boundary  shape  $R^2$", fontsize=12)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    allv = np.concatenate([data[ph] for ph in PHASES])
    for ph in PHASES:
        v = data[ph]
        print(f"{ph:9s}: n={len(v):4d} median={np.median(v):.3f} mean={np.mean(v):.3f} "
              f"min={v.min():.3f} max={v.max():.3f}  ≥0.9={np.mean(v>=0.9)*100:3.0f}%")
    print(f"ALL       : n={len(allv):4d} median={np.median(allv):.3f}  (skipped {n_skip} oob-k)")
    print(f"saved {OUT} + .pdf  (dpi=300)")


if __name__ == "__main__":
    main()
