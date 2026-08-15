#!/usr/bin/env python3
"""单图: FreeGSNKE 8-控制点边界误差, 按三阶段 (ramp_up/flat_top/ramp_down) 统计。

美观 + 观点: 平台/下降期亚 10 cm (favorable), 上升期较难 (诚实)。
  - 半小提琴 + 内嵌箱线 + 抖动散点 (raincloud-lite)
  - 10 cm 参考线; 每相位标注 中位/均值 与 "<10cm 占比"
口径: betan 100 最佳预测炮, converged 帧。
读: results/paper_betan/freegsnke_testset/per_frame.csv
写: results/paper_betan/figures/freegsnke_phase_stats.{png,pdf}
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/paper_betan/freegsnke_testset/per_frame.csv"
OUT = Path("results/paper_betan/figures/freegsnke_phase_stats.png")
PHASES = ["ramp_up", "flat_top", "ramp_down"]
PLABEL = {"ramp_up": "Ramp-up", "flat_top": "Flat-top", "ramp_down": "Ramp-down"}
PCOL = {"ramp_up": "#D55E00", "flat_top": "#0072B2", "ramp_down": "#009E73"}  # Okabe-Ito, CVD-safe


def main():
    rows = [r for r in csv.DictReader(open(CSV))]
    cv = [r for r in rows if r["converged"] == "True" and r.get("rmse_8ctrl_m")]
    data = {ph: np.array([float(r["rmse_8ctrl_m"]) * 100 for r in cv
                          if r["phase"] == ph and r.get("rmse_8ctrl_m")]) for ph in PHASES}
    n_cvg = {ph: len(data[ph]) for ph in PHASES}
    n_tot = sum(1 for r in rows if r.get("rmse_8ctrl_m"))
    conv_rate = len(cv) / max(n_tot, 1)

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444", "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    pos = np.arange(1, len(PHASES) + 1)
    vparts = ax.violinplot([data[ph] for ph in PHASES], positions=pos, widths=0.75,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, ph in zip(vparts["bodies"], PHASES):
        body.set_facecolor(PCOL[ph]); body.set_edgecolor(PCOL[ph]); body.set_alpha(0.45); body.set_linewidth(1.0)

    # 内嵌箱线 + 抖动散点
    for i, ph in enumerate(PHASES, 1):
        v = data[ph]
        bp = ax.boxplot([v], positions=[i], widths=0.18, patch_artist=True,
                        showfliers=False, whis=[10, 90], zorder=4)
        bp["boxes"][0].set_facecolor("white"); bp["boxes"][0].set_edgecolor("#333"); bp["boxes"][0].set_linewidth(1.0)
        for m in bp["medians"]: m.set_color(PCOL[ph]); m.set_linewidth(2.2)
        jit = np.random.default_rng(7).uniform(-0.13, 0.13, size=len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=10, color=PCOL[ph], alpha=0.35, edgecolors="none", zorder=3)

    # 10 cm 参考线
    ax.axhline(10, color="#999999", ls="--", lw=1.1, zorder=1)
    ax.text(len(PHASES) + 0.55, 10, " 10 cm", va="center", color="#666", fontsize=10)

    # 每相位标注: 中位 + <10cm 占比
    for i, ph in enumerate(PHASES, 1):
        v = data[ph]; med = float(np.median(v)); pct = float(np.mean(v < 10) * 100)
        ax.text(i, ax.get_ylim()[1] if False else 36.5, f"median {med:.1f} cm\n{pct:.0f}% within 10 cm",
                ha="center", va="top", fontsize=9.5,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc", boxstyle="round,pad=0.3"))

    ax.set_ylim(0, 40)
    ax.set_xticks(pos)
    ax.set_xticklabels([PLABEL[ph] for ph in PHASES], fontsize=12)
    ax.set_ylabel("8-control-point boundary RMSE (cm)", fontsize=12)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    for ph in PHASES:
        v = data[ph]
        print(f"{ph:9s}: n={len(v):4d} median={np.median(v):5.2f} mean={np.mean(v):5.2f} cm  <10cm={np.mean(v<10)*100:4.0f}%")
    print(f"saved {OUT} + .pdf  (dpi=300)")


if __name__ == "__main__":
    main()
