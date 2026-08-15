#!/usr/bin/env python3
"""betan 退化图 R² 版: per-shot R²(逐通道中位) 中位 vs β_N (7 等频箱).

是 plot_betan_degradation.py 的 R² 兄弟图 (主图 betan_degradation_rmse7 的 R² 对应物).
口径 (关键, 两处不能错):
  ① per-shot R² 用【逐通道中位】, 不 pooled —— pooled 跨通道方差(PF 12 线圈尺度差异巨大)主导,
     每炮 R² 都≈1 不随 β_N 变, 退化无信息 (与 table r2_median / 坑1 同口径).
  ② 与 RMSE 退化图同: 用 per-shot R² 的箱【中位】(鲁棒, 不被长尾高误差炮主导),
     不用 bin-level pooled R². 3 seed 平均后再分箱取中位.
per-shot R² = 箱内逐炮: 逐通道 R²(掩码时间步) 取中位 → 每炮一个值, 3 seed 平均.
轴: discharge-mean β_N (betan 划分轴; cut=0.8; 退化 = 离 train 区间越远, R² 越低).
模型: proposed = transformer_bidir_on, seed [11,44,20260424].

输出:
  results/betan_ablation/figures/betan_degradation_r2_7.{png,pdf}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
CFG = "transformer_bidir_on"
SEEDS = [11, 44, 20260424]
SPLIT = ROOT / "meta/split_by_order_betan"
CACHE = SPLIT / "betan_maxmean_cache.npz"
CUT = 0.8
FIG_DIR = ROOT / "results/betan_ablation/figures"
# test 集 per-shot PF6 R²<0 炮清单 (paper figure 同判据; 由 paper_betan/plot_scatter_density.py
# --drop-pf6-neg-r2 生成). 剔除后 = 仅保留 PF6 R²≥0 的炮.
PF6_CSV = ROOT / "results/paper_betan/figures/_pf6_neg_r2_shots.csv"

C_MAIN = "#2a6db0"; C_REF = "#52514e"; C_CUT = "#e34948"


def per_shot_r2(pred, tgt, mask) -> np.ndarray:
    """每炮 R² = 逐通道 R²(掩码时间步) 的中位. 返回 (n_shots,)."""
    nshot = pred.shape[0]
    out = np.full(nshot, np.nan)
    for i in range(nshot):
        r2s = []
        for c in range(pred.shape[2]):
            mc = mask[i, :, c]
            if mc.sum() < 2:
                continue
            p, t = pred[i, :, c][mc], tgt[i, :, c][mc]
            sst = float(((t - t.mean()) ** 2).sum())
            if sst > 0:
                r2s.append(1.0 - float(((p - t) ** 2).sum()) / sst)
        out[i] = float(np.median(r2s)) if r2s else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="betan R² vs β_N (per-shot R² bin mean).")
    ap.add_argument("--bins", type=int, default=7, help="等频箱数 (默认 7)")
    ap.add_argument("--out", default="betan_degradation_r2_7", help="输出文件名 stem (无扩展)")
    ap.add_argument("--ylabel", default="Bin-mean per-shot $R^2$")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean",
                    help="第二层箱内跨炮聚合 (默认 mean)")
    ap.add_argument("--drop-pf6-neg-r2", action="store_true",
                    help="剔 test 集 per-shot PF6 R²<0 炮 (仅留 PF6 R²≥0), 口径同 paper figure")
    args = ap.parse_args()
    n_bins, out, ylabel = args.bins, FIG_DIR / args.out, args.ylabel
    agg_fn = np.median if args.agg == "median" else np.mean
    agg_word = args.agg          # "mean" | "median" (第二层; 第一层逐通道中位不变)

    c = np.load(CACHE, allow_pickle=True)
    bn_map = {int(s): float(mn) for s, mn in zip(c["shots"], c["betan_mean"])}
    pshot_all, shots = [], None
    for s in SEEDS:
        f = np.load(ROOT / "results/betan_ablation" / CFG / f"{CFG}_betan_s{s}" / "per_shot_preds.npz",
                    allow_pickle=True)
        shots = f["shots"].astype(int) if shots is None else shots
        p, t, m = f["pred_kA"].astype(float), f["target_kA"].astype(float), f["action_mask"].astype(bool)
        pshot_all.append(per_shot_r2(p, t, m))
    pshot = np.nanmean(pshot_all, axis=0)                       # per-shot R² (3-seed avg)
    bn = np.array([bn_map[int(s)] for s in shots], dtype=float)
    if args.drop_pf6_neg_r2:
        drop = np.loadtxt(PF6_CSV, delimiter=",", skiprows=1, usecols=0).astype(int)
        keep = ~np.isin(shots, drop)
        nd = int((~keep).sum())
        pshot, bn, shots = pshot[keep], bn[keep], shots[keep]
        print(f"dropped {nd} per-shot PF6 R²<0 shots -> {len(bn)} remain (PF6 R²≥0 only)")
    overall = float(agg_fn(pshot))

    # N 等频箱: per-shot R² 的箱内 {agg_word} (第二层跨炮聚合)
    edges = np.unique(np.quantile(bn, np.linspace(0, 1, n_bins + 1))); nb = len(edges) - 1
    bidx = np.searchsorted(edges[1:-1], bn, side="right")
    centers = np.array([np.median(bn[bidx == k]) for k in range(nb)])
    vals = np.array([agg_fn(pshot[bidx == k]) for k in range(nb)])

    # 近区/远区 (按 test β_N 中位分) — R² 应 near > far (退化)
    med_split = float(np.median(bn))
    near = pshot[bn < med_split]; far = pshot[bn >= med_split]
    near_v, far_v = float(agg_fn(near)), float(agg_fn(far))
    _, pval = st.mannwhitneyu(near, far, alternative="greater")   # near 的 R² 更高

    print(f"per-shot R² (per-channel median): overall {agg_word}={overall:.3f}")
    print(f"{nb} 箱 {agg_word}: " + ", ".join(f"{v:.3f}" for v in vals))
    print(f"近区(β_N<{med_split:.3f}, n={len(near)}): R² {agg_word}={near_v:.3f}")
    print(f"远区(β_N≥{med_split:.3f}, n={len(far)}): R² {agg_word}={far_v:.3f}  Δ={far_v-near_v:+.3f}  p={pval:.1e}")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 11})

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax2 = ax.twinx()                                            # density 背景 (隐藏轴)
    dens_n, _, _ = ax2.hist(bn, bins=30, density=True, color="#d8e2ee", edgecolor="#bccbdc",
                            linewidth=0.4, alpha=0.55, zorder=1)
    ax2.set_ylim(0, dens_n.max() * 1.5)
    ax2.set_yticks([]); ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
    ax.plot(centers, vals, "-o", color=C_MAIN, lw=2.6, ms=9, zorder=5,
            label=f"Per-shot $R^2$ (bin {agg_word}, {nb} bins)")
    ax.axhline(overall, color=C_REF, ls="--", lw=1.1, zorder=2,
               label=f"Overall {agg_word} ({overall:.3f})")
    ax.axvline(CUT, color=C_CUT, ls=":", lw=1.0, alpha=0.55, zorder=1, label="Train/test split")
    ax.set_xlabel("Discharge-mean $\\beta_N$")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#ececea", lw=0.6)
    dens_patch = mpatches.Patch(facecolor="#d8e2ee", edgecolor="#bccbdc", label="Sample density")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [dens_patch], l + ["Sample density"], loc="upper right",
              fontsize=9, framealpha=0.92)
    for ext in ("png", "pdf"):
        fig.savefig(out.parent / f"{out.name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.parent}/{out.name}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
