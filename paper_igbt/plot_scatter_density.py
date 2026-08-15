"""F1-diff —— per_channel_scatter 密度版（3 种风格同跑）+ 池化总览 jointplot。

一次运行同时产出 kde / hist / hex 三种风格的密度图（各自独立文件，互不覆盖）：
  · kde  —— 圆形高斯核连续密度场（直方图 + 高斯模糊，无瓦片无阶梯），默认推荐
  · hist —— 2D 直方图 + 双线性插值（低密度尾有阶梯，作对比）
  · hex  —— 六边形 hexbin（瓦片，原始风格）

每风格两张图:
  主图 3×4 网格（12 PF 通道，统一量程，红虚线 y=x，R²/MAE 角标，共享 log count 色条）
  附图   全通道池化 jointplot（中心密度 + 顶部/右侧边缘直方图；默认 per-channel z-score 池化）

独立脚本，不改动 plot_figures.py；数据加载/指标/通道名复用其函数以保持口径一致。

读:  results/paper_igbt/predictions/per_shot_preds.npz
写:  results/paper_igbt/figures/per_channel_scatter_{kde,hist,hex}.png
     results/paper_igbt/figures/pooled_scatter_joint_{kde,hist,hex}.png
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter

from plot_figures import PF_NAMES, _metrics_1d, load_records

C_ACCENT = "#4477AA"   # 边缘直方图配色
STYLES = ["kde", "hist", "hex"]


def _save(fig, out, dpi: int = 300):
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} + .pdf  (dpi={dpi})")


def _gather_channels(records, ch: int):
    """返回某通道在全量 mask 下的 (truth, pred)；指标应在这上面算。"""
    truths = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
    preds = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
    return truths, preds


def _pooled_arrays(records, normalize: bool):
    """全 12 通道池化的 (truth, pred)。

    normalize=True: 每通道按【真值】mean/std 做 z-score（truth 与 pred 用同一组 stats，
    是对两者的同一仿射变换 → 保持 per-channel R² 不变），再池化。
    这样池化 R² = per-channel R² 的 N 加权均值（公平整体指标）。
    normalize=False: 原始 kA 直接池化（会被通道间量级差抬高分母 SST，R² 虚高）。
    """
    if normalize:
        ts, ps = [], []
        for ch in range(12):
            t, p = _gather_channels(records, ch)
            mu, sd = float(t.mean()), max(float(t.std()), 1e-8)
            ts.append((t - mu) / sd)
            ps.append((p - mu) / sd)
        return np.concatenate(ts), np.concatenate(ps)
    truths = np.concatenate([r["target_kA"][r["action_mask"]] for r in records])
    preds = np.concatenate([r["pred_kA"][r["action_mask"]] for r in records])
    return truths, preds


def _print_r2_table(records):
    """打印 per-channel R²/MAE，便于核对（图上角标的来源）。"""
    r2s = []
    for ch in range(12):
        t, p = _gather_channels(records, ch)
        r2, mae = _metrics_1d(p, t)
        r2s.append(r2)
        print(f"  {PF_NAMES[ch]:6s} R²={r2:.3f}  MAE={mae:.3f} kA")
    print(f"  mean per-channel R² = {np.mean(r2s):.3f}")


def _density_main(ax, truths, preds, lo, hi, gridsize, style="kde", sigma=2.5,
                  norm=None, cmap="viridis"):
    """主区域密度图。返回 mappable。

    style:
      kde  —— 直方图 + 高斯模糊（binned KDE，圆形高斯核 → 连续无瓦片无阶梯）
      hist —— 2D 直方图 + 双线性插值（低密度尾有阶梯）
      hex  —— 六边形 hexbin（瓦片）
    """
    if style == "hex":
        return ax.hexbin(truths, preds, gridsize=gridsize, cmap=cmap, mincnt=1,
                         edgecolors="none", norm=norm)
    H, _, _ = np.histogram2d(truths, preds, bins=gridsize, range=[[lo, hi], [lo, hi]])
    H = H.astype(float)
    if style == "kde":
        H = gaussian_filter(H, sigma=sigma)          # 圆形高斯核 → 连续密度
    cmap = copy.copy(plt.get_cmap(cmap)); cmap.set_bad("white")
    H = np.ma.masked_where(H < 0.5, H)               # 空区透白，避免满屏深紫
    im = ax.imshow(H.T, origin="lower", extent=[lo, hi, lo, hi], aspect="auto", cmap=cmap,
                   norm=norm or LogNorm(vmin=1, vmax=max(float(H.max()), 2.0)),
                   interpolation="bilinear")
    return im


# ── 主图准备: 每通道抽样(绘图) + 指标(全量) + 统一量程，只算一次 ──
def _prep_grid(records, cap: int = 1_000_000):
    rng = np.random.default_rng(0)
    chans = []
    glo, ghi = np.inf, -np.inf
    for ch in range(12):
        truths, preds = _gather_channels(records, ch)
        r2, mae = _metrics_1d(preds, truths)
        glo = min(glo, float(min(truths.min(), preds.min())))
        ghi = max(ghi, float(max(truths.max(), preds.max())))
        if preds.size > cap:
            idx = rng.choice(preds.size, cap, replace=False)
            truths, preds = truths[idx], preds[idx]
        chans.append((truths, preds, r2, mae))
    pad = 0.02 * (ghi - glo)
    return chans, glo - pad, ghi + pad


def render_grid(chans, lo, hi, out, gridsize: int, style: str, sigma: float):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.07, top=0.95, wspace=0.28, hspace=0.32)

    colls, max_cnt = [], 1.0
    for ch in range(12):
        ax = axes[ch // 4, ch % 4]
        truths, preds, r2, mae = chans[ch]
        mappable = _density_main(ax, truths, preds, lo, hi, gridsize, style, sigma)
        colls.append(mappable)
        max_cnt = max(max_cnt, float(np.asarray(mappable.get_array()).max()))

        ax.plot([lo, hi], [lo, hi], "r--", lw=0.9)                 # y=x，统一跨度
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)                   # 统一量程
        ax.set_title(PF_NAMES[ch], fontsize=10)
        ax.grid(alpha=0.25)
        ax.text(0.03, 0.97, f"$R^2$={r2:.3f}\nMAE={mae:.2f} kA", transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray",
                                               boxstyle="round,pad=0.3"))

    norm = LogNorm(vmin=1, vmax=max_cnt)
    for mp in colls:
        mp.set_norm(norm)
    cax = fig.add_axes([0.905, 0.07, 0.018, 0.88])
    fig.colorbar(colls[-1], cax=cax, label="count")

    for ax in axes[-1]:
        ax.set_xlabel("Truth (kA)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Prediction (kA)")
    _save(fig, out)


# ── 附图准备: 池化 + 指标 + 抽样，只算一次 ──
def _prep_joint(records, normalize: bool, cap: int = 3_000_000):
    truths, preds = _pooled_arrays(records, normalize)
    unit = "z-score" if normalize else "kA"
    r2, mae = _metrics_1d(preds, truths)
    lo = float(min(truths.min(), preds.min()))
    hi = float(max(truths.max(), preds.max()))
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if preds.size > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(preds.size, cap, replace=False)
        tp, pp = truths[idx], preds[idx]
    else:
        tp, pp = truths, preds
    return tp, pp, r2, mae, unit, lo, hi, normalize


def render_joint(prep, out, gridsize: int, style: str, sigma: float):
    tp, pp, r2, mae, unit, lo, hi, normalize = prep
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          left=0.10, right=0.88, bottom=0.10, top=0.95,
                          wspace=0.04, hspace=0.04)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    mappable = _density_main(ax_main, tp, pp, lo, hi, gridsize, style, sigma)
    ax_main.plot([lo, hi], [lo, hi], "r--", lw=1.0)
    ax_main.set_xlim(lo, hi); ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel(f"Truth ({unit})")
    ax_main.set_ylabel(f"Prediction ({unit})")
    ax_main.grid(alpha=0.25)
    ax_main.text(0.03, 0.97, f"$R^2$={r2:.3f}\nMAE={mae:.2f} {unit}", transform=ax_main.transAxes,
                 va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray",
                                                 boxstyle="round,pad=0.3"))

    nb = 90
    ax_top.hist(tp, bins=nb, range=(lo, hi), color=C_ACCENT, alpha=0.85)
    ax_right.hist(pp, bins=nb, range=(lo, hi), color=C_ACCENT, alpha=0.85, orientation="horizontal")
    ax_top.set_ylabel("count"); ax_top.tick_params(labelbottom=False)
    ax_right.set_xlabel("count"); ax_right.tick_params(labelleft=False)
    for a in (ax_top, ax_right):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    cax = fig.add_axes([0.90, 0.10, 0.022, 0.78])
    fig.colorbar(mappable, cax=cax, label="count")

    tag = "per-channel z-scored" if normalize else "raw kA"
    fig.suptitle(f"All PF channels pooled ({tag})", y=0.985, fontsize=11)
    _save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="results/paper_igbt/predictions/per_shot_preds.npz")
    ap.add_argument("--out-dir", default="results/paper_igbt/figures")
    ap.add_argument("--gridsize", type=int, default=180, help="主图网格分辨率（附图自动 +40）")
    ap.add_argument("--style", choices=STYLES, default=None,
                    help="只生成指定风格；默认一次同跑 kde/hist/hex 全部三种")
    ap.add_argument("--sigma", type=float, default=2.5, help="kde 高斯核带宽(网格格数)，越大越平滑")
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="jointplot: 每通道按真值 z-score 后再池化（默认，公平整体 R²）；"
                         "--no-normalize 用原始 kA 池化（R² 会被通道间量级差抬高分母而虚高）")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(Path(args.preds))
    styles = [args.style] if args.style else STYLES
    print(f"loaded {len(recs)} records  | styles={styles} sigma={args.sigma} normalize={args.normalize}")
    print("per-channel R²/MAE:")
    _print_r2_table(recs)

    # 数据只准备一次，三种风格复用
    chans, glo, ghi = _prep_grid(recs)
    jprep = _prep_joint(recs, normalize=args.normalize)

    for s in styles:
        render_grid(chans, glo, ghi, out / f"per_channel_scatter_{s}.png",
                    gridsize=args.gridsize, style=s, sigma=args.sigma)
        render_joint(jprep, out / f"pooled_scatter_joint_{s}.png",
                     gridsize=args.gridsize + 40, style=s, sigma=args.sigma)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
