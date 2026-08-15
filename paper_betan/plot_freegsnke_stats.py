#!/usr/bin/env python3
"""FreeGSNKE 统计验证图 (3 逻辑面板 / 2×2 物理布局), betan 100-best 炮口径。

只画 bc_pred: 预测 PF 电流 → FreeGSNKE 求解 → 8 控制点 vs EFIT R8/Z8。
统计只建立在 converged 片上 (每炮过采样后选收敛时间片)。

  (a) 8-控制点 RMSE 按相位分布 (小提琴, ramp_up/flat_top/ramp_down, 标中位)
  (b) 逐控制点 ΔR / ΔZ 分布 (分组箱线, x=控制点 1..8, 零线看偏置)
  (c) 拉长率 κ parity: FreeGSNKE(pred) elong vs EFIT elong (hexbin, y=x)
  (d) 上三角度 δ parity: FreeGSNKE(pred) triu vs EFIT triu (hexbin, y=x)

配色: Okabe-Ito (CVD-safe), 密度用 Purples; 风格对齐 paper 现有图 (dpi 200, Agg, tight)。
⚠️ 数据为"100 best-predicted"炮, 图注须声明 (选择偏差)。

用法:
  python -m paper_betan.plot_freegsnke_stats \
      --in-csv results/paper_betan/freegsnke_testset/per_frame.csv \
      --out results/paper_betan/figures/freegsnke_stats.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PHASE_NAMES = ["ramp_up", "flat_top", "ramp_down"]
PHASE_COLORS = {"ramp_up": "#D55E00", "flat_top": "#0072B2", "ramp_down": "#009E73"}  # Okabe-Ito
C_DR, C_DZ = "#0072B2", "#D55E00"   # ΔR / ΔZ
CMAP = "Purples"


def _save(fig, out: Path, dpi: int = 200):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} + .pdf  (dpi={dpi})")


def _metrics(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    # Pearson r (相关, 对系统偏置免疫) + MAE (偏置幅度)。不用 R²: FreeGSNKE 形状参数相对
    # EFIT 有系统偏移 (粗网格/定义差), R² 对带偏置的预测会变负, 误导。
    r = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(y - x)))
    bias = float(np.mean(y - x))
    return {"r": r, "mae": mae, "bias": bias, "n": len(x)}


def _efit_shape_cache(shot, efit_dir, cache):
    """惰性加载某炮 EFIT elong/triu (应用 atime>=0 mask), 缓存。"""
    if shot in cache:
        return cache[shot]
    import h5py
    out = None
    try:
        with h5py.File(Path(efit_dir) / f"{int(shot)}.h5", "r") as f:
            at = np.asarray(f["ATIME"][:], dtype=np.float64).ravel()
            m = at >= 0.0

            def take(k):
                a = np.asarray(f[k][:], dtype=np.float64).ravel()
                return a[m] if a.size == at.size else a
            out = {"elong": take("elong"), "triu": take("triu")}
    except Exception:
        out = {"elong": None, "triu": None}
    cache[shot] = out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-csv", default="results/paper_betan/freegsnke_testset/per_frame.csv")
    ap.add_argument("--efit-dir", default="/data/EFIT")
    ap.add_argument("--out", default="results/paper_betan/figures/freegsnke_stats.png")
    args = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(args.in_csv)))
    if not rows:
        sys.exit(f"no rows in {args.in_csv}")
    # 只保留 converged 且有 rmse 的帧
    cv = [r for r in rows if r.get("converged", "").strip().lower() == "true"
          and r.get("rmse_8ctrl_m") not in (None, "")]
    n_tot = len(rows)
    print(f"[plot] {len(cv)} converged frames / {n_tot} total "
          f"(converge rate {len(cv)/n_tot:.1%})")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ---- (a) 8-ctrl RMSE 按相位 ----
    data_ph = []
    labels_ph = []
    for ph in PHASE_NAMES:
        xs = [float(r["rmse_8ctrl_m"]) * 100 for r in cv if r.get("phase") == ph]
        if xs:
            data_ph.append(xs)
            labels_ph.append(f"{ph}\n(n={len(xs)})")
    if data_ph:
        parts = ax_a.violinplot(data_ph, showmedians=True, showextrema=False)
        for body, ph in zip(parts["bodies"], [p for p in PHASE_NAMES if any(r.get("phase") == p for r in cv)]):
            body.set_facecolor(PHASE_COLORS[ph]); body.set_edgecolor(PHASE_COLORS[ph]); body.set_alpha(0.55)
        if "cmedians" in parts:
            parts["cmedians"].set_color("k"); parts["cmedians"].set_linewidth(1.8)
        # 中位数标注
        for i, xs in enumerate(data_ph, 1):
            md = float(np.median(xs))
            ax_a.text(i, md, f"  {md:.1f}", va="center", ha="left", fontsize=8, fontweight="bold")
    ax_a.set_xticks(range(1, len(labels_ph) + 1)); ax_a.set_xticklabels(labels_ph, fontsize=9)
    ax_a.set_ylabel("8-control-point RMSE (cm)", fontsize=10)
    ax_a.set_title("(a) Boundary error by phase", fontsize=11, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.25, lw=0.5)
    all_rmse = [float(r["rmse_8ctrl_m"]) * 100 for r in cv]
    ax_a.text(0.02, 0.98, f"converged: {len(cv)}/{n_tot} ({len(cv)/n_tot:.0%})\n"
              f"overall median: {np.median(all_rmse):.1f} cm",
              transform=ax_a.transAxes, va="top", ha="left", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3"))

    # ---- (b) 逐控制点 ΔR / ΔZ (signed, cm) ----
    pts = list(range(8))
    dr_all = np.array([[float(r[f"dr_{i}"]) * 100 for r in cv] for i in pts]) if cv else np.empty((8, 0))
    dz_all = np.array([[float(r[f"dz_{i}"]) * 100 for r in cv] for i in pts]) if cv else np.empty((8, 0))
    w = 0.38
    x = np.arange(8)
    bp1 = ax_b.boxplot([dr_all[i] for i in range(8)], positions=x - w / 2, widths=w,
                       patch_artist=True, showfliers=False, whis=[5, 95])
    bp2 = ax_b.boxplot([dz_all[i] for i in range(8)], positions=x + w / 2, widths=w,
                       patch_artist=True, showfliers=False, whis=[5, 95])
    for b in bp1["boxes"]: b.set_facecolor(C_DR); b.set_alpha(0.55); b.set_edgecolor(C_DR)
    for b in bp2["boxes"]: b.set_facecolor(C_DZ); b.set_alpha(0.55); b.set_edgecolor(C_DZ)
    for bp in (bp1, bp2):
        for med in bp["medians"]:
            med.set_color("k"); med.set_linewidth(1.3)
    ax_b.axhline(0, color="gray", lw=0.8, ls="-")
    ax_b.set_xticks(x); ax_b.set_xticklabels([str(i + 1) for i in range(8)], fontsize=9)
    ax_b.set_xlabel("control point index", fontsize=10)
    ax_b.set_ylabel("displacement vs EFIT (cm)", fontsize=10)
    ax_b.set_title("(b) Per-control-point error", fontsize=11, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.25, lw=0.5)
    ax_b.legend([bp1["boxes"][0], bp2["boxes"][0]], [r"$\Delta R$", r"$\Delta Z$"],
                fontsize=9, loc="upper right")

    # ---- (c)(d) κ / δ parity ----
    cache = {}
    pe_ke, te_ke, pe_tr, te_tr = [], [], [], []   # pred/efit elong, pred/efit triu
    for r in cv:
        pe = r.get("pred_elong")
        ptr = r.get("pred_triu")
        if pe in (None, "") or ptr in (None, ""):
            continue
        sp = _efit_shape_cache(int(r["shot"]), args.efit_dir, cache)
        if sp["elong"] is None or sp["triu"] is None:
            continue
        k = int(r["k"])
        if k >= len(sp["elong"]) or k >= len(sp["triu"]):
            continue
        te, ttr = float(sp["elong"][k]), float(sp["triu"][k])
        if not (np.isfinite(te) and np.isfinite(ttr)):
            continue
        pe_ke.append(float(pe)); te_ke.append(te)
        pe_tr.append(float(ptr)); te_tr.append(ttr)

    for ax, xs, ys, name, sym in ((ax_c, te_ke, pe_ke, r"$\kappa$ (elongation)", "(c)"),
                                  (ax_d, te_tr, pe_tr, r"$\delta_{\mathrm{upper}}$ (triangularity)", "(d)")):
        if len(xs) >= 2:
            hb = ax.hexbin(xs, ys, gridsize=45, cmap=CMAP, mincnt=1,
                           norm=matplotlib.colors.LogNorm())
            lo = float(min(min(xs), min(ys))); hi = float(max(max(xs), max(ys)))
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#D55E00", lw=1.3, label="y = x")
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
            m = _metrics(xs, ys)
            txt = f"{name}\nr={m['r']:.3f}\nMAE={m['mae']:.3f}\nbias={m['bias']:+.3f}\nn={m['n']}"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3"))
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("# frames", fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.set_xlabel(f"EFIT {name}", fontsize=10)
        ax.set_ylabel(f"FreeGSNKE(pred) {name}", fontsize=10)
        ax.set_title(f"{sym} Shape parity", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2, lw=0.5); ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("FreeGSNKE forward equilibrium validation — 100 best-predicted high-$\\beta_N$ test discharges",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, Path(args.out))

    # ---- 控制台统计 (供正文引用) ----
    print(f"[stat] 8-ctrl RMSE(cm): overall median={np.median(all_rmse):.2f} "
          f"p25={np.percentile(all_rmse,25):.2f} p75={np.percentile(all_rmse,75):.2f}")
    for ph in PHASE_NAMES:
        xs = [float(r["rmse_8ctrl_m"]) * 100 for r in cv if r.get("phase") == ph]
        if xs:
            print(f"  {ph:9s}: n={len(xs):4d} median={np.median(xs):.2f} cm "
                  f"p95={np.percentile(xs,95):.2f}")
    mk = _metrics(te_ke, pe_ke)
    mt = _metrics(te_tr, pe_tr)
    if mk: print(f"[stat] elong parity: r={mk['r']:.3f} MAE={mk['mae']:.3f} bias={mk['bias']:+.3f} n={mk['n']}")
    if mt: print(f"[stat] triu parity:  r={mt['r']:.3f} MAE={mt['mae']:.3f} bias={mt['bias']:+.3f} n={mt['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
