#!/usr/bin/env python3
"""合并 freegsnke_testset 的分片 per_frame_shard*.csv -> summary.json + 4 张统计图。

用法 (所有 shard 跑完后):
  python -m paper_igbt.aggregate_freegsnke_testset [--in-dir results/paper_igbt/freegsnke_testset]
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PHASE_NAMES = ["ramp_up", "flat_top", "ramp_down"]


def _load(in_dir: Path) -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(str(in_dir / "per_frame_shard*.csv"))):
        with open(fp) as f:
            rows.extend(csv.DictReader(f))
    return rows


def _f(rec, key):
    v = rec.get(key)
    if v is None or v == "":
        return None
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _mstats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": float(np.mean(xs)), "median": float(np.median(xs)),
            "p25": float(np.percentile(xs, 25)), "p75": float(np.percentile(xs, 75)),
            "p95": float(np.percentile(xs, 95)), "max": float(np.max(xs))}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="results/paper_igbt/freegsnke_testset")
    ap.add_argument("--efit-dir", default="/data/EFIT")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    rows = _load(in_dir)
    if not rows:
        sys.exit(f"no per_frame_shard*.csv in {in_dir}")
    print(f"[aggregate] {len(rows)} frames from {len(glob.glob(str(in_dir/'per_frame_shard*.csv')))} shards")

    rmse = [_f(r, "rmse_8ctrl_m") for r in rows]
    rel = [_f(r, "rel_change") for r in rows]
    cvg = [r.get("converged", "").strip().lower() == "true" for r in rows]
    summary = {
        "n_frames": len(rows),
        "converge_rate": (sum(cvg) / len(cvg)) if cvg else None,
        "method_counts": dict((m, sum(1 for r in rows if r.get("method") == m))
                              for m in set(r.get("method", "") for r in rows)),
        "rmse_8ctrl_m": _mstats(rmse),
        "rel_change": _mstats(rel),
        "rmse_by_phase": {ph: _mstats([_f(r, "rmse_8ctrl_m") for r in rows if r.get("phase") == ph])
                          for ph in PHASE_NAMES},
    }
    (in_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # 合并 per_frame.csv
    keys = sorted({k for r in rows for k in r})
    with open(in_dir / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) 收敛
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if rel:
        rl = [x for x in rel if x is not None]
        axes[0].hist(np.log10(rl), bins=50, color="steelblue", edgecolor="k")
        axes[0].axvline(np.log10(8e-3), color="r", ls="--", label="rtol=8e-3")
        axes[0].set_xlabel("log10(relative_change)"); axes[0].set_ylabel("#frames")
        axes[0].set_title("Convergence residual"); axes[0].legend()
    mc = summary["method_counts"]
    axes[1].bar(list(mc.keys()), list(mc.values()))
    axes[1].set_title(f"Solver method (conv rate={summary['converge_rate']:.1%})")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].bar(["converged", "not"], [sum(cvg), len(cvg) - sum(cvg)], color=["seagreen", "crimson"])
    axes[2].set_title("Converged count")
    plt.tight_layout(); plt.savefig(in_dir / "fig_convergence.png", dpi=130); plt.close()

    # 2) 8 控制点 RMSE 分布
    if any(rmse):
        rr = [x for x in rmse if x is not None]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rr, bins=60, color="mediumpurple", edgecolor="k")
        ax.set_xlabel("8-control-point Euclidean RMSE (m)"); ax.set_ylabel("#frames")
        ax.set_title(f"8-ctrl RMSE distribution (median={summary['rmse_8ctrl_m']['median']:.4f} m)")
        plt.tight_layout(); plt.savefig(in_dir / "fig_8ctrl_rmse.png", dpi=130); plt.close()

    # 3) 分阶段箱线
    data = [[_f(r, "rmse_8ctrl_m") for r in rows if r.get("phase") == ph and _f(r, "rmse_8ctrl_m") is not None]
            for ph in PHASE_NAMES]
    data = [d for d in data if d]
    if data:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, tick_labels=PHASE_NAMES[:len(data)], showfliers=False)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.uniform(-0.05, 0.05, len(d)), d, s=6, alpha=0.25, color="steelblue")
        ax.set_ylabel("8-ctrl RMSE (m)"); ax.set_title("FreeGSNKE 8-ctrl RMSE by phase")
        plt.tight_layout(); plt.savefig(in_dir / "fig_phase_rmse.png", dpi=130); plt.close()

    # 4) 形状参数 pred vs EFIT 散点
    efit_dir = Path(args.efit_dir)
    sp_keys = ["elong", "triu", "tril", "squo"]
    fig, axes = plt.subplots(1, len(sp_keys), figsize=(4 * len(sp_keys), 4))
    for ax, key in zip(axes, sp_keys):
        xs, ys = [], []
        for r in rows:
            pv = _f(r, f"pred_{key}")
            if pv is None:
                continue
            try:
                shot, k = int(r["shot"]), int(r["k"])
                with h5py.File(efit_dir / f"{shot}.h5", "r") as f:
                    ev = float(np.asarray(f[key.upper()][:]).ravel()[k])
            except Exception:
                continue
            xs.append(ev); ys.append(pv)
        if len(xs) > 1:
            ax.scatter(xs, ys, s=8, alpha=0.35)
            lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_title(f"{key} (r={np.corrcoef(xs, ys)[0,1]:.3f}, n={len(xs)})")
            ax.set_xlabel("EFIT"); ax.set_ylabel("FreeGSNKE(pred-BC)")
    plt.tight_layout(); plt.savefig(in_dir / "fig_shape_params.png", dpi=130); plt.close()
    print(f"[figs] wrote fig_*.png -> {in_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
