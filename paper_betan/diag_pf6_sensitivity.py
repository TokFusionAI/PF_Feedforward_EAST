"""修正版敏感性: 正确地按 PF6 单炮 R² 升序丢最差的 k 炮, 看 PF6 全局 R² 曲线。
(+ 若 betan_per_shot.parquet 可读, 报 worst 炮的 β_N, 确认是否高 β_N OOD 区)"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import _metrics_1d, load_records  # noqa

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
PF6 = 5


def main():
    recs = load_records(PRED)
    # 单炮 PF6 r2
    per = []
    for r in recs:
        m = r["action_mask"][:, PF6]
        if m.sum() == 0:
            per.append((r["shot"], 1.0)); continue
        p = r["pred_kA"][m, PF6]; t = r["target_kA"][m, PF6]
        per.append((r["shot"], float(_metrics_1d(p, t)[0])))
    per.sort(key=lambda x: x[1])  # r2 升序: 最差在前
    shots_sorted = [s for s, _ in per]

    def pf6_r2(drop):
        keep = set(shots_sorted[drop:])
        ts = np.concatenate([r["target_kA"][r["action_mask"][:, PF6], PF6] for r in recs if r["shot"] in keep])
        ps = np.concatenate([r["pred_kA"][r["action_mask"][:, PF6], PF6] for r in recs if r["shot"] in keep])
        return _metrics_1d(ps, ts)[0], ts.size

    base, nb = pf6_r2(0)
    print(f"drop  0: R²={base:.4f}  n={nb}  (baseline)")
    for k in (10, 30, 50, 100, 200, 300):
        r2, n = pf6_r2(k)
        print(f"drop {k:>3d}: R²={r2:.4f}  n={n}  (kept {931-k} shots)")

    # β_N of worst shots
    worst50 = set(shots_sorted[:50])
    try:
        bp = pd.read_parquet("meta/split_by_order_betan/betan_per_shot.parquet")
        print("\nbetan_per_shot cols:", list(bp.columns))
        col = "betan_max" if "betan_max" in bp.columns else ("beta_N" if "beta_N" in bp.columns else bp.columns[1])
        # normalize shot col
        sc = "shot" if "shot" in bp.columns else bp.columns[0]
        bp = bp.rename(columns={sc: "shot", col: "betan"})
        allb = bp[bp.shot.isin(set(shots_sorted))]["betan"]
        wst = bp[bp.shot.isin(worst50)]["betan"]
        print(f"\nβ_N (per-shot max) — 全 test 炮: median={allb.median():.3f} p10={allb.quantile(.1):.3f} p90={allb.quantile(.9):.3f}")
        print(f"β_N — worst-50 PF6 炮: median={wst.median():.3f} p10={wst.quantile(.1):.3f} p90={wst.quantile(.9):.3f}  (n={len(wst)})")
    except Exception as e:
        print(f"\nβ_N parquet 不可读: {e}")


if __name__ == "__main__":
    main()
