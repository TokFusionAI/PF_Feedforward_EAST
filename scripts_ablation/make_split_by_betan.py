#!/usr/bin/env python
"""生成 split_by_order_betan: 在 igbt 干净时序上叠加 β_N 筛选,
做"低β_N训练 -> 高β_N测试"的跨β模式(∥时间, 防泄漏) leave-one-mode-out.

- train = igbt_train ∩ {β_N < CUT}   (低β_N, 早期)
- val   = igbt_val   ∩ {β_N < CUT}   (低β_N, 中期; 与train同模式用于选模型)
- test  = igbt_test  ∩ {β_N > CUT}   (高β_N, 晚期, held-out模式)

物理: 低比压(低压强)训 -> 高比压(高压强)测 (外推到更高压强区).
防泄漏: 继承 igbt 硬时间边界 (train max < val min < test min); β_N 区间不相交 (<CUT vs >CUT).
β_N 取自 extract_betan_per_shot.py (EFIT BETAN 放电均值).
用法: python scripts_ablation/make_split_by_betan.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IGBT = ROOT / "meta/split_by_order_igbt"
OUT = ROOT / "meta/split_by_order_betan"
BETAN_TBL = ROOT / "meta/split_by_order_betan/betan_per_shot.parquet"
CUT = 0.8   # β_N 阈值: train/val < CUT, test > CUT


def load_shots(p: Path) -> list[int]:
    return sorted(int(x) for x in p.read_text().split() if x.strip().isdigit())


def main() -> int:
    bn = pd.read_parquet(BETAN_TBL).set_index("shot")["betan_mean"]

    def lo(shots):   # 低β_N (β_N < CUT)
        return [s for s in shots if s in bn.index and float(bn.loc[s]) < CUT]

    def hi(shots):   # 高β_N (β_N > CUT)
        return [s for s in shots if s in bn.index and float(bn.loc[s]) > CUT]

    tr = lo(load_shots(IGBT / "train_shots.txt"))
    va = lo(load_shots(IGBT / "val_shots.txt"))
    te = hi(load_shots(IGBT / "test_shots.txt"))

    assert max(tr) < min(va), f"train/val 时间重叠: {max(tr)} >= {min(va)}"
    assert max(va) < min(te), f"val/test 时间重叠: {max(va)} >= {min(te)}"
    assert not (set(tr) & set(te)), "train/test 炮号相交!"
    assert not (set(tr) & set(va)) and not (set(va) & set(te)), "split 间炮号相交!"

    OUT.mkdir(parents=True, exist_ok=True)
    for name, arr in [("train", tr), ("val", va), ("test", te)]:
        (OUT / f"{name}_shots.txt").write_text("\n".join(map(str, arr)) + "\n")

    import numpy as np
    def stats(arr):
        v = np.array([float(bn.loc[s]) for s in arr])
        return float(np.median(v)), float(v.min()), float(v.max())

    trm, trlo, trhi = stats(tr); tem, telo, tehi = stats(te)
    manifest = {
        "name": "split_by_order_betan",
        "purpose": "cross-β_N-mode (∥time, 防泄漏) leave-one-mode-out: 低β_N训→高β_N测 (外推高压强)",
        "base_split": "meta/split_by_order_igbt (干净时序, 无泄漏)",
        "betan_source": "EFIT BETAN 放电均值 (extract_betan_per_shot.py)",
        "betan_cut": CUT,
        "design": {
            "train": f"igbt_train ∩ {{β_N < {CUT}}} 低β_N早期",
            "val": f"igbt_val ∩ {{β_N < {CUT}}} 低β_N中期(与train同模式)",
            "test": f"igbt_test ∩ {{β_N > {CUT}}} 高β_N晚期 held-out模式",
        },
        "no_leakage": {
            "train_max_shot": max(tr), "val_min_shot": min(va), "test_min_shot": min(te),
            "note": "train max < val min < test min, 硬时间边界; β_N 区间不相交 (train<CUT < test>CUT)",
        },
        "betan_separation": {"train_median": trm, "train_range": [trlo, trhi],
                             "test_median": tem, "test_range": [telo, tehi]},
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "ranges": {"train": [min(tr), max(tr)], "val": [min(va), max(va)], "test": [min(te), max(te)]},
        "norm_stats": "meta/split_by_order_betan/norm_stats_notime.npz (在train上重算, 19维notime)",
        "note": "β_N⊥Ip(Pearson≈0.07) → 与Ip实验是不同模式轴; β_N近⊥时间(弱漂移), ∥时间靠阈值筛选凑出β_N分离(防泄漏).",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"train: {len(tr)} [{min(tr)}-{max(tr)}] β_N中位{trm:.2f}({trlo:.2f}-{trhi:.2f})")
    print(f"val  : {len(va)} [{min(va)}-{max(va)}]")
    print(f"test : {len(te)} [{min(te)}-{max(te)}] β_N中位{tem:.2f}({telo:.2f}-{tehi:.2f})")
    print(f"no-leakage OK: train max {max(tr)} < val min {min(va)} < test min {min(te)}")
    print(f"β_N disjoint: train<{CUT} ({trhi:.2f}) | test>{CUT} ({telo:.2f})")
    print(f"wrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
