"""按炮号划分 train/val/test, 写到 meta/{train,val,test}_shots.txt.

Usage:
    python -m bc.split_shots \
        --shot-index meta/shot_index.parquet \
        --out-dir meta \
        --filter "all_ok and dt_median < 0.3 and T >= 20" \
        --ratios 0.80 0.10 0.10 \
        --seed 20260424
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def write_shots(path: Path, shots: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(s)) for s in shots) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot-index", type=Path, default=Path("meta/shot_index.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("meta"))
    ap.add_argument(
        "--filter",
        type=str,
        default="all_ok and dt_median < 0.3 and T >= 20",
        help="pandas DataFrame.query 表达式; 用于筛掉 dt 离群 / T 过短 / 失败炮",
    )
    ap.add_argument(
        "--ratios", type=float, nargs=3, default=(0.80, 0.10, 0.10), metavar=("TR", "VA", "TE")
    )
    ap.add_argument("--seed", type=int, default=20260424)
    args = ap.parse_args()

    df = pd.read_parquet(args.shot_index)
    print(f"shot_index loaded: rows={len(df)}  cols={list(df.columns)}")
    df_ok = df.query(args.filter).copy()
    print(f"after filter '{args.filter}': {len(df_ok)} shots")

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {args.ratios}")

    rng = np.random.default_rng(args.seed)
    shots = df_ok["shot"].astype(int).to_numpy()
    perm = rng.permutation(shots)
    n = len(perm)
    n_tr = int(round(n * args.ratios[0]))
    n_va = int(round(n * args.ratios[1]))
    splits = {
        "train": sorted(perm[:n_tr].tolist()),
        "val": sorted(perm[n_tr : n_tr + n_va].tolist()),
        "test": sorted(perm[n_tr + n_va :].tolist()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        out_path = args.out_dir / f"{name}_shots.txt"
        write_shots(out_path, s)
        print(f"  {name:5s} ({len(s):>5d} shots) -> {out_path}  e.g. {s[:5]}")

    total = sum(len(v) for v in splits.values())
    if total != n:
        raise SystemExit(f"split count mismatch: {total} vs {n}")
    print(f"total {total} shots, seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
