"""按炮号顺序（时序）划分 train/val/test, 写到 meta/split_by_order/.

只做一件事: 筛选合格炮 -> 按炮号升序 -> 按 ratios 切 train(最旧)/val/test(最新)。
三者炮号范围不交叠, "一炮所有时间点同属一个集合" 天然满足 (切分在炮号级,
先于任何窗口化)。

筛选标准 (对齐 Wan et al. 2020 的 EAST 正常炮判据 + 采样一致性):
    all_ok                 关键信号 (R8/Z8/lmsr/lmsz/PCRL01 + 12 PCPF) 覆盖率 >= 95%
    dt_median < 0.3        限制到标准采样率放电 (ATIME dt 分布 p95≈0.3s),
                           排除长脉冲粗重构炮 (不同运行模式)
    shot >= 100000         丢掉 97xxx 老 era
    flat_top_dur_s > 2     平顶 > 2 s (Wan et al. 正常炮判据; 由
                           bc.data.compute_flat_top_dur 预计算, 口径与
                           analyze_didt/eval 一致: Ip=|PCRL01|, 平顶=|Ip|>=0.9*Ip_max 段)

与 bc.data.split_shots 的区别: 那个随机打乱, 这个按炮号顺序——EAST 炮号随放电
时间单调, 故等价于"按实验时间前向划分", 用于检验模型对更新 campaign 的泛化能力,
而非历史分布内插值。统计量 (norm mean/std、dI/dt 阈值) 若用于该划分, 必须只用
train_shots 重算。

Usage:
    python -m bc.data.split_shots_by_order \
        --shot-index meta/shot_index.parquet \
        --flat-top-table meta/flat_top_dur.parquet \
        --out-dir meta/split_by_order \
        --min-shot 100000 --min-flat-top-s 2.0 \
        --ratios 0.80 0.10 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bc.data.split_shots import write_shots

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER = "all_ok and dt_median < 0.3"


def _ranges(shots: np.ndarray) -> dict:
    if len(shots) == 0:
        return {"n": 0, "min": None, "max": None}
    return {"n": int(len(shots)), "min": int(shots.min()), "max": int(shots.max())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shot-index", type=Path, default=ROOT / "meta" / "shot_index.parquet")
    ap.add_argument(
        "--flat-top-table", type=Path, default=ROOT / "meta" / "flat_top_dur.parquet",
        help="bc.data.compute_flat_top_dur 的产物, 提供 flat_top_dur_s 列",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "meta" / "split_by_order")
    ap.add_argument(
        "--filter", type=str, default=DEFAULT_FILTER,
        help="pandas query 质量筛选 (不含 shot 阈值与 flat-top 阈值, 二者由参数追加)",
    )
    ap.add_argument("--min-shot", type=int, default=100000, help="只保留 shot >= 该值 (丢 97xxx)")
    ap.add_argument("--min-flat-top-s", type=float, default=2.0, help="平顶时长下限 (秒), 默认 2.0")
    ap.add_argument(
        "--ratios", type=float, nargs=3, default=(0.80, 0.10, 0.10), metavar=("TR", "VA", "TE")
    )
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {args.ratios}")
    if not args.flat_top_table.exists():
        raise SystemExit(
            f"missing {args.flat_top_table}; 先跑 `python -m bc.data.compute_flat_top_dur`"
        )

    df = pd.read_parquet(args.shot_index)
    ft = pd.read_parquet(args.flat_top_table)[["shot", "flat_top_dur_s"]]
    df = df.merge(ft, on="shot", how="left")  # 缺失 -> NaN -> 被 >min 排除
    n_missing_ft = int(df["flat_top_dur_s"].isna().sum())

    q = f"{args.filter} and shot >= {args.min_shot} and flat_top_dur_s > {args.min_flat_top_s}"
    df_ok = df.query(q).copy()
    shots = np.sort(df_ok["shot"].astype(int).to_numpy())  # 升序 = 时序
    n = len(shots)
    print(f"shot_index loaded: rows={len(df)}  (flat_top 缺失 {n_missing_ft} 炮未参与)")
    print(f"pool: filter='{q}'  -> {n} shots (range {shots.min()}..{shots.max()})")
    if n == 0:
        raise SystemExit("empty pool after filter")

    n_tr = int(round(n * args.ratios[0]))
    n_va = int(round(n * args.ratios[1]))
    splits = {
        "train": shots[:n_tr],
        "val": shots[n_tr : n_tr + n_va],
        "test": shots[n_tr + n_va :],
    }

    tr_max, va_min, te_min = int(splits["train"].max()), int(splits["val"].min()), int(splits["test"].min())
    if not (tr_max < va_min < te_min):
        raise SystemExit(f"范围交叠/非递增: train.max={tr_max} val.min={va_min} test.min={te_min}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ordering": "chronological_by_shot",
        "filter": q,
        "min_shot": args.min_shot,
        "min_flat_top_s": args.min_flat_top_s,
        "flat_top_table": str(args.flat_top_table),
        "ratios": list(args.ratios),
        "pool_n": n,
    }
    for name, arr in splits.items():
        out_path = args.out_dir / f"{name}_shots.txt"
        write_shots(out_path, arr.tolist())
        manifest[name] = _ranges(arr)
        print(
            f"  {name:5s} ({len(arr):>5d}) -> {out_path}   "
            f"range {int(arr.min())}..{int(arr.max())}   e.g. {arr[:5].tolist()}"
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(len(v) for v in splits.values())
    if total != n:
        raise SystemExit(f"split count mismatch: {total} vs {n}")
    print(f"OK 时序不交叠: train.max={tr_max} < val.min={va_min} < test.min={te_min}")
    print(f"total {total} shots  ->  {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
