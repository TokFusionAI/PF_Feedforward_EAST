"""按炮号顺序（时序）划分 train/val/test, 写到 meta/split_by_order_igbt/.

与 v1 (bc.data.split_shots_by_order) 的差异（用户 2026-07-09 指定）:
    1. 时间起点从 100000 提高到 117203 ——
       117203 对应 2022-11 EAST PF PS11/12 更新为 IGBT-PWM 之后的放电,
       故数据集只覆盖 IGBT-PWM 时代, 避免 PS 升级前后的硬件不一致。
    2. 新增 "有等离子体" 判据 IP_max_kA >= 200（来自 logbook, 合并进 shot_index）。
    3. 去掉 v1 的 dt_median < 0.3 这一条（用户明确不要采样率限制）。
    4. 不按 groups 排除 theory 等, 也不按 comments 排除破裂 ——
       判据严格限定为下面 5 条 AND, 不额外增减。

筛选标准（5 条 AND, 阈值由命令行参数控制, 结构固定）:
    shot >= --min-shot        (默认 117203, IGBT-PWM 起点)
    all_ok                    (关键信号覆盖率 >= 95%, 缺失按 False 处理)
    flat_top_dur_s > --min-flat-top-s  (默认 2.0, Wan et al. 正常炮判据)
    src_efit 非空             (有 EFIT 重建)
    IP_max_kA >= --min-ip-ka  (默认 200, 有等离子体)

数据源:
    meta/shot_index.parquet          -> shot, all_ok, src_efit
    meta/flat_top_dur.parquet        -> shot, flat_top_dur_s
    meta/split_by_order/logbook_ge100000_shot_summary_unflagged.csv
                                      -> _shot_number -> shot, IP_max_kA
    以 shot_index 为左基准做 left merge, 三者按 shot 关联。

切分: 合格炮按炮号升序 -> train(最旧, round(n*0.8)) / val / test(最新),
三者炮号范围不交叠。EAST 炮号随放电时间单调, 等价于按实验时间前向划分,
用于检验模型对更新 campaign 的泛化能力。

Usage:
    python -m bc.data.split_shots_by_order_v2 \
        --out-dir meta/split_by_order_igbt \
        --ratios 0.80 0.10 0.10 \
        --min-shot 117203 --min-flat-top-s 2.0 --min-ip-ka 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bc.data.split_shots import write_shots

ROOT = Path(__file__).resolve().parents[2]


def _ranges(shots: np.ndarray) -> dict:
    if len(shots) == 0:
        return {"n": 0, "min": None, "max": None}
    return {"n": int(len(shots)), "min": int(shots.min()), "max": int(shots.max())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shot-index", type=Path, default=ROOT / "meta" / "shot_index.parquet",
    )
    ap.add_argument(
        "--flat-top-table", type=Path, default=ROOT / "meta" / "flat_top_dur.parquet",
        help="bc.data.compute_flat_top_dur 的产物, 提供 flat_top_dur_s 列",
    )
    ap.add_argument(
        "--logbook", type=Path,
        default=ROOT / "meta" / "split_by_order" / "logbook_ge100000_shot_summary_unflagged.csv",
        help="提供 IP_max_kA 列的 logbook CSV",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "meta" / "split_by_order_igbt",
    )
    ap.add_argument(
        "--ratios", type=float, nargs=3, default=(0.80, 0.10, 0.10),
        metavar=("TR", "VA", "TE"),
    )
    ap.add_argument("--min-shot", type=int, default=117203, help="只保留 shot >= 该值 (默认 117203, IGBT-PWM 起点)")
    ap.add_argument("--min-flat-top-s", type=float, default=2.0, help="平顶时长下限 (秒), 默认 2.0")
    ap.add_argument("--min-ip-ka", type=float, default=200, help="IP_max_kA 下限 (kA), 默认 200 (有等离子体)")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {args.ratios}")
    if not args.flat_top_table.exists():
        raise SystemExit(
            f"missing {args.flat_top_table}; 先跑 `python -m bc.data.compute_flat_top_dur`"
        )
    if not args.logbook.exists():
        raise SystemExit(f"missing logbook {args.logbook}")
    if not args.shot_index.exists():
        raise SystemExit(f"missing shot_index {args.shot_index}")

    # ---- 1) 读三数据源, 以 shot_index 为左基准合并 ----
    si = pd.read_parquet(args.shot_index)[["shot", "all_ok", "src_efit"]]
    ft = pd.read_parquet(args.flat_top_table)[["shot", "flat_top_dur_s"]]
    lb = pd.read_csv(args.logbook)
    lb = lb.rename(columns={"_shot_number": "shot"})
    lb["IP_max_kA"] = pd.to_numeric(lb["IP_max_kA"], errors="coerce")
    lb = lb[["shot", "IP_max_kA"]].drop_duplicates("shot")

    df = si.merge(ft, on="shot", how="left").merge(lb, on="shot", how="left")
    print(f"merged: shot_index={len(si)} flat_top={len(ft)} logbook={len(lb)} -> {len(df)} rows (left base = shot_index)")
    n_missing_ft = int(df["flat_top_dur_s"].isna().sum())
    n_missing_ip = int(df["IP_max_kA"].isna().sum())
    print(f"  flat_top_dur_s 缺失 {n_missing_ft} 炮, IP_max_kA 缺失 {n_missing_ip} 炮 (将按判据排除)")

    # ---- 2) 漏斗: 从 shot>=min_shot 起, 逐条累加 ----
    filt_str = (
        f"shot >= {args.min_shot} and all_ok and flat_top_dur_s > {args.min_flat_top_s} "
        f"and src_efit notna and IP_max_kA >= {args.min_ip_ka}"
    )

    funnel = []
    base = df[df["shot"] >= args.min_shot]
    funnel.append({"step": f"shot >= {args.min_shot}", "n": int(len(base))})
    f1 = base[base["all_ok"].fillna(False)]
    funnel.append({"step": "+all_ok", "n": int(len(f1))})
    f2 = f1[f1["flat_top_dur_s"] > args.min_flat_top_s]
    funnel.append({"step": f"+flat_top > {args.min_flat_top_s}s", "n": int(len(f2))})
    f3 = f2[f2["src_efit"].notna()]
    funnel.append({"step": "+EFIT (src_efit notna)", "n": int(len(f3))})
    f4 = f3[f3["IP_max_kA"] >= args.min_ip_ka]
    funnel.append({"step": f"+IP >= {int(args.min_ip_ka)}kA", "n": int(len(f4))})

    print("\n漏斗 (从 shot >= %d 起, 5 条 AND 累加):" % args.min_shot)
    for f in funnel:
        print(f"  {f['step']:<35s} -> {f['n']:>6d}")

    # ---- 3) mask: 精确 5 条 AND ----
    mask = (
        (df["shot"] >= args.min_shot)
        & (df["all_ok"].fillna(False))
        & (df["flat_top_dur_s"] > args.min_flat_top_s)
        & (df["src_efit"].notna())
        & (df["IP_max_kA"] >= args.min_ip_ka)
    )
    assert len(f4) == int(mask.sum()), f"funnel final {len(f4)} != mask sum {int(mask.sum())}"

    shots = np.sort(df[mask]["shot"].astype(int).to_numpy())  # 升序 = 时序
    n = len(shots)
    print(f"\n合格池 pool_n = {n} (range {int(shots.min())}..{int(shots.max()) if n else '?'})")
    print(f"filter = {filt_str}")
    if n == 0:
        raise SystemExit("empty pool after filter")

    # ---- 4) 切分 ----
    n_tr = int(round(n * args.ratios[0]))
    n_va = int(round(n * args.ratios[1]))
    splits = {
        "train": shots[:n_tr],
        "val": shots[n_tr: n_tr + n_va],
        "test": shots[n_tr + n_va:],
    }

    # ---- 5) 断言 ----
    tr_max = int(splits["train"].max())
    va_min = int(splits["val"].min())
    te_min = int(splits["test"].min())
    overlap_ok = tr_max < va_min < te_min
    if not overlap_ok:
        raise SystemExit(f"范围交叠/非递增: train.max={tr_max} val.min={va_min} test.min={te_min}")
    total = sum(len(v) for v in splits.values())
    count_ok = total == n
    if not count_ok:
        raise SystemExit(f"split count mismatch: {total} vs {n}")
    print(f"\nOK 时序不交叠: train.max={tr_max} < val.min={va_min} < test.min={te_min}")
    print(f"OK 总数一致: {total} == {n}")

    # ---- 6) 输出 ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ordering": "chronological_by_shot",
        "filter": filt_str,
        "data_sources": {
            "shot_index": str(args.shot_index),
            "flat_top": str(args.flat_top_table),
            "logbook": str(args.logbook),
        },
        "criteria": {
            "min_shot": args.min_shot,
            "min_flat_top_s": args.min_flat_top_s,
            "min_ip_kA": int(args.min_ip_ka),
            "require_all_ok": True,
            "require_efit": True,
            "note": "no dt_median filter, no group/comment exclusion per user spec",
        },
        "ratios": list(args.ratios),
        "pool_n": n,
        "funnel": funnel,
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

    print(f"\ntotal {total} shots  ->  {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
