"""逐炮计算 flat-top 时长 (秒), 写 meta/flat_top_dur.parquet.

为 train/val/test 划分的新筛选条件 'flat_top_dur_s > 2' 服务 —— 对齐 Wan et al.
2020 (EAST 正常炮判据之一: flat-top > 2 s)。详见 plans 与 bc.data.split_shots_by_order。

相位检测复用 bc.data.phases.detect_phase_slices (与 analyze_didt / eval /
build_shot_descriptors 同一套规则): Ip = |PCRL01|, flat-top = |Ip| >= 0.9*Ip_max 的
首末点连成的连续段 (且该段 >= 0.3 s 才算平顶)。每炮只读 `time` 和 `PCRL01`
(PCRL01 先 nan_to_num), 直接读 ATIME h5。

输出列: shot, flat_top_dur_s, n_flat_top_steps, has_flat_top, ok。

Usage:
    python -m bc.data.compute_flat_top_dur \
        --dataset-root /data/PF_ATIME_dataset \
        --out meta/flat_top_dur.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bc.data.phases import detect_phase_slices  # noqa: E402


def shot_flat_top(shot: int, dataset_root: str) -> tuple[int, float, int, bool, bool]:
    """Return (shot, flat_top_dur_s, n_flat_top_steps, has_flat_top, ok)."""
    p = Path(dataset_root) / f"{shot}.h5"
    if not p.exists():
        return shot, 0.0, 0, False, False
    try:
        with h5py.File(p, "r") as f:
            t = f["time"][:].astype(np.float64)
            pcrl = f["PCRL01"][:].astype(np.float64)
    except Exception:
        return shot, 0.0, 0, False, False
    T = int(t.shape[0])
    if T < 2:
        return shot, 0.0, 0, False, True
    pcrl = np.nan_to_num(pcrl, nan=0.0, posinf=0.0, neginf=0.0)
    sl = detect_phase_slices(t, pcrl)
    ft = sl["flat_top"]
    if ft.stop > ft.start and ft.start < T:
        n = ft.stop - ft.start
        ft_dur = float(t[min(ft.stop - 1, T - 1)] - t[ft.start])
    else:
        n, ft_dur = 0, 0.0
    return shot, ft_dur, int(n), bool(n > 0), True


def _worker(args):
    return shot_flat_top(args[0], args[1])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shot-index", type=Path, default=ROOT / "meta" / "shot_index.parquet",
        help="炮号来源 (读 'shot' 列); 默认对 shot_index 里所有炮计算",
    )
    ap.add_argument(
        "--dataset-root", type=Path,
        default=Path("/data/PF_ATIME_dataset"),
    )
    ap.add_argument("--out", type=Path, default=ROOT / "meta" / "flat_top_dur.parquet")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_parquet(args.shot_index)
    shots = df["shot"].astype(int).to_numpy()
    N = len(shots)
    print(f"computing flat-top duration for {N} shots ({args.shot_index}) with {args.workers} workers")

    from concurrent.futures import ProcessPoolExecutor

    rows: list[tuple] = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(_worker, [(int(s), str(args.dataset_root)) for s in shots]):
            rows.append(rec)
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{N}  ({done / max(time.time() - t0, 1e-6):.0f} shot/s)", flush=True)

    out = pd.DataFrame(rows, columns=["shot", "flat_top_dur_s", "n_flat_top_steps", "has_flat_top", "ok"])
    out["shot"] = out["shot"].astype("int64")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    nok = int(out["ok"].sum())
    nft = int(out["has_flat_top"].sum())
    n2 = int((out["flat_top_dur_s"] > 2).sum())
    print(f"done {done}/{N} in {time.time() - t0:.0f}s  ok={nok}  has_flat_top={nft}  flat_top>2s={n2}")
    print(f"flat_top_dur_s 分布 (ok 炮): "
          f"min={out.loc[out.ok,'flat_top_dur_s'].min():.2f} "
          f"med={out.loc[out.ok,'flat_top_dur_s'].median():.2f} "
          f"p95={out.loc[out.ok,'flat_top_dur_s'].quantile(.95):.2f} "
          f"max={out.loc[out.ok,'flat_top_dur_s'].max():.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
