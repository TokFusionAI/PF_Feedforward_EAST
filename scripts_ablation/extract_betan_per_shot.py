#!/usr/bin/env python
"""批量提取 igbt 池每炮的 β_N (EFIT BETAN 放电均值), 存 parquet, 供 split_by_order_betan 用.

β_N = BETAN (normalized beta, EFIT). 取放电时间段均值 (finite mask) 作为单炮 β_N 摘要.
(放电均值含爬升/下降, 略低于平顶值, 但用于"按 β_N 分模式"的相对划分足够;
 与可行性调研时一致, 故沿用, 不改用 max/平顶均值以免 cut 失配.)
"""
from __future__ import annotations
from pathlib import Path
import time as _t
import h5py
import numpy as np
import pandas as pd

ROOT = Path("/home/user/PF_current_EAST")
IGBT = ROOT / "meta/split_by_order_igbt"
EFIT_DIR = Path("/data/EFIT")
OUT = ROOT / "meta/split_by_order_betan/betan_per_shot.parquet"


def load(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().split() if x.strip().isdigit()]


def main() -> int:
    pool = (set(load(IGBT / "train_shots.txt")) | set(load(IGBT / "val_shots.txt"))
            | set(load(IGBT / "test_shots.txt")))
    shots = sorted(pool)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n_err = 0
    t0 = _t.monotonic()
    for i, s in enumerate(shots, 1):
        p = EFIT_DIR / f"{s}.h5"
        if not p.exists():
            n_err += 1
            continue
        try:
            with h5py.File(p, "r") as f:
                bn = f["BETAN"][:].astype(float)
            bn = bn[np.isfinite(bn)]
            if bn.size == 0:
                n_err += 1
                continue
            rows.append((s, float(bn.mean())))
        except Exception:
            n_err += 1
        if i % 4000 == 0:
            print(f"  {i}/{len(shots)}  ok={len(rows)} err={n_err}  "
                  f"elapsed={_t.monotonic()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["shot", "betan_mean"]).sort_values("shot")
    df.to_parquet(OUT, index=False)
    b = df["betan_mean"].to_numpy()
    print(f"\nDone: n={len(df)} err={n_err}  elapsed={_t.monotonic()-t0:.0f}s")
    print(f"β_N(mean/shot): p10={np.percentile(b,10):.3f} p25={np.percentile(b,25):.3f} "
          f"median={np.median(b):.3f} p75={np.percentile(b,75):.3f} p90={np.percentile(b,90):.3f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
