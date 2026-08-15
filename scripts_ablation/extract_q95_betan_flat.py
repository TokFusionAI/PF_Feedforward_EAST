#!/usr/bin/env python
"""提取 igbt 池每炮平顶段(Ip>=0.95*Ip_max)的 q95 与 β_N, 多进程并行.

平顶定义: PF h5 的 PCRL01(Ip) |Ip|>=0.95*max 的窗口 [t1,t2]; EFIT Q95/BETAN 取窗口内中位.
多进程: I/O bound, 登录节点多核, Pool(NWORK) 并行读 h5.
产物: meta/q95_betan_flat_pool.parquet (shot, q95_ft, betan_ft).
"""
from __future__ import annotations
from pathlib import Path
import time as _t
import multiprocessing as mp
import h5py
import numpy as np
import pandas as pd

ROOT = Path("/home/user/PF_current_EAST")
IGBT = ROOT / "meta/split_by_order_igbt"
EFIT_DIR = "/data/EFIT"
PF_DIR = "/data/PF_ATIME_dataset"
OUT = ROOT / "meta/q95_betan_flat_pool.parquet"
NWORK = 32


def load(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().split() if x.strip().isdigit()]


def process_shot(s: int):
    ef_p = f"{EFIT_DIR}/{s}.h5"
    pf_p = f"{PF_DIR}/{s}.h5"
    try:
        with h5py.File(pf_p, "r") as f:
            pf_t = f["time"][:].astype(float)
            ip = f["PCRL01"][:].astype(float)
        m = np.isfinite(pf_t) & np.isfinite(ip)
        pf_t, ip = pf_t[m], ip[m]
        if ip.size == 0:
            return None
        thr = 0.95 * np.nanmax(np.abs(ip))
        ft = np.abs(ip) >= thr
        if ft.sum() < 2:
            return None
        t1, t2 = float(pf_t[ft].min()), float(pf_t[ft].max())
        with h5py.File(ef_p, "r") as f:
            ef_t = f["ATIME"][:].astype(float)
            q = f["Q95"][:].astype(float)
            bn = f["BETAN"][:].astype(float)
        mm = np.isfinite(ef_t) & np.isfinite(q) & np.isfinite(bn) & (ef_t >= t1) & (ef_t <= t2)
        if mm.sum() < 2:
            return None
        return (int(s), float(np.median(q[mm])), float(np.median(bn[mm])))
    except Exception:
        return None


def main() -> int:
    pool = sorted(set(load(IGBT / "train_shots.txt")) | set(load(IGBT / "val_shots.txt"))
                  | set(load(IGBT / "test_shots.txt")))
    t0 = _t.monotonic()
    with mp.Pool(NWORK) as p:
        results = p.map(process_shot, pool, chunksize=64)
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows, columns=["shot", "q95_ft", "betan_ft"]).sort_values("shot")
    df.to_parquet(OUT, index=False)
    q = df["q95_ft"].to_numpy(); b = df["betan_ft"].to_numpy()
    print(f"Done: n={len(df)} err={len(pool)-len(rows)}  workers={NWORK}  elapsed={_t.monotonic()-t0:.0f}s")
    print(f"q95_ft:  p10={np.percentile(q,10):.2f} med={np.median(q):.2f} p90={np.percentile(q,90):.2f}")
    print(f"betan_ft: p10={np.percentile(b,10):.3f} med={np.median(b):.3f} p90={np.percentile(b,90):.3f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
