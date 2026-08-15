#!/usr/bin/env python3
"""对比 预测电流 vs PCS 实测电流 过 FreeGSNKE 的边界重建: 逐相位 median RMSE + median centered shape R²。

同口径: 两个 out-dir 用同一炮表(随机100)、同一 FPP/seed → 同一批 (shot,k) 切片, 只电流源不同。
  - pred-dir: 预测电流结果 (默认 pf6keep 全555跑目录, 自动筛这100炮; 或 freegsnke_testset_pf6keep_pred100)
  - pcs-dir : PCS 实测电流结果 (freegsnke_testset_pf6keep_pcs100)
RMSE = 8 控制点欧氏距离均方根(cm); shape R² = 去质心(去平移)后的形状吻合度。

用法: python -m paper_betan.compare_pred_vs_pcs
      python -m paper_betan.compare_pred_vs_pcs --pred-dir results/paper_betan/freegsnke_testset_pf6keep_pred100
"""
from __future__ import annotations
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import h5py

PH = ["ramp_up", "flat_top", "ramp_down"]
EFIT = Path("/data/EFIT")


def load_rows(d: str, shots: set[int]) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f"{d}/per_frame_shard*.csv")):
        try:
            for r in csv.DictReader(open(f)):
                if r["converged"] == "True" and r.get("rmse_8ctrl_m") and r.get("shot"):
                    if int(r["shot"]) in shots:
                        rows.append(r)
        except Exception:
            continue
    return rows


def load_r8z8(sz: int):
    with h5py.File(EFIT / f"{int(sz)}.h5", "r") as f:
        at = np.asarray(f["ATIME"][:], float).ravel()
        m = at >= 0.0
        T = int(m.sum())

        def tk(k):
            a = np.asarray(f[k][:], float)
            return a[m] if a.shape[0] == at.size else (a if a.shape[0] == T else None)
        R, Z = tk("R8"), tk("Z8")
        if R is not None and R.ndim == 2 and R.shape[1] != 8 and R.shape[0] == 8:
            R, Z = R.T, Z.T
    return R, Z


def stats(rows: list[dict], label: str) -> None:
    n_shots = len({int(r["shot"]) for r in rows})
    rmse = {ph: np.array([float(r["rmse_8ctrl_m"]) * 100 for r in rows if r["phase"] == ph]) for ph in PH}
    cache: dict[int, tuple] = {}
    r2 = {ph: [] for ph in PH}
    for r in rows:
        sz, k, ph = int(r["shot"]), int(r["k"]), r["phase"]
        if sz not in cache:
            try:
                cache[sz] = load_r8z8(sz)
            except Exception:
                cache[sz] = (None, None)
        R, Z = cache[sz]
        if R is None or not (0 <= k < R.shape[0]):
            continue
        dr = np.array([float(r[f"dr_{i}"]) for i in range(8)])
        dz = np.array([float(r[f"dz_{i}"]) for i in range(8)])
        num = np.sum((dr - dr.mean()) ** 2 + (dz - dz.mean()) ** 2)
        den = np.sum((R[k] - R[k].mean()) ** 2 + (Z[k] - Z[k].mean()) ** 2)
        v = 1.0 - num / den if den > 0 else np.nan
        if np.isfinite(v):
            r2[ph].append(v)
    print(f"\n=== {label}  ({n_shots} 炮, {len(rows)} 收敛帧) ===")
    print(f"{'phase':11s} {'n':>5s} {'RMSE中位(cm)':>13s} {'形状R²中位':>11s}")
    for ph in PH:
        v = rmse[ph]
        rv = np.array(r2[ph])
        print(f"{ph:11s} {len(v):5d} {np.median(v):13.2f} {np.median(rv):11.3f}")
    ar = np.concatenate([rmse[ph] for ph in PH if len(rmse[ph])]) if any(rmse[ph].size for ph in PH) else np.array([np.nan])
    avr = np.array([x for ph in PH for x in r2[ph]])
    print(f"{'All':11s} {len(ar):5d} {np.median(ar):13.2f} {np.median(avr):11.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default="results/paper_betan/freegsnke_testset_pf6keep",
                    help="预测电流结果目录(默认555跑目录,自动筛100炮; 或 freegsnke_testset_pf6keep_pred100)")
    ap.add_argument("--pcs-dir", default="results/paper_betan/freegsnke_testset_pf6keep_pcs100")
    ap.add_argument("--shots", default="meta/split_by_order_betan/freegsnke_pf6keep_random100_s20260718.txt")
    args = ap.parse_args()

    shots = set(int(x) for x in Path(args.shots).read_text().split() if x.strip())
    print(f"炮表 {args.shots}: {len(shots)} 炮")
    stats(load_rows(args.pred_dir, shots), f"预测电流 PRED  [{args.pred_dir}]")
    stats(load_rows(args.pcs_dir, shots), f"PCS实测电流 PCS  [{args.pcs_dir}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
