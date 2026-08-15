#!/usr/bin/env python3
"""Compare FreeGSNKE boundary reconstructions driven by predicted vs recorded PF currents.

Two analyses over the same shot list and sampled slices:

1. vs-EFIT (default): each reconstruction is compared with the EFIT target
   boundary (Table 5 of the paper). Boundary RMSE comes directly from the
   per-frame CSVs; the translation-invariant shape R^2 additionally reads the
   EFIT R8/Z8 control points from the local EFIT h5 database (--efit-dir).

2. --direct: the two recovered boundaries are compared with each other on the
   slices where both reconstructions converged (Table 2 of the paper). The
   direct control-point RMSE is computable from the CSVs alone; the shape R^2
   (after removing the per-slice centroid) needs the EFIT R8/Z8 points.

Usage:
  python -m paper_betan.compare_pred_vs_pcs --stat mean            # Table 5 style
  python -m paper_betan.compare_pred_vs_pcs --stat mean --direct   # Table 2 style
"""
from __future__ import annotations
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

PH = ["ramp_up", "flat_top", "ramp_down"]
PLABEL = {"ramp_up": "Ramp-up", "flat_top": "Flat-top", "ramp_down": "Ramp-down"}
THIS = Path(__file__).resolve().parent
REPO = THIS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_PRED = "results/paper_betan/freegsnke_testset_pf6keep_pred100"
DEFAULT_PCS = "results/paper_betan/freegsnke_testset_pf6keep_pcs100"
DEFAULT_LIST = "meta/split_by_order_betan/freegsnke_pf6keep_random100_s20260718.txt"


def load_frames(run_dir: str, shots: set[int]) -> dict:
    """(shot, k) -> row dict, converged frames of the listed shots."""
    out: dict = {}
    for f in sorted(glob.glob(f"{run_dir}/per_frame_shard*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("converged") == "True" and r.get("dr_0") and int(r["shot"]) in shots:
                out[(int(r["shot"]), int(r["k"]))] = r
    return out


def load_efit_r8z8(efit_dir: Path, shot: int):
    """Return (R8, Z8) arrays (T, 8) on valid ATIME slices, or None."""
    import h5py
    fp = efit_dir / f"{shot}.h5"
    if not fp.is_file():
        return None
    with h5py.File(fp, "r") as f:
        at = np.asarray(f["ATIME"][:], float).ravel()
        m = at >= 0
        T = int(m.sum())

        def tk(key):
            a = np.asarray(f[key][:], float)
            return a[m] if a.shape[0] == at.size else (a if a.shape[0] == T else None)

        R, Z = tk("R8"), tk("Z8")
        if R is None or Z is None:
            return None
        if R.ndim == 2 and R.shape[1] != 8 and R.shape[0] == 8:
            R, Z = R.T, Z.T
    return R, Z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default=DEFAULT_PRED)
    ap.add_argument("--pcs-dir", default=DEFAULT_PCS)
    ap.add_argument("--shots-list", default=DEFAULT_LIST)
    ap.add_argument("--efit-dir", default="/data/EFIT",
                    help="local EFIT h5 database (needed only for shape R^2)")
    ap.add_argument("--stat", choices=["median", "mean"], default="mean",
                    help="aggregation over frames (paper tables use mean)")
    ap.add_argument("--direct", action="store_true",
                    help="paired direct comparison of the two recovered boundaries (Table 2); "
                         "default compares each against EFIT (Table 5)")
    args = ap.parse_args()

    shots = set(int(x) for x in open(Path(args.shots_list)) if x.strip())
    print(f"shot list {args.shots_list}: {len(shots)} discharges")

    pred = load_frames(args.pred_dir, shots)
    pcs = load_frames(args.pcs_dir, shots)
    print(f"converged frames: predicted {len(pred)}, recorded {len(pcs)}")

    efit_dir = Path(args.efit_dir)
    ecache: dict = {}

    if args.direct:
        pairs = sorted(set(pred) & set(pcs))
        print(f"paired slices (both converged): {len(pairs)}\n")
        print(f"=== Direct comparison of the two recovered boundaries, {args.stat} ± std (cm) ===")
        rmse = {ph: [] for ph in PH}
        r2 = {ph: [] for ph in PH}
        for (sz, k) in pairs:
            rp, rc = pred[(sz, k)], pcs[(sz, k)]
            ph = rp["phase"]
            ddr = np.array([float(rp[f"dr_{i}"]) - float(rc[f"dr_{i}"]) for i in range(8)])
            ddz = np.array([float(rp[f"dz_{i}"]) - float(rc[f"dz_{i}"]) for i in range(8)])
            rmse[ph].append(np.sqrt(np.mean(ddr**2 + ddz**2)) * 100)
            if sz not in ecache:
                try:
                    ecache[sz] = load_efit_r8z8(efit_dir, sz)
                except Exception:
                    ecache[sz] = None
            RZ = ecache[sz]
            if RZ is not None and RZ[0] is not None and 0 <= k < RZ[0].shape[0]:
                R, Z = RZ
                dpp = np.array([float(rc[f"dr_{i}"]) for i in range(8)])
                dzc = np.array([float(rc[f"dz_{i}"]) for i in range(8)])
                Rpc, Zpc = R[k] + dpp, Z[k] + dzc
                num = np.sum((ddr - ddr.mean()) ** 2 + (ddz - ddz.mean()) ** 2)
                den = np.sum((Rpc - Rpc.mean()) ** 2 + (Zpc - Zpc.mean()) ** 2)
                if den > 0:
                    r2[ph].append(1 - num / den)
        for ph in PH:
            v = np.array(rmse[ph]); rv = np.array(r2[ph]) if len(r2[ph]) else None
            line = f"{PLABEL[ph]:10s} n={len(v):5d}   {v.mean():6.2f} ± {v.std():5.2f} cm"
            if rv is not None:
                line += f"   shape R2 {rv.mean():.3f} ± {rv.std():.3f}"
            print(line)
        allv = np.concatenate([np.array(rmse[ph]) for ph in PH])
        allr = (np.concatenate([np.array(r2[ph]) for ph in PH])
                if any(len(r2[ph]) for ph in PH) else None)
        line = f"{'All':10s} n={len(allv):5d}   {allv.mean():6.2f} ± {allv.std():5.2f} cm"
        if allr is not None:
            line += f"   shape R2 {allr.mean():.3f} ± {allr.std():.3f}"
        print(line)
        return 0

    # ---- vs-EFIT per source (Table 5) ----
    for name, frames in (("Predicted currents", pred), ("Recorded currents", pcs)):
        print(f"\n=== {name} vs EFIT, {args.stat} ± std ===")
        rmse = {ph: [] for ph in PH}
        r2 = {ph: [] for ph in PH}
        for (sz, k), r in frames.items():
            dr = np.array([float(r[f"dr_{i}"]) for i in range(8)])
            dz = np.array([float(r[f"dz_{i}"]) for i in range(8)])
            rmse[r["phase"]].append(np.sqrt(np.mean(dr**2 + dz**2)) * 100)
            if sz not in ecache:
                try:
                    ecache[sz] = load_efit_r8z8(efit_dir, sz)
                except Exception:
                    ecache[sz] = None
            RZ = ecache[sz]
            if RZ is not None and RZ[0] is not None and 0 <= k < RZ[0].shape[0]:
                R, Z = RZ
                num = np.sum((dr - dr.mean()) ** 2 + (dz - dz.mean()) ** 2)
                den = np.sum((R[k] - R[k].mean()) ** 2 + (Z[k] - Z[k].mean()) ** 2)
                if den > 0:
                    r2[r["phase"]].append(1 - num / den)
        for ph in PH:
            v = np.array(rmse[ph]); rv = np.array(r2[ph]) if len(r2[ph]) else None
            line = f"{PLABEL[ph]:10s} n={len(v):5d}   {v.mean():6.2f} ± {v.std():5.2f} cm"
            if rv is not None:
                line += f"   shape R2 {rv.mean():.3f} ± {rv.std():.3f}"
            print(line)
        allv = np.concatenate([np.array(rmse[ph]) for ph in PH])
        allr = (np.concatenate([np.array(r2[ph]) for ph in PH])
                if any(len(r2[ph]) for ph in PH) else None)
        line = f"{'All':10s} n={len(allv):5d}   {allv.mean():6.2f} ± {allv.std():5.2f} cm"
        if allr is not None:
            line += f"   shape R2 {allr.mean():.3f} ± {allr.std():.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
