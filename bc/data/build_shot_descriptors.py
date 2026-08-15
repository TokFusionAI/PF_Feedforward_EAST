"""Build a fixed-length, state-only "shot descriptor" for every shot.

This is the shared similarity key for the leakage diagnostic, the nearest-shot
retrieval baseline, and the data-driven scenario clustering. It uses ONLY the
19-D input state (R8, Z8, lmsr, lmsz, PCRL01) — never the target currents — so
the retrieval baseline cannot be circular.

Descriptor (118-D, raw / un-normalized; downstream z-scores with its own train
pool so one file serves every split):
    per phase (ramp_up / flat_top / ramp_down) x 19 channels x {mean, std} = 114
      + 4 context scalars: log10(max|Ip|), T_eff, dt_median, flat_top_duration = 118

Empty flat-top (low-Ip / no-plateau shots) is filled with that shot's GLOBAL
mean/std so the descriptor stays smooth; phase_present flags which phases exist.

Phase detection uses RAW PCRL01 in amperes (the 1e5 / 0.9 thresholds are physical),
exactly like ablation/data/dataset.py. Descriptors are read straight from the
raw ATIME h5 (not PFDataset, which normalizes + pads).

Output: meta/shot_descriptors.npz  {shots, desc(N,118) f4, phase_present(N,3), ok(N,)}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bc.data.phases import detect_phase_slices  # noqa: E402

STATE_COLS = 19  # R8(8)+Z8(8)+lmsr+lmsz+PCRL01


def shot_descriptor(shot: int, dataset_root: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    p = Path(dataset_root) / f"{shot}.h5"
    if not p.exists():
        return np.full(118, np.nan, np.float32), np.zeros(3, bool), False
    try:
        with h5py.File(p, "r") as f:
            time_s = f["time"][:].astype(np.float64)
            R8 = f["R8"][:].astype(np.float64)
            Z8 = f["Z8"][:].astype(np.float64)
            lmsr = f["lmsr"][:].astype(np.float64).reshape(-1, 1)
            lmsz = f["lmsz"][:].astype(np.float64).reshape(-1, 1)
            pcrl = f["PCRL01"][:].astype(np.float64).reshape(-1, 1)
    except Exception:
        return np.full(118, np.nan, np.float32), np.zeros(3, bool), False

    T = int(time_s.shape[0])
    if T < 2:
        return np.full(118, np.nan, np.float32), np.zeros(3, bool), False

    state = np.concatenate([R8, Z8, lmsr, lmsz, pcrl], axis=1)  # (T,19)
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    pcrl1d = state[:, 18]

    slices = detect_phase_slices(time_s, pcrl1d)
    gmean = state.mean(axis=0)
    gstd = state.std(axis=0) + 1e-8

    feats = []
    present = []
    for name in ("ramp_up", "flat_top", "ramp_down"):
        s = slices[name]
        if s.stop > s.start and s.stop <= T:
            seg = state[s.start:s.stop]
            feats.append(seg.mean(axis=0))
            feats.append(seg.std(axis=0) + 1e-8)
            present.append(True)
        else:
            feats.append(gmean.copy())
            feats.append(gstd.copy())
            present.append(False)
    feat = np.concatenate(feats)  # 114

    ft = slices["flat_top"]
    if ft.stop > ft.start and ft.stop <= T and ft.start < T:
        ft_dur = float(time_s[min(ft.stop - 1, T - 1)] - time_s[ft.start])
    else:
        ft_dur = 0.0
    dt_med = float(np.median(np.diff(time_s))) if T > 1 else 0.0
    ctx = np.array([
        np.log10(max(float(np.abs(pcrl1d).max()), 1.0)),
        float(T),
        dt_med,
        ft_dur,
    ])
    vec = np.concatenate([feat, ctx]).astype(np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, np.array(present, dtype=bool), True


def _worker(args):
    shot, dataset_root = args
    vec, present, ok = shot_descriptor(shot, dataset_root)
    return shot, vec, present, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots-file", type=Path, default=ROOT / "meta" / "valid_shots_input.txt")
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/data/PF_ATIME_dataset"))
    ap.add_argument("--out", type=Path, default=ROOT / "meta" / "shot_descriptors.npz")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    shots = [int(x) for x in args.shots_file.read_text().split() if x.strip()]
    if args.limit:
        shots = shots[: args.limit]
    print(f"building descriptors for {len(shots)} shots with {args.workers} workers")

    from concurrent.futures import ProcessPoolExecutor
    N = len(shots)
    desc = np.zeros((N, 118), np.float32)
    present = np.zeros((N, 3), bool)
    ok = np.zeros((N,), bool)
    shots_out = np.zeros((N,), np.int64)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for shot, vec, pres, o in ex.map(_worker, [(s, str(args.dataset_root)) for s in shots]):
            i = done
            shots_out[i] = shot
            desc[i] = vec
            present[i] = pres
            ok[i] = o
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{N}  ({done/max(time.time()-t0,1e-6):.0f} shot/s)", flush=True)
    print(f"done {done}/{N} in {time.time()-t0:.0f}s; ok={int(ok.sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, shots=shots_out, desc=desc, phase_present=present, ok=ok)
    print(f"wrote {args.out}  desc{desc.shape}  nan={int(np.isnan(desc).sum())}  "
          f"no_flat_top={int((~present[:,1]).sum())}")
    # quick z-score sanity (global)
    mu = desc[ok].mean(axis=0); sd = desc[ok].std(axis=0) + 1e-8
    z = (desc - mu) / sd
    print(f"global z: mean={z[ok].mean():.3f} std={z[ok].std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
