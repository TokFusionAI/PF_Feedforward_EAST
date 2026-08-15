"""Generalization-friendly train/val/test splits (namespaced, never clobbers the
canonical meta/{train,val,test}_shots.txt).

Why: the canonical split is a flat random per-shot permutation. EAST adjacent /
same-day / same-scenario shots share near-identical target waveforms, so random
splitting lets near-twin shots leak across train/test. This generator produces
splits that prevent that leakage:

  temporal   : chronological (sort by shot number, monotonic in time on EAST).
               train = oldest 80%, val = middle 10%, test = NEWEST 10%.
               Answers the reviewer's "split by experiment time".
  external   : hold out the NEWEST block (default 15%) as a fully-quarantined
               external test; train/val come from the older 85%. Answers
               "keep a newer complete campaign as an external test set".
  random     : fresh flat random 80/10/10 on the SAME (>= min_shot) pool, as a
               fair-pool baseline to compare against temporal/cluster.
  cluster    : group-split on data-driven scenario clusters (whole cluster in one
               split). Needs meta/shot_descriptors.npz (build_shot_descriptors.py).

All modes apply the same quality filter and the same --min-shot cutoff (default
100000, to drop the oldest 97xxx era). Outputs go to meta/splits/<name>/ with a
manifest JSON recording sizes + shot-number ranges + config (for year mapping).

Usage:
    python -m bc.data.split_shots_generalization --mode temporal  --min-shot 100000
    python -m bc.data.split_shots_generalization --mode external  --min-shot 100000 --external-frac 0.15
    python -m bc.data.split_shots_generalization --mode random    --min-shot 100000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bc.data.split_shots import write_shots

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER = "all_ok and dt_median < 0.3 and T >= 20"


def _ranges(shots: np.ndarray) -> dict:
    if len(shots) == 0:
        return {"n": 0, "min": None, "max": None}
    return {"n": int(len(shots)), "min": int(shots.min()), "max": int(shots.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot-index", type=Path, default=ROOT / "meta" / "shot_index.parquet")
    ap.add_argument("--mode", required=True, choices=["random", "temporal", "external", "cluster"])
    ap.add_argument("--min-shot", type=int, default=100000, help="drop shots below this number (default 100000)")
    ap.add_argument("--filter", type=str, default=DEFAULT_FILTER)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.80, 0.10, 0.10), metavar=("TR", "VA", "TE"))
    ap.add_argument("--external-frac", type=float, default=0.15, help="newest fraction held out as external test")
    ap.add_argument("--n-clusters", type=int, default=100)
    ap.add_argument("--descriptor", type=Path, default=ROOT / "meta" / "shot_descriptors.npz")
    ap.add_argument("--seed", type=int, default=20260424)
    ap.add_argument("--out-dir", type=Path, default=None, help="default meta/splits/<mode>_ge<min_shot>")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {args.ratios}")

    df = pd.read_parquet(args.shot_index)
    q = f"{args.filter} and shot >= {args.min_shot}"
    df_ok = df.query(q).copy()
    shots = np.sort(df_ok["shot"].astype(int).to_numpy())
    n = len(shots)
    print(f"pool: filter='{q}'  -> {n} shots (range {shots.min()}..{shots.max()})")
    if n == 0:
        raise SystemExit("empty pool after filter")

    name = args.out_dir.name if args.out_dir else f"{args.mode}_ge{args.min_shot}"
    out_dir = args.out_dir or (ROOT / "meta" / "splits" / name)
    rng = np.random.default_rng(args.seed)

    splits: dict[str, np.ndarray] = {}
    cfg: dict = {"mode": args.mode, "filter": q, "min_shot": args.min_shot,
                 "ratios": list(args.ratios), "seed": args.seed, "pool_n": n}

    if args.mode in ("random", "temporal"):
        n_tr = int(round(n * args.ratios[0]))
        n_va = int(round(n * args.ratios[1]))
        order = rng.permutation(shots) if args.mode == "random" else shots  # temporal: already sorted
        splits["train"] = np.sort(order[:n_tr])
        splits["val"] = np.sort(order[n_tr:n_tr + n_va])
        splits["test"] = np.sort(order[n_tr + n_va:])
        cfg["ordering"] = "random_permutation" if args.mode == "random" else "chronological_by_shot"

    elif args.mode == "external":
        # newest block -> external test; older part -> train/val (proportional to ratios)
        n_ext = int(round(n * args.external_frac))
        older = shots[:-n_ext] if n_ext > 0 else shots.copy()
        ext = shots[-n_ext:] if n_ext > 0 else np.array([], dtype=shots.dtype)
        rem = len(older)
        va_frac = args.ratios[1] / (args.ratios[0] + args.ratios[1])
        n_va = int(round(rem * va_frac))
        splits["train"] = np.sort(older[: rem - n_va])
        splits["val"] = np.sort(older[rem - n_va:])
        splits["external"] = np.sort(ext)
        cfg["external_frac"] = args.external_frac

    elif args.mode == "cluster":
        if not args.descriptor.exists():
            raise SystemExit(f"cluster mode needs {args.descriptor}; run bc.data.build_shot_descriptors first")
        d = np.load(args.descriptor, allow_pickle=False)
        desc_shots = d["shots"].astype(int)
        desc = d["desc"].astype(np.float32)
        # align descriptors to the pool order
        idx_map = {int(s): i for i, s in enumerate(desc_shots)}
        keep = [i for s in shots if (i := idx_map.get(int(s))) is not None]
        X = desc[keep]
        shots_c = shots[[idx_map[int(s)] for s in shots]]
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        labels = km.fit_predict(X)
        # group-aware shuffle: assign whole clusters to train/val/test by shuffled cluster order
        uniq = np.unique(labels)
        rng.shuffle(uniq)
        csize = {c: int((labels == c).sum()) for c in uniq}
        # target counts
        n_tr = int(round(n * args.ratios[0])); n_va = int(round(n * args.ratios[1]))
        order_clusters = sorted(uniq, key=lambda c: -csize[c])  # big first for stable fill
        assign = {}
        cur_tr = cur_va = 0
        for c in order_clusters:
            if cur_tr < n_tr:
                assign[c] = "train"; cur_tr += csize[c]
            elif cur_va < n_va:
                assign[c] = "val"; cur_va += csize[c]
            else:
                assign[c] = "test"
        splits["train"] = np.sort(shots_c[assign_arr(labels, assign, "train")])
        splits["val"] = np.sort(shots_c[assign_arr(labels, assign, "val")])
        splits["test"] = np.sort(shots_c[assign_arr(labels, assign, "test")])
        cfg["n_clusters"] = args.n_clusters

    # write
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, **cfg}
    for split_name, arr in splits.items():
        p = out_dir / f"{split_name}_shots.txt"
        write_shots(p, arr.tolist())
        manifest[split_name] = _ranges(arr)
        print(f"  {split_name:9s} {manifest[split_name]}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    total = sum(v.get("n", 0) for k, v in manifest.items() if isinstance(v, dict) and "min" in v)
    print(f"wrote {out_dir}/  (total {total} shots)")
    return 0


def assign_arr(labels, assign, key):
    return np.array([i for i, c in enumerate(labels) if assign[c] == key], dtype=int)


if __name__ == "__main__":
    raise SystemExit(main())
