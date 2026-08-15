"""Harvest per-shot discharge metadata from EAST MDSplus (efit_east tree).

Pulls DATE_RUN (discharge date), COMMENTS (shot comment / scenario hint),
RUN_TYPE, EFIT_RUN for every shot and writes:
  * meta/shot_dates.csv   (incremental, resume-safe checkpoint)
  * meta/shot_dates.parquet (final, with parsed discharge_date / year)

Connection: this cluster has no DNS for mds.ipp.ac.cn, so run on a node that
can reach EAST MDS (e.g. compute-node) and pass the server IP via --mds-server
or the MDS_SERVER env var:

  # from a shell that can reach EAST MDS (node 383, torch env):
  MDS_SERVER=<MDS_IP> python -m bc.data.build_shot_dates --workers 24

  # or via SLURM on node 383:
  srun -p shenmadcu -w compute-node --gres=dcu:1 --cpus-per-task=32 -t 04:00:00 \
       python3 -m bc.data.build_shot_dates \
       --mds-server "$MDS_SERVER" --workers 28

Resume: re-running skips shots already present in the CSV checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scan_data.mds_bootstrap import _DEFAULT_MDS_SERVER  # EAST MDS IP（集群 DNS 坏，零配置直连，见模块头注释）

NODES_DEFAULT = "DATE_RUN,COMMENTS,RUN_TYPE,EFIT_RUN"
TREE = "efit_east"


def _dec(d) -> str:
    """Decode an MDS node value to a clean string."""
    if d is None:
        return ""
    if hasattr(d, "tobytes"):
        d = d.tobytes()
    if isinstance(d, (bytes, bytearray)):
        s = d.decode("utf-8", "replace")
    else:
        s = str(d)
    return s.strip().strip("\x00").strip()


# ---- per-worker state (created fresh in each forked worker) ----
_WORKER = {}


def _worker_init(srv: str, nodes: list[str]) -> None:
    import scan_data.mds_bootstrap as mb  # noqa: F401
    from scan_data.compat import _set_tree_env, bootstrap_mdsplus  # noqa: F401
    from scan_data.mds_bootstrap import bootstrap_mdsplus as _bs

    _bs()  # registers mdsthin.MDSplus as sys.modules['MDSplus']
    _set_tree_env(TREE, srv)  # sets efit_east_path / EFIT_EAST_path
    import MDSplus  # now the mdsthin-backed shim

    # fresh, per-worker connection (avoid sharing a forked socket FD)
    MDSplus.setDefaultConnection(MDSplus.Connection(srv))
    _WORKER["srv"] = srv
    _WORKER["nodes"] = nodes


def fetch_one(args) -> dict:
    shot, nodes = args
    from scan_data.compat import _fetch_node_data

    srv = _WORKER["srv"]
    out: dict = {"shot": int(shot)}
    any_ok = False
    for node in nodes:
        try:
            d, _ = _fetch_node_data(shot, TREE, node, mds_server=srv)
            out[node] = _dec(d)
            out[f"{node}_ok"] = True
            any_ok = True
        except Exception as e:  # node missing / shot not in tree / network
            out[node] = ""
            out[f"{node}_ok"] = False
            out[f"{node}_err"] = f"{type(e).__name__}: {str(e)[:100]}"
    out["ok"] = any_ok
    return out


def parse_date(s: str):
    """EAST EFIT DATE_RUN often like '13-APR-2019 09:42:31' or '2019-04-13 ...'."""
    if not s:
        return pd.NaT
    s = s.strip()
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mds-server", default=os.environ.get("MDS_SERVER", _DEFAULT_MDS_SERVER))
    ap.add_argument("--shots-file", type=Path, default=ROOT / "meta" / "valid_shots_input.txt")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "meta" / "shot_dates.csv")
    ap.add_argument("--out-parquet", type=Path, default=ROOT / "meta" / "shot_dates.parquet")
    ap.add_argument("--nodes", default=NODES_DEFAULT, help="comma-separated efit_east nodes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="process only first N todo shots (test)")
    args = ap.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    shots = [int(x) for x in args.shots_file.read_text().split() if x.strip()]
    print(f"shots in file: {len(shots)}; server={args.mds_server}; nodes={nodes}; workers={args.workers}")

    # resume: drop shots already in checkpoint
    done: set[int] = set()
    if args.out_csv.exists():
        try:
            done = set(pd.read_csv(args.out_csv, usecols=["shot"])["shot"].astype(int))
            print(f"resume: {len(done)} shots already in {args.out_csv}")
        except Exception as e:
            print(f"resume: could not read checkpoint ({e}); starting fresh")
    todo = [s for s in shots if int(s) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"todo: {len(todo)} shots")

    if not todo:
        print("nothing to do")
    else:
        cols = (["shot"] + [n for nd in nodes for n in (nd, f"{nd}_ok", f"{nd}_err")] + ["ok"])
        write_header = not args.out_csv.exists() or args.out_csv.stat().st_size == 0
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_done = 0
        with open(args.out_csv, "a", newline="", encoding="utf-8") as f, \
             ProcessPoolExecutor(max_workers=args.workers,
                                  initializer=_worker_init,
                                  initargs=(args.mds_server, nodes)) as ex:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if write_header:
                w.writeheader()
            batch: list[dict] = []
            for rec in ex.map(fetch_one, [(s, nodes) for s in todo]):
                batch.append(rec)
                if len(batch) >= args.flush_every:
                    for r in batch:
                        w.writerow(r)
                    f.flush()
                    n_done += len(batch)
                    batch.clear()
                    rate = n_done / max(time.time() - t0, 1e-6)
                    print(f"  {n_done}/{len(todo)}  ({rate:.1f} shot/s)", flush=True)
            for r in batch:
                w.writerow(r)
            f.flush()
        print(f"checkpoint written: {args.out_csv} ({time.time()-t0:.0f}s)")

    # final parquet with parsed date
    df = pd.read_csv(args.out_csv)
    df["discharge_date"] = df["DATE_RUN"].map(parse_date)
    df["year"] = pd.to_datetime(df["discharge_date"], errors="coerce").dt.year
    df = df.sort_values("shot").reset_index(drop=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    ok = int(df["ok"].sum()) if "ok" in df else 0
    has_date = int(df["discharge_date"].notna().sum())
    print(f"wrote {args.out_parquet}: rows={len(df)}, ok={ok}, with_date={has_date}")
    if has_date:
        print(df.dropna(subset=["discharge_date"]).assign(
            d=lambda x: x["discharge_date"].dt.strftime("%Y-%m"))["d"].value_counts()
            .sort_index().head(5))
        print("  ...")
        print(df.dropna(subset=["discharge_date"]).assign(
            d=lambda x: x["discharge_date"].dt.strftime("%Y-%m"))["d"].value_counts()
            .sort_index().tail(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
