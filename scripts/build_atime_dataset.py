"""CLI entrypoint: build the ATIME-aligned per-shot h5 dataset in parallel.

Example:
    python scripts/build_atime_dataset.py --limit 20 --dry-run
    python scripts/build_atime_dataset.py --limit 200
    python scripts/build_atime_dataset.py --resume

See ``plans/atime_aligned_dataset_build.md`` for the full spec.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dataset_io import (  # noqa: E402
    BUILD_VERSION,
    COVERAGE_COLUMNS,
    PCPF_NAMES,
    STATE_SCALAR_NAMES,
    ShotSource,
    coverage_dict,
    gate_check,
    load_shot_sources,
    resample_shot,
    write_shot_h5,
)


REPO_ROOT = _THIS_DIR.parent
DEFAULT_EFIT_DIR = Path("/data/EFIT")
DEFAULT_PCS_DIR = Path("/data/DataBase/PCSEASTRaw")
DEFAULT_OVERLAP_FILE = REPO_ROOT / "meta" / "valid_shots_input.txt"
DEFAULT_OUT_DIR = Path("/data/PF_ATIME_dataset")
DEFAULT_META_DIR = REPO_ROOT / "meta"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"


def _configure_logger(log_dir: Path, run_tag: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"build_{run_tag}.log"
    logger = logging.getLogger("build_atime_dataset")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(sh)
    logger.info(f"log file: {log_path}")
    return logger


def _read_shots_txt(path: Path) -> list[int]:
    shots: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t.isdigit():
            shots.append(int(t))
    return sorted(set(shots))


def _existing_output_shots(out_dir: Path) -> set[int]:
    if not out_dir.exists():
        return set()
    out: set[int] = set()
    for fp in out_dir.glob("*.h5"):
        stem = fp.stem
        if stem.isdigit():
            out.add(int(stem))
    return out


_WORKER_CONFIG: dict[str, Any] = {}


def _worker_init(cfg: dict[str, Any]) -> None:
    _WORKER_CONFIG.clear()
    _WORKER_CONFIG.update(cfg)


def _process_one(shot: int) -> dict[str, Any]:
    cfg = _WORKER_CONFIG
    efit_dir: Path = cfg["efit_dir"]
    pcs_dir: Path = cfg["pcs_dir"]
    out_dir: Path = cfg["out_dir"]
    dry_run: bool = cfg["dry_run"]
    build_ts: str = cfg["build_ts"]
    compression: str | None = cfg["compression"]
    compression_opts: int | None = cfg["compression_opts"]

    result: dict[str, Any] = {"shot": int(shot), "status": "error", "reason": "", "written": False}
    t0 = time.monotonic()
    try:
        src: ShotSource = load_shot_sources(shot, efit_dir, pcs_dir)
    except Exception as e:
        result["reason"] = f"load_error:{type(e).__name__}:{e}"
        return result

    ok, reason = gate_check(src)
    if not ok:
        result["status"] = "rejected"
        result["reason"] = reason
        return result

    try:
        arr = resample_shot(src)
    except Exception as e:
        result["reason"] = f"resample_error:{type(e).__name__}:{e}"
        return result

    cov = coverage_dict(arr)
    result.update(
        status="ok",
        T=int(arr.T),
        t_start=float(arr.t_start),
        t_end=float(arr.t_end),
        dt_median=float(arr.dt_median),
        src_efit=str(arr.src_efit),
        src_pcs=str(arr.src_pcs),
        **cov,
    )
    result["all_ok"] = bool(
        all(v >= 0.95 for v in cov.values()) and not math.isnan(arr.dt_median)
    )

    if not dry_run:
        try:
            write_shot_h5(
                out_dir / f"{shot}.h5",
                arr,
                compression=compression,
                compression_opts=compression_opts,
                build_timestamp=build_ts,
            )
            result["written"] = True
        except Exception as e:
            result["status"] = "error"
            result["reason"] = f"write_error:{type(e).__name__}:{e}"
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            result["reason"] += f"|{tb}"
            return result

    result["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap-file", type=Path, default=DEFAULT_OVERLAP_FILE)
    ap.add_argument("--efit-dir", type=Path, default=DEFAULT_EFIT_DIR)
    ap.add_argument("--pcs-dir", type=Path, default=DEFAULT_PCS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N shots")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute stats but do not write per-shot h5 files",
    )
    ap.add_argument(
        "--resume", action="store_true", help="skip shots that already have an h5 in --out-dir"
    )
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    ap.add_argument("--compression-opts", type=int, default=4)
    ap.add_argument(
        "--progress-every", type=int, default=500, help="log a rolling stats line every N shots"
    )
    ap.add_argument("--run-tag", type=str, default=None)
    ap.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "also overwrite canonical meta files (shot_index.parquet, "
            "valid_shots_output.txt, build_errors.csv). Only pass this on full or "
            "resumed-full runs; small/demo runs should NOT pass it to avoid polluting "
            "canonical."
        ),
    )
    args = ap.parse_args()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _configure_logger(args.log_dir, run_tag)

    if not args.overlap_file.exists():
        logger.error(f"overlap file missing: {args.overlap_file}")
        return 2

    args.meta_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    shots_all = _read_shots_txt(args.overlap_file)
    logger.info(f"overlap shots: {len(shots_all)} from {args.overlap_file}")

    if args.resume and not args.dry_run:
        existing = _existing_output_shots(args.out_dir)
        shots_todo = [s for s in shots_all if s not in existing]
        logger.info(f"resume: {len(existing)} already present, {len(shots_todo)} to process")
    else:
        shots_todo = list(shots_all)

    if args.limit is not None:
        shots_todo = shots_todo[: args.limit]
        logger.info(f"limit applied: {len(shots_todo)} shots")

    if not shots_todo:
        logger.info("nothing to do")
        return 0

    build_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    compression: str | None = None if args.compression == "none" else args.compression
    cfg = dict(
        efit_dir=args.efit_dir,
        pcs_dir=args.pcs_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        build_ts=build_ts,
        compression=compression,
        compression_opts=args.compression_opts,
    )

    build_config = {
        "run_tag": run_tag,
        "build_version": BUILD_VERSION,
        "build_timestamp": build_ts,
        "overlap_file": str(args.overlap_file),
        "efit_dir": str(args.efit_dir),
        "pcs_dir": str(args.pcs_dir),
        "out_dir": str(args.out_dir),
        "meta_dir": str(args.meta_dir),
        "workers": int(args.workers),
        "limit": args.limit,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "compression": args.compression,
        "compression_opts": args.compression_opts,
        "PCPF_names": PCPF_NAMES,
        "state_scalar_names": STATE_SCALAR_NAMES,
    }
    (args.meta_dir / f"build_config_{run_tag}.json").write_text(
        json.dumps(build_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"build_config: {build_config}")

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        def tqdm(it, **kw):  # type: ignore
            return it

    workers = max(1, min(args.workers, len(shots_todo)))
    chunksize = max(1, math.ceil(len(shots_todo) / (workers * 16)))
    ctx = mp.get_context("fork")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    counts = {"ok": 0, "rejected": 0, "error": 0}
    t_wall = time.monotonic()

    with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        iterator = pool.imap_unordered(_process_one, shots_todo, chunksize=chunksize)
        for i, res in enumerate(tqdm(iterator, total=len(shots_todo), desc="build"), start=1):
            counts[res.get("status", "error")] = counts.get(res.get("status", "error"), 0) + 1
            if res["status"] == "ok":
                rows.append(res)
            else:
                errors.append(
                    {
                        "shot": res["shot"],
                        "status": res["status"],
                        "reason": res.get("reason", ""),
                    }
                )
            if i % args.progress_every == 0:
                pass_rate = counts["ok"] / i
                if rows:
                    t_mean = float(np.mean([r["T"] for r in rows]))
                    dt_mean = float(np.mean([r["dt_median"] for r in rows]))
                    cov_pcpf = float(
                        np.mean([r[f"coverage_{PCPF_NAMES[0]}"] for r in rows])
                    )
                else:
                    t_mean = float("nan")
                    dt_mean = float("nan")
                    cov_pcpf = float("nan")
                logger.info(
                    f"progress {i}/{len(shots_todo)}: "
                    f"pass={pass_rate:.3f} ok={counts['ok']} "
                    f"rej={counts['rejected']} err={counts['error']} "
                    f"T_mean={t_mean:.1f} dt_med={dt_mean:.3f} cov_pcpf1={cov_pcpf:.3f}"
                )

    wall_s = time.monotonic() - t_wall
    logger.info(
        f"done in {wall_s:.1f}s: ok={counts['ok']} "
        f"rej={counts['rejected']} err={counts['error']} total={len(shots_todo)}"
    )

    if rows:
        cols = [
            "shot",
            "T",
            "t_start",
            "t_end",
            "dt_median",
            "all_ok",
            *COVERAGE_COLUMNS,
            "src_efit",
            "src_pcs",
        ]
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols].sort_values("shot").reset_index(drop=True)

        idx_path = args.meta_dir / f"shot_index_{run_tag}.parquet"
        df.to_parquet(idx_path, index=False)
        logger.info(f"wrote {idx_path} rows={len(df)}")

        valid_out_path = args.meta_dir / f"valid_shots_output_{run_tag}.txt"
        valid_out_path.write_text(
            "\n".join(str(int(s)) for s in df["shot"].tolist()) + "\n", encoding="utf-8"
        )
        logger.info(f"wrote {valid_out_path}")

        if args.canonical:
            canonical_idx = args.meta_dir / "shot_index.parquet"
            df.to_parquet(canonical_idx, index=False)
            canonical_valid = args.meta_dir / "valid_shots_output.txt"
            canonical_valid.write_text(
                "\n".join(str(int(s)) for s in df["shot"].tolist()) + "\n",
                encoding="utf-8",
            )
            logger.info(f"updated canonical: {canonical_idx}, {canonical_valid}")
        else:
            logger.info(
                "canonical shot_index.parquet / valid_shots_output.txt NOT updated "
                "(pass --canonical on full runs to update them)"
            )

    if errors:
        err_df = pd.DataFrame(errors).sort_values("shot").reset_index(drop=True)
        err_path = args.meta_dir / f"build_errors_{run_tag}.csv"
        err_df.to_csv(err_path, index=False)
        logger.info(f"wrote {err_path} rows={len(err_df)}")
        if args.canonical:
            canonical_err = args.meta_dir / "build_errors.csv"
            err_df.to_csv(canonical_err, index=False)
            logger.info(f"updated canonical: {canonical_err}")

    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
