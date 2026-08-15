#!/usr/bin/env python3
"""并行处理 ``results/freegsnke_precursors`` 下所有（或指定）炮号：整炮 PCS + bc_pred freegsnke，再拼三阶段最优图。

每个子进程负责 **一炮** 的完整流水，便于在多核 CPU 上并行；多进程同机 GPU 时易争显存，默认在
``--workers > 1`` 时 BC 推理改用 **CPU**（``--gpu-parallel`` 可改回 GPU，风险自负）。

对目录中每个含 ``precursor.npz`` 的子目录名（整数炮号）::

    1) ``python -m bc.batch_freegsnke.run_freegsnke_whole_shot --coil-source pcs …``
    2) ``python -m bc.batch_freegsnke.run_freegsnke_whole_shot --coil-source bc_pred …``
    3) ``python -m bc.batch_freegsnke.montage_best_per_phase …``

示例::

    # 扫描 precursors 下全部炮号，8 进程并行（BC 用 CPU）
    python -m bc.batch_freegsnke.batch_precursors_parallel --precursors-root results/freegsnke_precursors --workers 8

    # 只跑少量炮测试
    python -m bc.batch_freegsnke.batch_precursors_parallel --shots 105653,158413 --workers 2

    # 单 GPU： workers=1 ，BC 可走 CUDA
    python -m bc.batch_freegsnke.batch_precursors_parallel --workers 1 --gpu-parallel

环境须已安装 freegsnke；bc_pred 步需要 torch + ckpt。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent.parent


def _list_precursor_shots(precursors_root: Path) -> list[int]:
    out: list[int] = []
    if not precursors_root.is_dir():
        return out
    for d in sorted(precursors_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "precursor.npz").is_file():
            continue
        try:
            out.append(int(d.name))
        except ValueError:
            continue
    return sorted(out)


def _parse_shots_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]


def _worker_one_shot(payload: dict[str, Any]) -> dict[str, Any]:
    """子进程入口（须可 pickle）。"""
    shot = int(payload["shot"])
    repo = Path(payload["repo"])
    py = str(payload.get("python") or sys.executable)
    prec_root = Path(payload["precursor_root"])
    prec = prec_root / str(shot) / "precursor.npz"
    if not prec.is_file():
        return {"shot": shot, "rc": 2, "msg": f"missing {prec}"}

    if payload.get("dry_run"):
        print(
            f"[dry-run] shot={shot} would run: pcs whole_shot, bc_pred whole_shot, montage_best_per_phase "
            f"(out={payload['whole_shot_root']})",
            flush=True,
        )
        return {"shot": shot, "rc": 0, "msg": "dry-run"}

    def _run(args: list[str]) -> int:
        r = subprocess.run(
            [py, *args],
            cwd=str(repo),
            env={**os.environ, **(payload.get("env_extra") or {})},
        )
        return int(r.returncode)

    rc_max = 0
    common_tail: list[str] = [
        "--shot",
        str(shot),
        "--precursor-npz",
        str(prec),
        "--out-root",
        str(payload["whole_shot_root"]),
        "--rtol",
        str(payload["rtol"]),
        "--nx",
        str(payload["nx"]),
        "--ny",
        str(payload["ny"]),
        "--by-t-efit-subdir",
        str(payload["by_t_efit_subdir"]),
        "--slices-subdir",
        str(payload["slices_subdir"]),
    ]
    if payload.get("dataset_root"):
        common_tail.extend(["--dataset-root", str(payload["dataset_root"])])
    if payload.get("mds_server"):
        common_tail.extend(["--mds-server", str(payload["mds_server"])])
    if payload.get("k_start") is not None:
        common_tail.extend(["--k-start", str(int(payload["k_start"]))])
    if payload.get("k_stop") is not None:
        common_tail.extend(["--k-stop", str(int(payload["k_stop"]))])
    if payload.get("reuse_slices"):
        common_tail.append("--reuse-slices")
    if payload.get("skip_existing"):
        common_tail.append("--skip-existing")
    if payload.get("overlay_only"):
        common_tail.append("--overlay-only")
    ew = int(payload.get("eval_workers") or 1)
    if ew > 1:
        common_tail.extend(["--eval-workers", str(ew)])
    if payload.get("freegsnke_cpu"):
        common_tail.append("--freegsnke-cpu")
    if payload.get("freegsnke_omp_threads") is not None:
        common_tail.extend(["--freegsnke-omp-threads", str(int(payload["freegsnke_omp_threads"]))])

    if payload.get("run_pcs", True):
        cmd_pcs = (
            ["-m", "bc.batch_freegsnke.run_freegsnke_whole_shot", "--coil-source", "pcs"]
            + common_tail
        )
        rc = _run(cmd_pcs)
        rc_max = max(rc_max, rc)

    if payload.get("run_bc", True):
        cmd_bc = (
            ["-m", "bc.batch_freegsnke.run_freegsnke_whole_shot", "--coil-source", "bc_pred"]
            + common_tail
            + ["--ckpt", str(payload["ckpt"])]
        )
        dev = payload.get("infer_device")
        if dev:
            cmd_bc.extend(["--device", str(dev)])
        rc = _run(cmd_bc)
        rc_max = max(rc_max, rc)

    if payload.get("run_montage", True):
        cmd_m = [
            "-m",
            "bc.batch_freegsnke.montage_best_per_phase",
            "--shot",
            str(shot),
            "--precursor-root",
            str(prec_root),
            "--whole-shot-root",
            str(payload["whole_shot_root"]),
            "--by-t-efit-subdir",
            str(payload["by_t_efit_subdir"]),
            "--out-dir",
            str(payload["montage_out_dir"]),
            "--metric",
            str(payload.get("montage_metric", "lcfs_rmsep")),
        ]
        if payload.get("montage_converged_only"):
            cmd_m.append("--converged-only")
        rc = _run(cmd_m)
        rc_max = max(rc_max, rc)

    return {"shot": shot, "rc": min(rc_max, 255), "msg": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--precursors-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_precursors"),
    )
    ap.add_argument(
        "--shots",
        type=str,
        default=None,
        help="逗号分隔炮号；默认扫描 ``--precursors-root`` 下全部含 precursor.npz 的目录",
    )
    ap.add_argument(
        "--whole-shot-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_whole_shot"),
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_REPO / "results" / "bc_v1" / "run1" / "checkpoints" / "best_val.pt"),
    )
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--mds-server", type=str, default=None)
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    ap.add_argument("--by-t-efit-subdir", type=str, default="by_t_efit")
    ap.add_argument("--slices-subdir", type=str, default="_slices")
    ap.add_argument("--k-start", type=int, default=None)
    ap.add_argument("--k-stop", type=int, default=None)
    ap.add_argument("--reuse-slices", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overlay-only", action="store_true", help="传给 whole_shot（推荐节省磁盘）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数（每进程一炮）")
    ap.add_argument(
        "--gpu-parallel",
        action="store_true",
        help="多进程时仍不强制 BC 用 CPU（可能多进程抢同一块 GPU）",
    )
    ap.add_argument(
        "--infer-device",
        type=str,
        default=None,
        help="强制写入 bc_pred 的 ``--device``（如 cuda:0 / cpu）；默认由 workers 与 --gpu-parallel 推断",
    )
    ap.add_argument("--skip-pcs", action="store_true")
    ap.add_argument("--skip-bc", action="store_true")
    ap.add_argument("--skip-montage", action="store_true")
    ap.add_argument("--montage-out-dir", type=str, default=None)
    ap.add_argument(
        "--montage-metric",
        type=str,
        choices=("lcfs_rmsep", "forward_rel_residual"),
        default="lcfs_rmsep",
    )
    ap.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="传给 run_freegsnke_whole_shot：同时跑多少个 freegsnke 子进程",
    )
    ap.add_argument(
        "--freegsnke-cpu",
        action="store_true",
        help="freegsnke 子进程遮罩 GPU（与多卡/多进程 BC 分工时推荐开启）",
    )
    ap.add_argument(
        "--freegsnke-omp-threads",
        type=int,
        default=None,
        help="写入子进程 OMP_NUM_THREADS；默认由 whole_shot 在 --freegsnke-cpu 下设为 1",
    )

    prec_root = Path(args.precursors_root).resolve()
    whole_root = Path(args.whole_shot_root).resolve()
    montage_out = Path(args.montage_out_dir).resolve() if args.montage_out_dir else (whole_root / "_montage")

    if args.shots:
        shots = sorted(set(_parse_shots_csv(args.shots)))
    else:
        shots = _list_precursor_shots(prec_root)
    if not shots:
        raise SystemExit(f"未找到炮号：{prec_root}")

    infer_dev: str | None
    if args.infer_device:
        infer_dev = str(args.infer_device)
    elif int(args.workers) > 1 and not bool(args.gpu_parallel):
        infer_dev = "cpu"
    else:
        infer_dev = None

    base: dict[str, Any] = {
        "repo": str(_REPO),
        "python": sys.executable,
        "precursor_root": str(prec_root),
        "whole_shot_root": str(whole_root),
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_root": args.dataset_root,
        "mds_server": args.mds_server,
        "infer_device": infer_dev,
        "rtol": float(args.rtol),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "by_t_efit_subdir": str(args.by_t_efit_subdir),
        "slices_subdir": str(args.slices_subdir),
        "k_start": args.k_start,
        "k_stop": args.k_stop,
        "reuse_slices": bool(args.reuse_slices),
        "skip_existing": bool(args.skip_existing),
        "overlay_only": bool(args.overlay_only),
        "dry_run": bool(args.dry_run),
        "run_pcs": not bool(args.skip_pcs),
        "run_bc": not bool(args.skip_bc),
        "run_montage": not bool(args.skip_montage),
        "montage_out_dir": str(montage_out),
        "montage_metric": str(args.montage_metric),
        "montage_converged_only": bool(args.montage_converged_only),
        "eval_workers": int(args.eval_workers),
        "freegsnke_cpu": bool(args.freegsnke_cpu),
        "freegsnke_omp_threads": args.freegsnke_omp_threads,
        "env_extra": {},
    }

    print(
        f"[batch_precursors_parallel] shots={len(shots)} workers={int(args.workers)} "
        f"bc_device={infer_dev or 'default'} precursors_root={prec_root}",
        flush=True,
    )

    payloads = [{**base, "shot": s} for s in shots]
    worst = 0
    if int(args.workers) <= 1:
        for p in payloads:
            r = _worker_one_shot(p)
            print(f"shot={r['shot']} rc={r['rc']} {r.get('msg', '')}", flush=True)
            worst = max(worst, int(r["rc"]))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(_worker_one_shot, p): p["shot"] for p in payloads}
            for fu in as_completed(futs):
                shot_id = futs[fu]
                try:
                    r = fu.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"shot={shot_id} FAIL {exc}", flush=True)
                    worst = max(worst, 1)
                    continue
                print(f"shot={r['shot']} rc={r['rc']}", flush=True)
                worst = max(worst, int(r["rc"]))

    return min(worst, 255) if worst > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
