#!/usr/bin/env python3
"""多 GPU 分片：每卡独立进程 + 串行多炮；每炮内 freegsnke 用 CPU 线程池并行且子进程不碰 GPU。

适用场景：例如 **8 张卡、24 炮**，每卡分到约 3 炮（按炮列表 **轮询** 分桶）。

- **顶层**：``multiprocessing`` 启 ``len(gpu_ids)`` 个进程；第 ``i`` 个进程设
  ``CUDA_VISIBLE_DEVICES=<gpu_ids[i]>``，只看见 **一张物理卡**，其上 **BC 推理用 cuda:0**。
- **freegsnke**：沿用 ``run_freegsnke_whole_shot`` 的 ``--eval-workers`` +
  ``--freegsnke-cpu``（子进程 ``CUDA_VISIBLE_DEVICES=``），使前向平衡只在 **CPU** 上跑，
  **不与该卡上的 BC 抢 GPU**（BC 仅在各炮的 bc_pred 整段开头跑少量 GPU）。

**CPU 并发量**（经验上避免超额订阅）：单卡进程内并发约 ``eval_workers`` 个 freegsnke；
全机约 ``len(gpu_ids) * eval_workers``；若每进程 OpenMP 再开多线程，可设
``--freegsnke-omp-threads 1`` 并与机器总核数对照调节 ``--eval-workers``。

示例（24 炮、8 卡，每卡 4 路 freegsnke CPU 子进程）::

    python -m bc.batch_freegsnke.batch_precursors_gpu_sharded \\
      --precursors-root results/freegsnke_precursors \\
      --shots "$(paste -sd, shots24.txt)" \\
      --gpu-ids 0,1,2,3,4,5,6,7 \\
      --eval-workers 4 --freegsnke-cpu --overlay-only

不写 ``--shots`` 时扫描 precursors 下全部炮号（仍按 8 桶轮询）。

与 ``batch_precursors_parallel`` 的区别：后者是「一炮一进程」抢 GPU；本脚本保证 **一进程一卡**，
适合「卡数固定、要均分炮」的集群节点。
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bc.batch_freegsnke.batch_precursors_parallel import (
    _list_precursor_shots,
    _parse_shots_csv,
    _worker_one_shot,
)

_REPO = Path(__file__).resolve().parent.parent.parent


def _gpu_shard_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """子进程：绑定单卡，再按顺序跑分配给本卡的若干炮。"""
    gpu_id = int(payload["gpu_id"])
    shots = list(payload["shots"])
    base: dict[str, Any] = dict(payload["base"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    base["infer_device"] = "cuda:0"
    worst = 0
    for shot in shots:
        r = _worker_one_shot({**base, "shot": int(shot)})
        worst = max(worst, int(r["rc"]))
        print(f"[gpu{gpu_id}] shot={r['shot']} rc={r['rc']}", flush=True)
    return {"gpu_id": gpu_id, "rc": min(worst, 255), "n_shots": len(shots)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precursors-root", type=str, default=str(_REPO / "results" / "freegsnke_precursors"))
    ap.add_argument("--shots", type=str, default=None)
    ap.add_argument("--num-gpus", type=int, default=8, help="未指定 --gpu-ids 时用 0..N-1")
    ap.add_argument("--gpu-ids", type=str, default=None, help="逗号分隔物理卡号，长度即分片进程数")
    ap.add_argument("--whole-shot-root", type=str, default=str(_REPO / "results" / "freegsnke_whole_shot"))
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
    ap.add_argument("--overlay-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument(
        "--freegsnke-cpu",
        action="store_true",
        default=True,
        help="默认开启：freegsnke 子进程不占 GPU（可用 --no-freegsnke-cpu 关闭）",
    )
    ap.add_argument("--no-freegsnke-cpu", action="store_false", dest="freegsnke_cpu")
    ap.add_argument("--freegsnke-omp-threads", type=int, default=None)
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
    ap.add_argument("--montage-converged-only", action="store_true")
    args = ap.parse_args()

    prec_root = Path(args.precursors_root).resolve()
    whole_root = Path(args.whole_shot_root).resolve()
    montage_out = Path(args.montage_out_dir).resolve() if args.montage_out_dir else (whole_root / "_montage")

    if args.shots:
        shots = sorted(set(_parse_shots_csv(args.shots)))
    else:
        shots = _list_precursor_shots(prec_root)
    if not shots:
        raise SystemExit(f"未找到炮号：{prec_root}")

    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip().lstrip("-").isdigit()]
    else:
        gpu_ids = list(range(int(args.num_gpus)))
    if not gpu_ids:
        raise SystemExit("gpu-ids 为空")

    G = len(gpu_ids)
    buckets: list[list[int]] = [[] for _ in range(G)]
    for idx, s in enumerate(shots):
        buckets[idx % G].append(s)

    base: dict[str, Any] = {
        "repo": str(_REPO),
        "python": sys.executable,
        "precursor_root": str(prec_root),
        "whole_shot_root": str(whole_root),
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_root": args.dataset_root,
        "mds_server": args.mds_server,
        "infer_device": "cuda:0",
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
        f"[gpu_sharded] total_shots={len(shots)} shards={G} gpu_ids={gpu_ids} "
        f"eval_workers={args.eval_workers} freegsnke_cpu={args.freegsnke_cpu} "
        f"buckets={[len(b) for b in buckets]}",
        flush=True,
    )

    payloads = [{"gpu_id": gid, "shots": buckets[i], "base": base} for i, gid in enumerate(ggpu_ids)]
