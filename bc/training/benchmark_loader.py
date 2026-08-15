"""DataLoader 参数基准: 测 5 组 (batch_size, num_workers, pin_memory, ...) 的吞吐.

跑在计算节点上 (login 节点没有 torch DCU runtime). 输出 csv 到
results/bc_v1/loader_benchmark.csv, 用最高吞吐的一组写进 configs/bc_v1.yaml.

Usage (在计算节点 / 单卡):
    python -m bc.benchmark_loader \
        --shots-file meta/train_shots.txt \
        --norm-stats meta/norm_stats.npz \
        --out results/bc_v1/loader_benchmark.csv

参考默认 5 组 (基于 plans/bc_transformer_v1.md §7):
    v1: bs=32  workers=4  pin=T prefetch=2 persistent=T
    v2: bs=64  workers=8  pin=T prefetch=4 persistent=T
    v3: bs=128 workers=16 pin=T prefetch=4 persistent=T   (推荐起点)
    v4: bs=128 workers=32 pin=T prefetch=4 persistent=T
    v5: bs=128 workers=16 pin=F prefetch=2 persistent=T
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


CONFIGS = [
    dict(name="v1", batch_size=32, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True),
    dict(name="v2", batch_size=64, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True),
    dict(name="v3", batch_size=128, num_workers=16, pin_memory=True, prefetch_factor=4, persistent_workers=True),
    dict(name="v4", batch_size=128, num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True),
    dict(name="v5", batch_size=128, num_workers=16, pin_memory=False, prefetch_factor=2, persistent_workers=True),
]


def _bench_one(cfg, shots, dataset_root, norm_stats, n_steps, device):
    import torch  # local import; only used at runtime
    from torch.utils.data import DataLoader

    from bc.data.dataset import PFDataset

    ds = PFDataset(shots, dataset_root=dataset_root, norm_stats_path=norm_stats)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        prefetch_factor=cfg["prefetch_factor"],
        persistent_workers=cfg["persistent_workers"],
        shuffle=True,
        drop_last=True,
    )

    # warmup 1 batch
    it = iter(loader)
    _ = next(it)

    t0 = time.monotonic()
    moved_steps = 0
    for step, batch in enumerate(it, start=1):
        if device != "cpu":
            for k, v in batch.items():
                if hasattr(v, "to"):
                    batch[k] = v.to(device, non_blocking=cfg["pin_memory"])
        moved_steps += 1
        if moved_steps >= n_steps:
            break
    elapsed = time.monotonic() - t0
    steps_per_s = moved_steps / elapsed if elapsed > 0 else float("inf")
    samples_per_s = steps_per_s * cfg["batch_size"]

    # peak GPU mem
    if device != "cpu":
        try:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            peak_mb = float("nan")
    else:
        peak_mb = float("nan")

    return dict(
        config=cfg["name"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        prefetch_factor=cfg["prefetch_factor"],
        persistent_workers=cfg["persistent_workers"],
        steps=moved_steps,
        elapsed_s=round(elapsed, 3),
        steps_per_s=round(steps_per_s, 2),
        samples_per_s=round(samples_per_s, 1),
        peak_gpu_mb=round(peak_mb, 1) if peak_mb == peak_mb else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-file", type=Path, default=Path("meta/train_shots.txt"))
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/PF_ATIME_dataset"),
    )
    ap.add_argument("--norm-stats", type=Path, default=Path("meta/norm_stats.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/bc_v1/loader_benchmark.csv"))
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0 to time GPU pin/transfer; cpu to time pure IO",
    )
    args = ap.parse_args()

    from bc.data.dataset import read_shots_txt

    shots = read_shots_txt(args.shots_file)
    # use first 4096 shots to keep wall-clock < 10 min total
    shots = shots[:4096]
    print(f"benchmark on {len(shots)} shots, n_steps={args.n_steps}, device={args.device}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        print(f"\n>>> {cfg['name']}: {cfg}")
        try:
            r = _bench_one(cfg, shots, args.dataset_root, args.norm_stats, args.n_steps, args.device)
        except Exception as e:
            r = dict(cfg, error=f"{type(e).__name__}:{e}")
        print(f"    -> {r}")
        rows.append(r)

    # write csv
    keys = sorted({k for r in rows for k in r.keys()})
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    # pick best
    ok = [r for r in rows if "error" not in r and r.get("steps_per_s")]
    if ok:
        best = max(ok, key=lambda r: r["steps_per_s"])
        print(f"\nbest config: {best['config']} "
              f"({best['samples_per_s']:.0f} samples/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
