"""DDP 训练主入口.

单卡 smoke:
    python -m bc.train --config configs/bc_v1_smoke.yaml --tag smoke

8 卡 DDP smoke (1 epoch):
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        -m bc.train --config configs/bc_v1.yaml --tag smoke8 --override optim.epochs=1

8 卡正式训练:
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        -m bc.train --config configs/bc_v1.yaml --tag run1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from bc.common.constants import HARD_LIMIT_DIDT_KAPS, HARD_LIMIT_I_KA
from bc.data.dataset import PFDataset, read_shots_txt
from bc.models.losses import masked_mse, per_channel_r2
from bc.models.model import CausalTransformer
from .utils import (
    apply_overrides,
    count_parameters,
    deep_get,
    load_yaml,
    save_ckpt,
    set_seed,
    setup_logger,
)


# ----------------------------- DDP / device ----------------------------- #


def is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_ddp(backend: str) -> tuple[int, int, int, str]:
    """Returns (rank, world_size, local_rank, device_str)."""
    if is_distributed():
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        rank = 0
        world = 1
        local_rank = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return rank, world, local_rank, device


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return t


# ----------------------------- data builders ----------------------------- #


def build_loaders(cfg: dict, rank: int, world: int) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    train_shots = read_shots_txt(deep_get(cfg, "data.train_shots"))
    val_shots = read_shots_txt(deep_get(cfg, "data.val_shots"))
    n_tr = deep_get(cfg, "data.train_subset_n")
    n_va = deep_get(cfg, "data.val_subset_n")
    if n_tr:
        train_shots = train_shots[: int(n_tr)]
    if n_va:
        val_shots = val_shots[: int(n_va)]

    train_ds = PFDataset(
        train_shots,
        dataset_root=deep_get(cfg, "data.dataset_root"),
        norm_stats_path=deep_get(cfg, "data.norm_stats"),
        T_max=deep_get(cfg, "data.T_max", 128),
    )
    val_ds = PFDataset(
        val_shots,
        dataset_root=deep_get(cfg, "data.dataset_root"),
        norm_stats_path=deep_get(cfg, "data.norm_stats"),
        T_max=deep_get(cfg, "data.T_max", 128),
    )

    if world > 1:
        train_sampler: DistributedSampler | None = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler: DistributedSampler | None = DistributedSampler(
            val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None

    common = dict(
        num_workers=int(deep_get(cfg, "data.num_workers", 4)),
        pin_memory=bool(deep_get(cfg, "data.pin_memory", True)),
        prefetch_factor=int(deep_get(cfg, "data.prefetch_factor", 2)),
        persistent_workers=bool(deep_get(cfg, "data.persistent_workers", False)),
    )
    bs = int(deep_get(cfg, "optim.batch_per_rank", 32))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler


# ----------------------------- train / eval ----------------------------- #


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    sched: Any,
    scaler: Any,
    amp_dtype: torch.dtype,
    device: str,
    epoch: int,
    cfg: dict,
    logger,
    rank: int,
    tb,
) -> dict:
    model.train()
    log_every = int(deep_get(cfg, "log.log_every", 50))
    grad_clip = float(deep_get(cfg, "optim.grad_clip", 1.0))
    use_scaler = (amp_dtype == torch.float16)

    sum_loss = torch.zeros(1, device=device)
    n_batches = 0
    t_epoch = time.monotonic()

    for step, batch in enumerate(loader):
        state = batch["state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        time_phys = batch["time_phys"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32 and device != "cpu")):
            pred = model(state, time_phys, token_mask)
            loss = masked_mse(pred, action, action_mask, token_mask)

        opt.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step()

        sum_loss += loss.detach()
        n_batches += 1

        if rank == 0 and (step % log_every == 0):
            lr = opt.param_groups[0]["lr"]
            logger.info(
                f"epoch {epoch} step {step}/{len(loader)}  loss={loss.item():.4f}  lr={lr:.2e}"
            )
            if tb is not None:
                tb.add_scalar("train/loss_step", loss.item(), epoch * len(loader) + step)
                tb.add_scalar("train/lr", lr, epoch * len(loader) + step)

    avg_loss = (sum_loss / max(n_batches, 1)).item()
    elapsed = time.monotonic() - t_epoch
    if rank == 0:
        logger.info(f"epoch {epoch} train avg_loss={avg_loss:.4f}  elapsed={elapsed:.1f}s")
        if tb is not None:
            tb.add_scalar("train/loss_epoch", avg_loss, epoch)
            tb.add_scalar("train/elapsed_s", elapsed, epoch)
    return {"loss": avg_loss, "elapsed_s": elapsed}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    cfg: dict,
    norm_stats: dict[str, np.ndarray],
) -> dict:
    """val loss + per-channel R^2 + 物理违规率 (反归一化到 kA)."""
    model.eval()
    n_batches = 0
    sum_loss = torch.zeros(1, device=device)
    # per-channel SS_res / SS_tot accumulators in normalized domain (12,)
    ss_res = torch.zeros(12, device=device)
    ss_tot = torch.zeros(12, device=device)
    sum_n = torch.zeros(12, device=device)
    sum_target = torch.zeros(12, device=device)

    # phys violations
    n_total = torch.zeros(1, device=device)
    n_amp_v = torch.zeros(1, device=device)
    n_didt_v = torch.zeros(1, device=device)

    a_mean = torch.as_tensor(norm_stats["action_mean"], device=device)
    a_std = torch.as_tensor(norm_stats["action_std"], device=device)

    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        time_phys = batch["time_phys"].to(device, non_blocking=True)
        dt_phys = batch["dt_phys"].to(device, non_blocking=True)

        pred = model(state, time_phys, token_mask)
        loss = masked_mse(pred, action, action_mask, token_mask)
        sum_loss += loss.detach()
        n_batches += 1

        mask = action_mask & token_mask.unsqueeze(-1)  # (B, T, 12)
        # accumulate per-channel running mean of target -> SS_tot done at end
        sum_n += mask.sum(dim=(0, 1))
        sum_target += (action * mask).sum(dim=(0, 1))
        ss_res += ((pred - action) ** 2 * mask).sum(dim=(0, 1))

        # 物理量违规 (反归一化)
        pred_kA = (pred * a_std + a_mean) / 1e3   # (B, T, 12), kA
        # |I| > 14.5 kA
        amp_bad = pred_kA.abs() > HARD_LIMIT_I_KA
        n_amp_v += (amp_bad & mask).sum()
        # |dI/dt| > 4 kA/s using physical dt; compute step-to-step difference
        # dI/dt at index t = (pred[t] - pred[t-1]) / dt[t]
        if pred_kA.shape[1] >= 2:
            d_pred = pred_kA[:, 1:] - pred_kA[:, :-1]                  # (B, T-1, 12)
            dt_loc = dt_phys[:, 1:].clamp_min(1e-6).unsqueeze(-1)       # (B, T-1, 1)
            didt = d_pred / dt_loc
            mask_didt = mask[:, 1:] & token_mask[:, :-1].unsqueeze(-1)  # 必须前后都是 real
            didt_bad = didt.abs() > HARD_LIMIT_DIDT_KAPS
            n_didt_v += (didt_bad & mask_didt).sum()
        n_total += mask.sum()

    # SS_tot: sum (target - mean)^2 with mask. Need a 2nd pass to compute
    # mean accurately, but doing one pass with running mean is acceptable
    # since BC val data dist is stationary; use simple mean estimate:
    chan_mean = sum_target / sum_n.clamp_min(1)
    # (we'll re-iterate ss_tot by approximating: SS_tot = sum (target-mean)^2.
    #  exact form: sum t^2 - n * mean^2 ; we don't have sum t^2, but mse ratio is a
    #  cheap proxy.  For correct R^2 we need sum t^2, accumulate it.)
    # Re-pass for SS_tot for correctness:
    sum_target_sq = torch.zeros(12, device=device)
    for batch in loader:
        action = batch["action"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        mask = action_mask & token_mask.unsqueeze(-1)
        sum_target_sq += ((action * mask) ** 2).sum(dim=(0, 1))
    ss_tot = sum_target_sq - sum_n * chan_mean ** 2
    r2_chan = 1.0 - ss_res / ss_tot.clamp_min(1e-12)

    # all-reduce across DDP ranks
    if dist.is_available() and dist.is_initialized():
        for t in (sum_loss, sum_n, ss_res, sum_target, sum_target_sq,
                  n_total, n_amp_v, n_didt_v):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        # recompute after reduce
        chan_mean = sum_target / sum_n.clamp_min(1)
        ss_tot = sum_target_sq - sum_n * chan_mean ** 2
        r2_chan = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
        avg_loss = (sum_loss / dist.get_world_size() / max(n_batches, 1)).item()
    else:
        avg_loss = (sum_loss / max(n_batches, 1)).item()

    return {
        "loss": float(avg_loss),
        "r2_per_channel": r2_chan.detach().cpu().numpy().tolist(),
        "r2_median": float(np.median(r2_chan.detach().cpu().numpy())),
        "amp_violation_rate": float((n_amp_v / n_total.clamp_min(1)).item()),
        "didt_violation_rate": float((n_didt_v / n_total.clamp_min(1)).item()),
    }


# ----------------------------- main ----------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--override", "-o", action="append", default=[],
                    help="dotted overrides, e.g. -o optim.epochs=1 -o data.num_workers=4")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    seed = int(deep_get(cfg, "seed", 20260424))
    set_seed(seed)

    backend = deep_get(cfg, "ddp.backend", "nccl")
    rank, world, local_rank, device = setup_ddp(backend)

    out_root = Path(deep_get(cfg, "log.out_root")) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root, rank)
    logger.info(f"world={world} rank={rank} local_rank={local_rank} device={device}")
    logger.info(f"cfg=\n{cfg}")

    train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world)
    logger.info(
        f"train batches/epoch={len(train_loader)}  val batches={len(val_loader)}  "
        f"train_size={len(train_loader.dataset)}  val_size={len(val_loader.dataset)}"
    )

    model = CausalTransformer(**cfg["model"]).to(device)
    n_params = count_parameters(model)
    logger.info(f"model parameters: {n_params:,}")

    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None,
                    find_unused_parameters=bool(deep_get(cfg, "ddp.find_unused_parameters", False)))
        model_no_ddp = model.module
    else:
        model_no_ddp = model

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(cfg, "optim.lr", 3e-4)),
        weight_decay=float(deep_get(cfg, "optim.weight_decay", 1e-2)),
        betas=tuple(deep_get(cfg, "optim.betas", [0.9, 0.95])),
    )
    epochs = int(deep_get(cfg, "optim.epochs", 60))
    total_steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=float(deep_get(cfg, "optim.lr", 3e-4)),
        total_steps=total_steps,
        pct_start=float(deep_get(cfg, "optim.warmup_pct", 0.05)),
        anneal_strategy="cos",
    )

    amp_name = deep_get(cfg, "optim.amp_dtype", "bf16")
    amp_dtype = _amp_dtype(amp_name)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    tb = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(deep_get(cfg, "log.tb_dir") + f"/{args.tag}")
        except Exception as e:
            logger.warning(f"TensorBoard unavailable: {e}")

    norm_stats = np.load(deep_get(cfg, "data.norm_stats"))
    norm_dict = {k: norm_stats[k] for k in ("state_mean", "state_std", "action_mean", "action_std")}

    save_every = int(deep_get(cfg, "log.save_every_epoch", 5))

    best_val = float("inf")
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, train_loader, opt, sched, scaler, amp_dtype,
            device, epoch, cfg, logger, rank, tb,
        )
        val_metrics = evaluate(model_no_ddp, val_loader, device, cfg, norm_dict)
        if rank == 0:
            logger.info(
                f"epoch {epoch} VAL loss={val_metrics['loss']:.4f} "
                f"R^2_med={val_metrics['r2_median']:.4f} "
                f"amp_v={val_metrics['amp_violation_rate']:.4f} "
                f"didt_v={val_metrics['didt_violation_rate']:.4f}"
            )
            if tb is not None:
                tb.add_scalar("val/loss", val_metrics["loss"], epoch)
                tb.add_scalar("val/r2_median", val_metrics["r2_median"], epoch)
                tb.add_scalar("val/amp_violation_rate", val_metrics["amp_violation_rate"], epoch)
                tb.add_scalar("val/didt_violation_rate", val_metrics["didt_violation_rate"], epoch)
                for i, r2 in enumerate(val_metrics["r2_per_channel"]):
                    tb.add_scalar(f"val/r2_ch{i}", r2, epoch)

            if epoch % save_every == 0 or epoch == epochs - 1:
                save_ckpt(
                    out_root / "checkpoints", f"ep_{epoch}",
                    model_no_ddp.state_dict(),
                    opt.state_dict(), sched.state_dict(),
                    extra={"epoch": epoch, "val_loss": val_metrics["loss"], "cfg": cfg},
                )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_ckpt(
                    out_root / "checkpoints", "best_val",
                    model_no_ddp.state_dict(),
                    opt.state_dict(), sched.state_dict(),
                    extra={"epoch": epoch, "val_loss": val_metrics["loss"], "cfg": cfg},
                )

        maybe_barrier()

    if tb is not None:
        tb.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
