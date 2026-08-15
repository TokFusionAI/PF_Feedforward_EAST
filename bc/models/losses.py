"""Masked MSE 与 per-channel metrics, 全部支持 (B, T, C) + 双层 mask.

设计要点见 plans/bc_transformer_v1.md §9.
"""

from __future__ import annotations

import torch


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    token_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Args:
        pred, target:  (B, T, C)
        action_mask:   (B, T, C)  per-channel valid
        token_mask:    (B, T)     per-token valid (pad-aware)
        reduction:     "mean" -> scalar; "per_channel" -> (C,)
    """
    mask = action_mask & token_mask.unsqueeze(-1)        # (B, T, C)
    diff2 = (pred - target) ** 2
    if reduction == "mean":
        return (diff2 * mask).sum() / mask.sum().clamp_min(1)
    elif reduction == "per_channel":
        return (diff2 * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1)
    elif reduction == "none":
        return diff2 * mask
    else:
        raise ValueError(f"unknown reduction={reduction!r}")


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    token_mask: torch.Tensor,
    reduction: str = "per_channel",
) -> torch.Tensor:
    mask = action_mask & token_mask.unsqueeze(-1)
    abs_diff = (pred - target).abs()
    if reduction == "mean":
        return (abs_diff * mask).sum() / mask.sum().clamp_min(1)
    elif reduction == "per_channel":
        return (abs_diff * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1)
    else:
        raise ValueError(f"unknown reduction={reduction!r}")


def per_channel_r2(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """
    R^2 = 1 - SS_res / SS_tot per channel (with mask).
    Returns (C,)
    """
    mask = action_mask & token_mask.unsqueeze(-1)
    n = mask.sum(dim=(0, 1)).clamp_min(1).float()
    mean_t = (target * mask).sum(dim=(0, 1)) / n
    ss_res = ((pred - target) ** 2 * mask).sum(dim=(0, 1))
    ss_tot = ((target - mean_t) ** 2 * mask).sum(dim=(0, 1)).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot
