"""Sinusoidal positional encoding based on physical time (seconds).

逐字取自 bc/models/model.py 的 time_sinusoidal_pe, 供三类模型 pe_mode=="time" 共用。
"""

from __future__ import annotations

import math

import torch


def time_sinusoidal_pe(
    time_phys: torch.Tensor,
    d_model: int,
    base_period_s: float = 20.0,
) -> torch.Tensor:
    """每个 token 的物理时间 -> sinusoidal positional encoding.

    Args:
        time_phys: (B, T) 物理时间, 秒. 未归一化.
        d_model:   编码维度
        base_period_s: 最长周期 (秒). EAST 单炮 ~10-15s, 取 20s 留余量.
    Returns:
        (B, T, d_model)  与 time_phys.dtype 一致
    """
    half = d_model // 2
    device = time_phys.device
    inv_freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * -(math.log(base_period_s) / max(half - 1, 1))
    )  # (half,)
    args = time_phys.unsqueeze(-1).float() * inv_freq * (2.0 * math.pi)  # (B, T, half)
    pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, T, 2*half)
    if d_model % 2 == 1:
        pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
    return pe.to(time_phys.dtype)
