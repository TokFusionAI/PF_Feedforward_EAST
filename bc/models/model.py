"""CausalTransformer 模型 (time-aware sinusoidal PE).

设计要点见 plans/bc_transformer_v1.md §8.

- 输入: state (B, T, 21), time_phys (B, T) 物理秒, token_mask (B, T)
- 输出: action (B, T, 12)
- attention causal (下三角 mask)
- PE 用未归一化的物理 time, 让不同 dt 的炮在 attention 几何上一致
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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


class CausalTransformer(nn.Module):
    """Causal Transformer encoder for state -> action regression."""

    def __init__(
        self,
        d_state: int = 21,
        d_action: int = 12,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        T_max: int = 128,
        pe_base_period_s: float = 20.0,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.d_model = d_model
        self.T_max = T_max
        self.pe_base_period_s = pe_base_period_s

        self.state_embed = nn.Linear(d_state, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, d_action)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(T_max, T_max), diagonal=1).bool(),
            persistent=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        time_phys: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state:      (B, T, d_state)  归一化后
            time_phys:  (B, T)           物理秒, 未归一化
            token_mask: (B, T)           bool, True=real token, False=pad
        Returns:
            action: (B, T, d_action)     归一化域
        """
        T = state.size(1)
        if T > self.T_max:
            raise ValueError(f"T={T} > T_max={self.T_max}")

        pe = time_sinusoidal_pe(time_phys, self.d_model, self.pe_base_period_s)
        x = self.state_embed(state) + pe
        out = self.transformer(
            x,
            mask=self.causal_mask[:T, :T],
            src_key_padding_mask=~token_mask,
            is_causal=True,
        )
        return self.head(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
