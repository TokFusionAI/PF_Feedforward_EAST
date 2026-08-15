"""PointwiseMLP: 逐 token (pointwise) MLP, 无时序上下文, 无 causal mask.

T 维当 batch 维处理 (每步独立 state -> action). pe_mode: "time"/"learnable"/"none".
这是下界基线: 若 PE 在这里也有用, 是纯输入特征效应; 若只在 Transformer 有用, 是 attention+PE 交互.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ablation.models.pe import time_sinusoidal_pe


class PointwiseMLP(nn.Module):
    def __init__(
        self,
        d_state: int = 19,
        d_action: int = 12,
        d_model: int = 256,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.1,
        T_max: int = 128,
        pe_base_period_s: float = 20.0,
        pe_mode: str = "time",
    ):
        super().__init__()
        assert pe_mode in ("none", "time", "learnable"), f"bad pe_mode={pe_mode!r}"
        self.d_model = d_model
        self.T_max = T_max
        self.pe_mode = pe_mode
        self.pe_base_period_s = pe_base_period_s

        self.state_embed = nn.Linear(d_state, d_model)
        if pe_mode == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(T_max, d_model))

        layers: list[nn.Module] = []
        prev = d_model
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, d_action)

    def forward(
        self,
        state: torch.Tensor,
        time_phys: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        T = state.size(1)
        x = self.state_embed(state)
        if self.pe_mode == "time":
            x = x + time_sinusoidal_pe(time_phys, self.d_model, self.pe_base_period_s)
        elif self.pe_mode == "learnable":
            x = x + self.pos_embed[:T]
        x = self.mlp(x)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
