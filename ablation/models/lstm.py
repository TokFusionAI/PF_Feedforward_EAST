"""CausalLSTM: 标准 causal LSTM, 受 pe_mode 控制.

因果性来自 LSTM 左→右递归 (天然因果, 与 causal Transformer 公平).
pe_mode: "time" / "learnable" / "none".

不变性前提 (重要): real token 必须占序列前缀 [0, T_eff), 尾部 padding 不污染 real-token 输出
—— 仅因 nn.LSTM 严格左→右因果, 且 PFDataset 把 real data 写到 out[:T_eff].
换双向 LSTM 会破坏这一性质, 须改用 pack_padded_sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ablation.models.pe import time_sinusoidal_pe


class CausalLSTM(nn.Module):
    def __init__(
        self,
        d_state: int = 19,
        d_action: int = 12,
        d_model: int = 256,
        n_layers: int = 2,
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

        self.lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_model, d_action)

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
        out, _ = self.lstm(x)
        return self.head(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
