"""CausalTransformerModel: causal Transformer, 受 pe_mode / causal 开关控制.

消融用 (见 plans/paper-mypaper-pdf-time-encoding-time-transient-diffie.md):
- pe_mode: "time" (sinusoidal time PE, 论文 "time encoding") / "learnable" (序数可学习 PE, 非时间) / "none".
- causal: True = 因果 (causal mask, 提供**序数**顺序); False = 双向 full attention (PE 成为唯一顺序源).

forward(state, time_phys, token_mask) -> (B, T, d_action) 签名与 bc 一致, 训练/评测循环零改动。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ablation.models.pe import time_sinusoidal_pe


class CausalTransformerModel(nn.Module):
    """Causal Transformer encoder for state -> action regression."""

    def __init__(
        self,
        d_state: int = 19,
        d_action: int = 12,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        T_max: int = 128,
        pe_base_period_s: float = 20.0,
        pe_mode: str = "time",
        causal: bool = True,
    ):
        super().__init__()
        assert pe_mode in ("none", "time", "learnable"), f"bad pe_mode={pe_mode!r}"
        self.d_state = d_state
        self.d_action = d_action
        self.d_model = d_model
        self.T_max = T_max
        self.pe_mode = pe_mode
        self.causal = causal
        self.pe_base_period_s = pe_base_period_s

        self.state_embed = nn.Linear(d_state, d_model)
        # learnable PE 仅在 pe_mode=="learnable" 注册 —— 否则 none/time 配置里它是未用参数,
        # DDP find_unused_parameters=false 会报错.
        if pe_mode == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(T_max, d_model))

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

        if causal:
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
            time_phys:  (B, T)           物理秒, 未归一化 (pe_mode!="time" 时忽略)
            token_mask: (B, T)           bool, True=real token, False=pad
        Returns:
            action: (B, T, d_action)     归一化域
        """
        T = state.size(1)
        if T > self.T_max:
            raise ValueError(f"T={T} > T_max={self.T_max}")

        x = self.state_embed(state)
        if self.pe_mode == "time":
            x = x + time_sinusoidal_pe(time_phys, self.d_model, self.pe_base_period_s)
        elif self.pe_mode == "learnable":
            x = x + self.pos_embed[:T]
        # "none": 不加任何位置编码

        kw = dict(src_key_padding_mask=~token_mask)
        if self.causal:  # False=双向 full attention
            kw["mask"] = self.causal_mask[:T, :T]
            kw["is_causal"] = True
        out = self.transformer(x, **kw)
        return self.head(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
