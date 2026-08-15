"""兼容入口：实现位于 ``bc.models.losses``。"""
from __future__ import annotations

from bc.models.losses import masked_mae, masked_mse, per_channel_r2

__all__ = ["masked_mse", "masked_mae", "per_channel_r2"]
