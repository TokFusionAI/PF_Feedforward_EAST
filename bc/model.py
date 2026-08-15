"""兼容入口：实现位于 ``bc.models.model``。"""
from __future__ import annotations

from bc.models.model import CausalTransformer, time_sinusoidal_pe

__all__ = ["CausalTransformer", "time_sinusoidal_pe"]
