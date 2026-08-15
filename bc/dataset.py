"""兼容入口：实现位于 ``bc.data.dataset``。"""
from __future__ import annotations

from bc.data.dataset import PFDataset, build_sample_arrays, load_norm_stats, read_shots_txt

__all__ = ["PFDataset", "build_sample_arrays", "load_norm_stats", "read_shots_txt"]
