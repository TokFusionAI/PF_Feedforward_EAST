"""通用工具: seed / 配置加载 / checkpoint / 日志.

不依赖具体训练循环, 可被 train.py / eval.py / notebook 共用.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: dict, dotted: str, default=None):
    """deep_get(cfg, 'data.num_workers') -> cfg['data']['num_workers']"""
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """overrides like ['optim.epochs=1', 'data.num_workers=4'] (string values
    parsed via yaml.safe_load to keep types)."""
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be 'key=val', got {ov!r}")
        k, v = ov.split("=", 1)
        try:
            v_parsed = yaml.safe_load(v)
        except Exception:
            v_parsed = v
        deep_set(cfg, k.strip(), v_parsed)
    return cfg


def setup_logger(out_dir: Path, rank: int, name: str = "bc") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(f"%(asctime)s [r{rank}] %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if rank == 0:
        fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_ckpt(
    out_dir: Path,
    tag: str,
    model_state: dict,
    optim_state: dict | None = None,
    sched_state: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Atomic save (.tmp -> rename)."""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model_state,
        "optim": optim_state,
        "sched": sched_state,
        "extra": extra or {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_ckpt(path: str | Path, map_location=None) -> dict:
    import torch
    return torch.load(path, map_location=map_location, weights_only=False)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class CfgNamespace:
    """Lightweight wrapper, optional - most code uses raw dict for flexibility."""

    raw: dict = field(default_factory=dict)

    def get(self, dotted: str, default=None):
        return deep_get(self.raw, dotted, default)
