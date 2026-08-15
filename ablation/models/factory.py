"""build_model: 严格校验 model config, 拼写错误立即报错 (不静默吞键).

cfg_model 须含 "name" (transformer/lstm/mlp) + 该模型 ctor 接受的键 (pe_mode / causal / ...).
未知键 -> ValueError, 防止 pe_mod / 残留 use_time_pe 等拼写错误被默默丢弃导致静默错结果.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn

from ablation.models.transformer import CausalTransformerModel
from ablation.models.lstm import CausalLSTM
from ablation.models.mlp import PointwiseMLP

_REGISTRY: dict[str, type[nn.Module]] = {
    "transformer": CausalTransformerModel,
    "lstm": CausalLSTM,
    "mlp": PointwiseMLP,
}


def build_model(cfg_model: dict[str, Any]) -> nn.Module:
    cfg = dict(cfg_model)
    if "name" not in cfg:
        raise ValueError(f"model config missing 'name'; allowed={list(_REGISTRY)}")
    name = cfg.pop("name")
    if name not in _REGISTRY:
        raise ValueError(f"unknown model.name={name!r}; allowed={list(_REGISTRY)}")
    cls = _REGISTRY[name]

    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(
            f"model {name!r}: unknown keys {sorted(unknown)}; allowed={sorted(allowed)}"
        )

    assert cfg.get("d_state", 19) == 19, (
        "ablation 必须用 19 维 state (meta/norm_stats_notime.npz); "
        f"got d_state={cfg.get('d_state')!r}"
    )
    return cls(**cfg)
