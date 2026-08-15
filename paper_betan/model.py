"""加载最优模型 (transformer_bidir_on, cross-betan best seed s44; 双向 Transformer + 物理时间正弦位置编码) —— 走 ablation.build_model (非 bc.CausalTransformer)。

s44 的 cfg["model"] = {name:transformer, pe_mode:time, causal:false, d_state:19, ...},
CausalTransformer(**cfg) 会崩 (不认 name/pe_mode/causal), 故必须 build_model。
seed 选法: 3 seed (11/44/20260424) 中 val_loss 最小者 = s44 (0.0106)。
"""
from __future__ import annotations

from pathlib import Path

import torch

from ablation.models.factory import build_model

BEST_CKPT = Path(
    "results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/checkpoints/best_val.pt"
)


def load_best_model(ckpt: str | Path = BEST_CKPT, device: str = "cuda:0"):
    """return (model, cfg, extra). cfg 含 data.dataset_root / data.norm_stats / data.test_shots 等。"""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["extra"]["cfg"]
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["extra"]
