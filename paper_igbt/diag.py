"""诊断 s44: 权重是否真的加载了 + 模型输出是否和 target 相关。"""
import sys, numpy as np
sys.path.insert(0, ".")
import torch
from paper_igbt.model import load_best_model
from ablation.data.dataset import PFDataset, load_norm_stats, read_shots_txt
from bc.training.utils import deep_get

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, cfg, extra = load_best_model(device=device)
# 1) 权重统计
print("=== 权重检查 (trained 应非零/非初始) ===")
print(f"  pos_embed: min={model.pos_embed.data.min():.4f} max={model.pos_embed.data.max():.4f} std={model.pos_embed.data.std():.4f} (trained std>0.01)")
print(f"  state_embed.weight: std={model.state_embed.weight.std():.4f}")
print(f"  head.weight: std={model.head.weight.std():.4f}  bias_mean={model.head.bias.mean():.4f}")
# 2) 单炮 forward
shots = read_shots_txt(deep_get(cfg, "data.test_shots"))[:3]
ds = PFDataset(shots, dataset_root=deep_get(cfg,"data.dataset_root"), norm_stats_path=deep_get(cfg,"data.norm_stats"), T_max=128)
for i in range(3):
    s = ds[i]; T = int(s["T"])
    state = s["state"].unsqueeze(0).to(device); tphys = s["time_phys"].unsqueeze(0).to(device); tmask = s["token_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(state, tphys, tmask)
    tgt = s["action"][:T]
    p = pred[0,:T].cpu().numpy(); t = tgt.numpy()
    corr = np.corrcoef(p.flatten(), t.flatten())[0,1]
    print(f"  shot {shots[i]}: pred_norm mean={p.mean():.3f} std={p.std():.3f} | tgt_norm mean={t.mean():.3f} std={t.std():.3f} | corr={corr:.4f}")
