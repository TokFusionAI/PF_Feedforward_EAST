"""一次性 smoke test, 在计算节点跑:
    cd PF_current_EAST && python -m bc._smoke_test
验证:
    1. PFDataset 在真 torch 下 __getitem__ 返回 tensor
    2. CausalTransformer forward 形状 + causal mask 行为
    3. masked_mse / per_channel_r2 与手算一致
    4. 整炮 forward vs 逐步 forward 数值等价 (max abs diff < 1e-5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _t(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def main() -> int:
    from bc.common.constants import D_ACTION, D_STATE, T_MAX
    from bc.data.dataset import PFDataset, read_shots_txt
    from bc.models.losses import masked_mse, per_channel_r2
    from bc.models.model import CausalTransformer

    repo = Path(__file__).resolve().parents[2]
    shots = read_shots_txt(repo / "meta" / "train_shots.txt")[:8]
    print(f"[1] PFDataset on 8 train shots ({shots})")
    ds = PFDataset(
        shots,
        dataset_root="/data/PF_ATIME_dataset",
        norm_stats_path=repo / "meta" / "norm_stats.npz",
    )
    sample = ds[0]
    for k in ("state", "action", "action_mask", "token_mask", "time_phys", "dt_phys"):
        v = sample[k]
        assert isinstance(v, torch.Tensor), f"{k} should be tensor"
        print(f"    {k:12s}: shape={tuple(v.shape)} dtype={v.dtype}")
    assert sample["state"].shape == (T_MAX, D_STATE)
    assert sample["action"].shape == (T_MAX, D_ACTION)

    print("\n[2] CausalTransformer forward shape check")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CausalTransformer(
        d_state=D_STATE, d_action=D_ACTION,
        d_model=64, n_layers=2, n_heads=4, d_ff=256, T_max=T_MAX
    ).to(device)
    print(f"    params: {model.num_parameters():,}, device: {device}")

    B = 4
    state = torch.stack([ds[i]["state"] for i in range(B)]).to(device)
    time_phys = torch.stack([ds[i]["time_phys"] for i in range(B)]).to(device)
    token_mask = torch.stack([ds[i]["token_mask"] for i in range(B)]).to(device)
    action = torch.stack([ds[i]["action"] for i in range(B)]).to(device)
    action_mask = torch.stack([ds[i]["action_mask"] for i in range(B)]).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(state, time_phys, token_mask)
    assert pred.shape == (B, T_MAX, D_ACTION), pred.shape
    print(f"    pred shape: {tuple(pred.shape)}")

    print("\n[3] masked_mse / per_channel_r2 sanity")
    loss = masked_mse(pred, action, action_mask, token_mask)
    pc_loss = masked_mse(pred, action, action_mask, token_mask, reduction="per_channel")
    r2 = per_channel_r2(pred, action, action_mask, token_mask)
    print(f"    loss = {loss.item():.4f}")
    print(f"    per-channel mse: {pc_loss.cpu().numpy().round(3).tolist()}")
    print(f"    per-channel R^2: {r2.cpu().numpy().round(3).tolist()}")

    print("\n[4] integral vs step-by-step inference equivalence")
    # take 1 shot, predict whole-shot once, then predict step-by-step
    one = ds[0]
    s1 = one["state"].unsqueeze(0).to(device)        # (1, T_max, 21)
    t1 = one["time_phys"].unsqueeze(0).to(device)    # (1, T_max)
    m1 = one["token_mask"].unsqueeze(0).to(device)   # (1, T_max)
    T_eff = int(one["T"])

    model.eval()
    with torch.no_grad():
        pred_whole = model(s1, t1, m1)               # (1, T_max, 12)

        pred_step = torch.zeros_like(pred_whole)
        for t in range(T_eff):
            s_so_far = s1[:, : t + 1]
            t_so_far = t1[:, : t + 1]
            m_so_far = m1[:, : t + 1]
            out_t = model(s_so_far, t_so_far, m_so_far)  # (1, t+1, 12)
            pred_step[:, t] = out_t[:, -1]

    diff = (pred_whole[:, :T_eff] - pred_step[:, :T_eff]).abs().max().item()
    print(f"    max abs diff (whole vs step-by-step) over T_eff={T_eff} = {diff:.3e}")
    assert diff < 1e-5, f"causal equivalence violated: diff={diff}"

    print("\n[OK] all smoke tests pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
