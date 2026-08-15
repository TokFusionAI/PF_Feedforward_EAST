"""对比 PF6 (及全通道) 在 cross-β_N (paper_betan s44) vs IID (paper_best) 上的 R²。
若 IID 上 PF6 也差 → PF6 本身难(架构/损失问题, 可能可修);
若 IID 上 PF6 好 → betan 的欠预测是跨β_N 泛化gap (输入态不同, 非目标值OOD)。

同时: 把 betan 的 PF6 按【真值】分箱, 看 pred/truth 比 (压缩 vs 线性scaling)。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import PF_NAMES, _metrics_1d, load_records  # noqa

BETAN = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
IID = Path("results/paper_best/predictions/per_shot_preds.npz")


def per_channel(records):
    out = []
    for ch in range(12):
        ts = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
        ps = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
        out.append((_metrics_1d(ps, ts)[0], float(np.sqrt(np.mean((ps - ts) ** 2))), ts.size))
    return out


def main():
    print("loading betan (cross-β_N) ...", flush=True)
    rb = load_records(BETAN)
    print("loading IID (paper_best) ...", flush=True)
    ri = load_records(IID)
    mb = per_channel(rb); mi = per_channel(ri)
    print(f"\n{'chan':6s} {'betan_R²':>9s} {'IID_R²':>9s} {'betan_RMSE':>10s} {'IID_RMSE':>9s}")
    for ch in range(12):
        print(f"{PF_NAMES[ch]:6s} {mb[ch][0]:>9.4f} {mi[ch][0]:>9.4f} {mb[ch][1]:>10.3f} {mi[ch][1]:>9.3f}")

    # PF6 分箱 pred/truth 比 (betan) —— 压缩 vs 线性
    PF6 = 5
    t = np.concatenate([r["target_kA"][:, PF6][r["action_mask"][:, PF6]] for r in rb])
    p = np.concatenate([r["pred_kA"][:, PF6][r["action_mask"][:, PF6]] for r in rb])
    print("\n=== betan PF6: pred/truth ratio by truth bin (压缩/饱和 vs 线性) ===")
    edges = [-15,-5,-3,0,3,5,7,9,11,15]
    for a, b in zip(edges[:-1], edges[1:]):
        s = (t >= a) & (t < b)
        if s.sum() < 20: continue
        print(f"  truth[{a:>3d},{b:>3d}): n={s.sum():>6d}  pred_mean={p[s].mean():>6.2f}  "
              f"truth_mean={t[s].mean():>6.2f}  ratio={p[s].mean()/t[s].mean():.3f}")


if __name__ == "__main__":
    main()
