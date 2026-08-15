"""最终确认: PF6 误差是否由"高 PF6 真值区间的系统性欠预测"驱动, 而非破裂炮。

1. 全 PF6 样本按真值分箱, 每箱报 n / pred均值 / bias / 局部RMSE / 真值范围。
2. 把 worst-30 炮的 PF6 点投影到真值轴, 看它们是否都落在高真值箱。
3. 反向 sanity: 只保留 |PF6 truth| <= 3 kA 的"低电流"样本, R² 是多少? (若很高 → 证实是高电流尾拖累)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import _metrics_1d, load_records  # noqa

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
PF6 = 5


def main():
    recs = load_records(PRED)
    # worst-30 shot set (from previous diag)
    worst30 = {156947,156948,156949,156951,156952,156953,156956,156965,156966,156987,
               157050,157051,157098,157538,157587,157774,157775,157776,158367,158471,
               158877,158878,159112,159458,159459,159540,159565,159694,159698,159701}

    t_all, p_all, isworst = [], [], []
    for r in recs:
        m = r["action_mask"][:, PF6]
        t_all.append(r["target_kA"][m, PF6])
        p_all.append(r["pred_kA"][m, PF6])
        isworst.append(np.full(int(m.sum()), r["shot"] in worst30))
    t = np.concatenate(t_all); p = np.concatenate(p_all); w = np.concatenate(isworst)

    print(f"PF6 global: n={t.size}  R²={_metrics_1d(p,t)[0]:.4f}")
    print(f"PF6 truth range: [{t.min():.2f}, {t.max():.2f}] kA  mean={t.mean():.2f} std={t.std():.2f}")
    print(f"  truth>5kA: {(t>5).sum()} pts ({100*(t>5).mean():.1f}%)   truth>7kA: {(t>7).sum()} ({100*(t>7).mean():.1f}%)")

    # 分箱
    edges = [-15,-10,-7,-5,-3,-1,0,1,3,5,7,10,15]
    print("\n=== PF6 按【真值】分箱: n / pred均值 / bias / 局部RMSE ===")
    print(f"{'truth_bin':>12s} {'n':>7s} {'frac%':>6s} {'pred_mean':>9s} {'bias':>8s} {'rmse':>7s} {'worst30%':>8s}")
    for a,b in zip(edges[:-1], edges[1:]):
        sel = (t>=a)&(t<b)
        if sel.sum()==0: continue
        tt, pp = t[sel], p[sel]
        print(f"[{a:>4d},{b:>3d}) {sel.sum():>7d} {100*sel.mean():>5.1f}% {pp.mean():>9.2f} "
              f"{(pp-tt).mean():>+8.2f} {np.sqrt(((pp-tt)**2).mean()):>7.2f} {100*w[sel].mean():>7.1f}")

    # sanity: 低电流子集 R²
    for thr in (3.0, 5.0):
        sel = np.abs(t) <= thr
        print(f"\n|truth|<={thr:.0f}kA: n={sel.sum()} ({100*sel.mean():.1f}%)  R²={_metrics_1d(p[sel],t[sel])[0]:.4f}  "
              f"RMSE={np.sqrt(((p[sel]-t[sel])**2).mean()):.3f}")

    # 去掉 worst30 后 (客观上不靠模型误差, 而是看它们是否本就该剔除)
    keep = ~w.astype(bool)
    print(f"\n去掉 worst-30 炮后 PF6: n={keep.sum()}  R²={_metrics_1d(p[keep],t[keep])[0]:.4f}  "
          f"(对比 baseline {_metrics_1d(p,t)[0]:.4f})")

    # worst30 炮的 PF6 真值分布
    print(f"\nworst-30 炮 PF6 真值: mean={t[w.astype(bool)].mean():.2f}  >=7kA 占比={100*(t[w.astype(bool)]>=7).mean():.1f}%")


if __name__ == "__main__":
    main()
