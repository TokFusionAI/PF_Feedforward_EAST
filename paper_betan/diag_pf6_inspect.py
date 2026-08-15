"""深入: 为什么 PF6 在某些炮上 R² 暴负? 这些炮是不是客观异常(破裂/截断)?

1. 对 PF6 最差的 TOP-N 炮, 打印 PF6 target/pred 的 min/max/mean/std, 看是否有离群/反号。
2. PF6 target 在该炮上是否近常数 (低 std) → R² 分母小, 少量噪声就暴负。
3. 去掉 TOP-K 最差炮后 PF6 全局 R² 的变化 (敏感性)。
4. PF6 逐炮 R² 与客观炮属性 (Ip_max, dur_s, tail_frac, shot 号) 的相关性。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import PF_NAMES, _metrics_1d, load_records  # noqa

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
PF6 = 5  # index of PF6


def main():
    print(f"loading {PRED} ...", flush=True)
    recs = load_records(PRED)
    print(f"loaded {len(recs)} shots", flush=True)

    # 逐炮 PF6 指标 + target/pred 统计
    rows = []
    for r in recs:
        m = r["action_mask"][:, PF6]
        if m.sum() == 0:
            continue
        p = r["pred_kA"][m, PF6]; t = r["target_kA"][m, PF6]
        ip = np.abs(r["Ip_A"])
        rows.append(dict(
            shot=r["shot"], n=int(m.sum()),
            t_min=float(t.min()), t_max=float(t.max()), t_mean=float(t.mean()), t_std=float(t.std()),
            p_min=float(p.min()), p_max=float(p.max()), p_mean=float(p.mean()), p_std=float(p.std()),
            rmse=float(np.sqrt(np.mean((p - t) ** 2))),
            r2=float(_metrics_1d(p, t)[0]),
            bias=float((p - t).mean()),
            dur_s=float(r["time"][-1] - r["time"][0]),
            Ip_max_kA=float(ip.max()) / 1e3,
            tail_frac=float(ip[max(0, int(0.9 * ip.size)):].mean() / max(ip.max(), 1e-9)),
        ))
    df = pd.DataFrame(rows)

    # ---- 1/2. TOP-12 最差炮的 PF6 target/pred 统计 ----
    worst = df.sort_values("r2").head(12)
    print("\n=== TOP-12 worst-PF6 shots: PF6 target vs pred 统计 (kA) ===")
    cols = ["shot", "n", "r2", "rmse", "bias",
            "t_min", "t_max", "t_mean", "t_std",
            "p_min", "p_max", "p_mean", "p_std"]
    print(worst[cols].to_string(index=False))

    # target 近常数判断
    print("\n=== 这些最差炮里, PF6 target std < 0.5 kA (近常数, R² 分母小) 的 ===")
    print(worst[worst["t_std"] < 0.5][["shot", "t_std", "t_min", "t_max", "p_std", "rmse", "r2"]].to_string(index=False))

    # ---- 3. 去掉 TOP-K 最差炮后 PF6 全局 R² ----
    def pf6_r2_after_drop(k):
        keep_shots = set(df.sort_values("r2", ascending=False).iloc[k:]["shot"])  # 丢 r2 最低的 k 炮
        ts = np.concatenate([r["target_kA"][r["action_mask"][:, PF6], PF6] for r in recs if r["shot"] in keep_shots])
        ps = np.concatenate([r["pred_kA"][r["action_mask"][:, PF6], PF6] for r in recs if r["shot"] in keep_shots])
        return _metrics_1d(ps, ts)[0], ts.size

    base_r2, base_n = pf6_r2_after_drop(0)
    print(f"\n=== PF6 全局 R² 去除敏感性 (n_total={base_n}) ===")
    print(f"  drop  0: R²={base_r2:.4f}  (baseline, 全 931 炮)")
    for k in (5, 10, 15, 20, 25, 30, 50):
        r2, n = pf6_r2_after_drop(k)
        print(f"  drop {k:2d}: R²={r2:.4f}  (n={n}, Δ={r2 - base_r2:+.4f})")

    # ---- 4. 相关性: 逐炮 PF6 r2 与客观属性 ----
    print("\n=== 逐炮 PF6 R² 与客观炮属性 的 Spearman/Pearson 相关 ===")
    for col in ["Ip_max_kA", "dur_s", "tail_frac", "shot"]:
        # 用 R² 但 R² 可暴负, 用 RMSE 更稳; 同时报 RMSE 相关
        for metric in ["rmse", "r2"]:
            sub = df[[col, metric]].dropna()
            pr = sub[col].corr(sub[metric], method="pearson")
            sp = sub[col].corr(sub[metric], method="spearman")
            print(f"  {col:11s} vs {metric}: pearson={pr:+.3f}  spearman={sp:+.3f}")

    # 看看最差炮的 shot 号是否聚集 (时间泄漏/某批次)
    print("\n=== TOP-30 worst-PF6 shot 号 (升序, 看是否聚集) ===")
    print(sorted(df.sort_values("r2").head(30)["shot"].tolist()))


if __name__ == "__main__":
    main()
