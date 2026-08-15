"""探索: (1) action_mask 是否已排除破裂后时段? (2) 多少炮在数据窗内有可见破裂(Ip骤降)?
(3) 破裂点相对 masked-in 样本的位置?

对每炮:
  - |Ip| 全轨迹 (T 步), Ip_max
  - T (记录长度), n_valid (mask True 步数, 各通道一致)
  - masked-OUT 步的 |Ip| 均值 vs masked-IN 步的 |Ip| 均值 → 看 mask 是不是砍了低 Ip 尾
  - 破裂检测: flat_top 之后 |Ip| 骤降到 <CRASH_FRAC*Ip_max 的首个索引 (且此前曾 >=0.7*Ip_max)
  - 破裂后还有多少 masked-in 步 (即"待丢弃"的样本)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import load_records  # noqa
from bc.data.phases import detect_phase_slices  # noqa

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
CRASH_FRAC = 0.3   # |Ip| < 0.3*Ip_max 视为已破裂/崩溃
HIGH_FRAC = 0.7    # 此前须到过 0.7*Ip_max 才算"有等离子体 then 崩"


def detect_crash(time_s, ip_abs, ip_max, phase_ids):
    """返回破裂索引 (首个 flat_top 之后 |Ip| 跌破 CRASH_FRAC 且此前到过 HIGH_FRAC); None=无破裂。"""
    if ip_max < 100_000:
        return None
    # flat_top 段
    ft = phase_ids == 1
    if ft.sum() == 0:
        return None
    ft_end = np.where(ft)[0][-1]
    # 在 ft_end 之后找首个跌破 CRASH_FRAC 的点
    tail = np.where(ft)[0]  # flat_top 内
    # 从 flat_top 末尾往后扫描
    below = np.where(ip_abs[ft_end:] < CRASH_FRAC * ip_max)[0]
    if len(below) == 0:
        return None
    crash = ft_end + below[0]
    # 确认此前到过 HIGH_FRAC (排除一开始就没等离子体的)
    if ip_abs[:crash].max() < HIGH_FRAC * ip_max:
        return None
    return int(crash)


def main():
    recs = load_records(PRED)
    rows = []
    for r in recs:
        ip_abs = np.abs(r["Ip_A"])
        ip_max = float(ip_abs.max()) if ip_abs.size else 0.0
        mask = r["action_mask"][:, 0]  # (T,) 各通道一致
        T = r["T"]
        n_valid = int(mask.sum())
        crash = detect_crash(r["time"], ip_abs, ip_max, r["phase_ids"])
        # masked-in / masked-out 的 Ip
        ip_in = ip_abs[mask].mean() if n_valid else 0.0
        ip_out = ip_abs[~mask].mean() if (~mask).sum() else float("nan")
        # 破裂后 masked-in 步数
        post_valid = int(mask[crash:].sum()) if crash is not None else 0
        rows.append(dict(
            shot=r["shot"], T=T, n_valid=n_valid,
            Ip_max_kA=ip_max / 1e3,
            ip_in_kA=ip_in / 1e3, ip_out_kA=(ip_out if ip_out == ip_out else np.nan) / 1e3,
            crash_idx=crash if crash is not None else -1,
            crash_at_end=crash if crash is not None else -1,
            post_valid_steps=post_valid,
            has_crash=crash is not None,
        ))
    df = pd.DataFrame(rows)

    print(f"=== {len(df)} shots ===")
    print(f"有可见破裂(has_crash): {df['has_crash'].sum()} / {len(df)}")
    print(f"\nmask 是否已砍低 Ip 尾?  (ip_out = masked-OUT 步 |Ip|; 若 << ip_in 说明 mask 已排除尾段)")
    print(f"  ip_in  median = {df['ip_in_kA'].median():.1f} kA")
    print(f"  ip_out median = {df['ip_out_kA'].median():.1f} kA  (NaN={df['ip_out_kA'].isna().sum()} 炮无 masked-out)")
    sub = df[df["ip_out_kA"].notna()]
    print(f"  ip_out < 100 kA 的炮 (mask 砍的是低 Ip): {(sub['ip_out_kA'] < 100).sum()}")

    print(f"\n=== 有破裂的炮: 破裂后还剩多少 masked-in 步 (这些是用户想丢的) ===")
    cdf = df[df["has_crash"]]
    if len(cdf):
        print(f"  破裂后 masked-in 步总数: {cdf['post_valid_steps'].sum()}")
        print(f"  破裂后仍有 >0 masked-in 步的炮: {(cdf['post_valid_steps'] > 0).sum()}")
        print(cdf[["shot", "T", "n_valid", "crash_idx", "post_valid_steps", "Ip_max_kA"]]
              .sort_values("post_valid_steps", ascending=False).head(30).to_string(index=False))
    else:
        print("  (无炮在窗内破裂)")

    df.to_csv("results/paper_betan/figures/_diag_disruption.csv", index=False)
    print("\nwrote results/paper_betan/figures/_diag_disruption.csv")

    # 最差 PF6 炮是否在破裂列表里
    worst = {156947,156948,156949,156951,156952,156953,156956,156965,156966,156987,157050,157051,157098,157538,157587,157774,157775,157776,158367,158471,158877,158878,159112,159458,159459,159540,159565,159694,159698,159701}
    print(f"\n最差30 PF6 炮里 has_crash 的: {df[df.shot.isin(worst)]['has_crash'].sum()} / 30")


if __name__ == "__main__":
    main()
