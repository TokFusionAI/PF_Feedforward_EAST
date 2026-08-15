"""真·破裂悬崖检测: 区分"单步骤降(破裂)" vs "渐进 ramp-down"。

dt~226ms: 破裂=ms 级 → |Ip[k]| / |Ip[k-1]| 极小 (单步从满 Ip 到 ~0)。
          正常 ramp-down → 每步 ratio ~0.9 (缓慢下降)。

对每炮, 在 flat_top 起点之后找最大单步降幅; 若存在 ratio<RATIO_THR 且前值>HIGH_FRAC*max → 判破裂。
报告真破裂炮数 + 各炮悬崖位置/前后 Ip。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import load_records  # noqa

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
RATIO_THR = 0.5     # 单步 |Ip[k]|/|Ip[k-1]| < 0.5 视为骤降
HIGH_FRAC = 0.4     # 骤降前 |Ip[k-1]| 须 > 0.4*max (排除从近零开始的抖动)


def find_cliff(ip_abs, ip_max, phase_ids):
    """flat_top 起之后首个满足 ratio<RATIO_THR 且 prev>HIGH_FRAC*max 的索引 k (悬崖点); None=无。"""
    if ip_max < 100_000:
        return None, None
    ft = phase_ids == 1
    start = int(np.where(ft)[0][0]) if ft.sum() else 0
    if start + 1 >= ip_abs.size:
        return None, None
    seg = ip_abs[start:]
    ratio = seg[1:] / np.maximum(seg[:-1], 1.0)
    cand = np.where((ratio < RATIO_THR) & (seg[:-1] > HIGH_FRAC * ip_max))[0]
    if len(cand) == 0:
        return None, None
    k = start + 1 + int(cand[0])
    return k, float(ratio[cand[0]])


def main():
    recs = load_records(PRED)
    rows = []
    for r in recs:
        ip = np.abs(r["Ip_A"]); ip_max = float(ip.max())
        k, ratio = find_cliff(ip, ip_max, r["phase_ids"])
        rows.append((r["shot"], k, ratio, ip_max / 1e3))
    cliffs = [(s, k, q, m) for s, k, q, m in rows if k is not None]
    print(f"=== 真·单步骤降(破裂悬崖) 的炮: {len(cliffs)} / {len(recs)} ===")
    for s, k, q, m in sorted(cliffs):
        r = next(x for x in recs if x["shot"] == s)
        ip = np.abs(r["Ip_A"])
        before = ip[k - 1] / 1e3 if k >= 1 else float("nan")
        after = ip[k] / 1e3
        post_keep = int(r["action_mask"][k:].sum())  # 悬崖点及之后 masked-in 步
        print(f"  shot {s}: cliff@{k}/{r['T']}  |Ip| {before:.0f}->{after:.0f} kA (ratio={q:.2f})  "
              f"post-steps(incl cliff)={post_keep}")

    # 也报: 最差30 PF6 炮里有几个真悬崖
    worst = {156947,156948,156949,156951,156952,156953,156956,156965,156966,156987,157050,157051,157098,157538,157587,157774,157775,157776,158367,158471,158877,158878,159112,159458,159459,159540,159565,159694,159698,159701}
    cw = [s for s, _, _, _ in cliffs if s in worst]
    print(f"\n最差30 PF6 炮里真悬崖: {len(cw)} / 30  {cw}")

    total_post = sum(next(x for x in recs if x['shot']==s)['action_mask'][k:].sum() for s,k,_,_ in cliffs)
    print(f"\n所有真悬崖炮, 悬崖点及之后的 masked-in 步总数 = {total_post}  (这是严格'破裂后'可丢样本的上界)")


if __name__ == "__main__":
    main()
