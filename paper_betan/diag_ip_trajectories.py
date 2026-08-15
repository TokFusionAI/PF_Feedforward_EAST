"""核实: 看几炮的 |Ip| 轨迹 + dt, 确认 (a) 20 破裂炮的破裂确实在末拍且仅1步;
(b) 最差 PF6 炮整窗 Ip 平直无破裂; (c) 数据窗基本=flat-top 段。
并统计全 931 炮的窗内时间范围 / dt / 末拍 Ip 分位。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import load_records  # noqa

PRED = Path("results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/per_shot_preds.npz")
CRASH_SHOTS = [157026, 158001, 159820]
WORST_SHOTS = [159694, 159698, 159701, 158367]


def show(recs, shot):
    r = next(x for x in recs if x["shot"] == shot)
    ip = np.abs(r["Ip_A"])
    t = r["time"]
    T = r["T"]
    dt = np.diff(t)
    print(f"\n--- shot {shot}: T={T}  t=[{t[0]:.2f},{t[-1]:.2f}]s  dur={t[-1]-t[0]:.2f}s  dt_med={np.median(dt)*1e3:.0f}ms ---")
    print(f"  |Ip| max={ip.max()/1e3:.0f}kA  first={ip[0]/1e3:.0f}  last5={np.round(ip[-5:]/1e3,1)}")
    ft = r["phase_ids"] == 1
    ru = r["phase_ids"] == 0
    rd = r["phase_ids"] == 2
    print(f"  phase counts: ramp_up={ru.sum()} flat_top={ft.sum()} ramp_down={rd.sum()}")
    print(f"  t at flat_top start={t[np.where(ft)[0][0]]:.2f}s  end={t[np.where(ft)[0][-1]]:.2f}s" if ft.sum() else "  no flat_top")


def main():
    recs = load_records(PRED)
    for s in CRASH_SHOTS + WORST_SHOTS:
        show(recs, s)

    # 全集统计: 窗内末拍 Ip / dt / 是否含 ramp_down
    last_ip = []; dts = []; has_rd = []; win_start = []; win_end = []
    for r in recs:
        ip = np.abs(r["Ip_A"]); t = r["time"]
        last_ip.append(ip[-1]/ip.max() if ip.max()>0 else 0)
        dts.append(np.median(np.diff(t))*1e3)
        has_rd.append((r["phase_ids"]==2).sum()>0)
        win_start.append(t[0]); win_end.append(t[-1])
    last_ip=np.array(last_ip); dts=np.array(dts); has_rd=np.array(has_rd)
    print(f"\n=== 全 {len(recs)} 炮窗内统计 ===")
    print(f"  dt_median: med={np.median(dts):.0f}ms  p5={np.percentile(dts,5):.0f}  p95={np.percentile(dts,95):.0f}")
    print(f"  末拍 |Ip|/Ip_max 分位: p5={np.percentile(last_ip,5):.2f} p50={np.median(last_ip):.2f} p95={np.percentile(last_ip,95):.2f}")
    print(f"  含 ramp_down 相位的炮: {has_rd.sum()} / {len(recs)}  (窗基本止于 flat_top)")
    print(f"  末拍 Ip 已跌破 0.3*max 的炮 (窗内拍到破裂): {(last_ip<0.3).sum()}")


if __name__ == "__main__":
    main()
