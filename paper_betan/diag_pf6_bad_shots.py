"""诊断 paper_betan (transformer_bidir_on betan s44) test 集里拉低 PF6 R² 的炮。

读一次 per_shot_preds.npz, 对每炮算:
  · 每通道 RMSE / R² / 有效样本数 (action_mask 该通道 True 的步数)
  · 整炮有效样本数、时长 (time 末-首)、dt
  · Ip 轨迹: max / 末段值 / 是否骤降 (末 10% 均值 vs max 的比值) → 破裂迹象
  · phase_ids 构成 (各相位占比)
  · pred/target 的极端值 (PF6 最大预测/真值, 是否离群)

输出 CSV (按 PF6 RMSE 降序) + 终端摘要。
不画图, 纯诊断。
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
OUT_CSV = Path("results/paper_betan/figures/_diag_pf6_bad_shots.csv")


def per_shot_channel_metrics(recs):
    rows = []
    for r in recs:
        shot = r["shot"]
        T = r["T"]
        mask = r["action_mask"]              # (T,12) bool
        pred = r["pred_kA"]; tgt = r["target_kA"]
        Ip = r["Ip_A"]; t = r["time"]; ph = r["phase_ids"]
        rec = dict(shot=shot)
        rec["T"] = T
        rec["dur_s"] = float(t[-1] - t[0]) if T > 1 else 0.0
        rec["dt_ms"] = float(np.median(r["dt"])) * 1e3 if T > 0 else 0.0
        rec["n_valid_total"] = int(mask.sum())
        # Ip 轨迹 (破裂迹象)
        ip = np.abs(Ip) if Ip.size else np.zeros(0)
        if ip.size:
            rec["Ip_max_kA"] = float(ip.max()) / 1e3
            tail = ip[max(0, int(0.9 * ip.size)):]
            rec["Ip_tail_frac"] = float(tail.mean() / max(ip.max(), 1e-9))
        else:
            rec["Ip_max_kA"] = 0.0; rec["Ip_tail_frac"] = 1.0
        # phase 构成
        for pid in sorted(np.unique(ph).tolist()):
            rec[f"ph{pid}_frac"] = float((ph == pid).mean())
        # 每通道
        pf6_rmse = 0.0
        for ch in range(12):
            m = mask[:, ch]
            n = int(m.sum())
            if n == 0:
                rec[f"{PF_NAMES[ch]}_n"] = 0
                rec[f"{PF_NAMES[ch]}_rmse"] = np.nan
                rec[f"{PF_NAMES[ch]}_r2"] = np.nan
                continue
            p = pred[m, ch]; gt = tgt[m, ch]
            rec[f"{PF_NAMES[ch]}_n"] = n
            rec[f"{PF_NAMES[ch]}_rmse"] = float(np.sqrt(np.mean((p - gt) ** 2)))
            rec[f"{PF_NAMES[ch]}_r2"] = float(_metrics_1d(p, gt)[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    print(f"loading {PRED} ...", flush=True)
    recs = load_records(PRED)
    print(f"loaded {len(recs)} shots", flush=True)

    # 全局 per-channel R²/RMSE (口径同 plot_scatter_density, 用于核对图)
    print("\n=== global per-channel R²/RMSE (should match figure) ===")
    for ch in range(12):
        ts = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in recs])
        ps = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in recs])
        r2 = _metrics_1d(ps, ts)[0]
        rmse = float(np.sqrt(np.mean((ps - ts) ** 2)))
        print(f"  {PF_NAMES[ch]:5s} R²={r2:.4f}  RMSE={rmse:.3f} kA  n={ts.size}")

    df = per_shot_channel_metrics(recs)
    df = df.sort_values("PF6_rmse", ascending=False).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} shots)")

    print("\n=== TOP 25 worst shots by PF6 RMSE ===")
    cols = ["shot", "dur_s", "Ip_max_kA", "Ip_tail_frac", "n_valid_total",
            "PF6_n", "PF6_rmse", "PF6_r2", "PF6_r2.0_frac" if "PF6_r2.0_frac" in df else "ph0_frac"]
    cols = [c for c in cols if c in df.columns]
    # 加上其它通道 r2 看是不是只有 PF6 差
    extra = [c for c in df.columns if c.endswith("_r2")]
    show = df.head(25)[cols]
    print(show.to_string(index=False))

    # PF6 r2 与其它通道对比: 这些坏炮是不是 PF6 独差, 还是整炮都差?
    r2cols = [f"{n}_r2" for n in PF_NAMES]
    print("\n=== TOP10 worst-PF6 shots: 全 12 通道 R² (看 PF6 是否独差) ===")
    print(df.head(10)[["shot"] + r2cols].to_string(index=False))

    # 破裂迹象: Ip_tail_frac < 0.3 视为末段 Ip 崩塌
    print("\n=== Ip 末段崩塌 (tail_frac<0.3, 疑似破裂) 的炮 ===")
    crash = df[df["Ip_tail_frac"] < 0.3].sort_values("PF6_rmse", ascending=False)
    print(f"count={len(crash)}")
    if len(crash):
        print(crash[["shot", "dur_s", "Ip_max_kA", "Ip_tail_frac", "PF6_rmse", "PF6_r2"]].head(30).to_string(index=False))

    print("\n=== PF6 有效样本=0 的炮 (该炮 PF6 全程 mask 关) ===")
    zero = df[df["PF6_n"] == 0]
    print(f"count={len(zero)}")
    if len(zero):
        print(zero[["shot", "dur_s", "n_valid_total", "PF6_n"]].to_string(index=False))


if __name__ == "__main__":
    main()
