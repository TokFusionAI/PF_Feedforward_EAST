"""汇总 cross-β_N-mode (低β_N训→高β_N测, 干净时序+β_N筛选 leave-one-mode-out, ∥时间防泄漏) 划分消融结果 -> 6 行表。

读 results/betan_ablation/<cfg>/<cfg>_betan_s<seed>/metrics_summary.json,
对每个 cfg 在 3 个 seed 上求 mean±std (RMSE_kA / R²_med / MSE), 写
  results/betan_ablation/table_betan.md
  results/betan_ablation/table_betan.csv
并在可能时与随机划分消融 (results/ablation/<cfg>/<cfg>_s<seed>/metrics_summary.json, kA 重评) 并列对比, 给出泛化 gap。

与 aggregate_igbt_ablation.py 的区别: ROOT=results/betan_ablation, tag=_betan_s,
对应 cross-β_N-mode split (meta/split_by_order_betan: train=低β_N(<0.8)早炮 8048,
val=低β_N 1022, test=高β_N(>0.8)晚炮 931, held-out 模式; 无泄漏无交叠)。
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("results/betan_ablation")
CONFIGS = [
    "transformer_bidir_on",  # proposed (bidir + time PE)
    "transformer_off",       # transformer, causal, none PE
    "transformer_on",        # transformer, causal, time PE
    "transformer_bidir_off", # transformer, bidir, none PE
    "lstm_off",              # LSTM, none PE
    "mlp_off",               # MLP, none PE
]
SEEDS = [11, 44, 20260424]
# random-split 消融用的是另一组 8 seed; 其 ckpt 在 results/ablation/<cfg>/<cfg>_s<seed>/
RANDOM_SEEDS = [11, 22, 33, 44, 55, 66, 77, 20260424]
LABELS = {
    "transformer_bidir_on": "Transformer (proposed), time (sinusoidal)",
    "transformer_off": "Transformer, causal, none",
    "transformer_on": "Transformer, causal, time (sinusoidal)",
    "transformer_bidir_off": "Transformer (bidir), none",
    "lstm_off": "LSTM, none",
    "mlp_off": "MLP, none",
}


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def load_random_summary() -> dict:
    """随机划分消融的 kA 重评 (cfg -> {rmse} mean over RANDOM_SEEDS)。

    random ablation 原仅存归一化 global_mse (无 kA 指标)。这里改为读
    results/ablation/<cfg>/<cfg>_s<seed>/metrics_summary.json (由
    run_random_kA_eval.sh 重评生成) 的 overall_rmse_kA, 取均值。
    缺失则该 cfg 不出 random 对比。
    """
    out = {}
    for cfg in CONFIGS:
        rmses = []
        for s in RANDOM_SEEDS:
            p = Path(f"results/ablation/{cfg}/{cfg}_s{s}/metrics_summary.json")
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
                if "overall_rmse_kA" in d:
                    rmses.append(float(d["overall_rmse_kA"]))
            except (ValueError, TypeError):
                pass
        if rmses:
            out[cfg] = {"rmse": sum(rmses) / len(rmses), "r2med": None}
    return out


def main() -> int:
    random = load_random_summary()
    rows = []
    for cfg in CONFIGS:
        rmses, r2meds, mses, n = [], [], [], 0
        for s in SEEDS:
            p = ROOT / cfg / f"{cfg}_betan_s{s}" / "metrics_summary.json"
            if not p.exists():
                print(f"  missing {p}")
                continue
            d = json.loads(p.read_text())
            rmse = d["overall_rmse_kA"]
            rmses.append(rmse)
            r2meds.append(d["r2_median"])
            mses.append(rmse ** 2)
            n += 1
        rm, rs = mean_std(rmses)
        r2m, r2s = mean_std(r2meds)
        ms, msd = mean_std(mses)
        rnd = random.get(cfg, {})
        rows.append({
            "cfg": cfg, "label": LABELS[cfg], "n": n,
            "rmse_mean": rm, "rmse_std": rs,
            "r2med_mean": r2m, "r2med_std": r2s,
            "mse_mean": ms, "mse_std": msd,
            "rnd_rmse": rnd.get("rmse"), "rnd_r2med": rnd.get("r2med"),
        })

    # markdown
    md = ["# Cross-β_N-mode 划分 (低β_N训→高β_N测, 干净时序+β_N筛选 leave-one-mode-out, ∥时间防泄漏) 消融结果", "",
          "划分: `meta/split_by_order_betan` (在 igbt 干净时序上叠加 β_N 筛选: "
          "train=低β_N(<0.8)早炮 8048 / val=低β_N 1022 / test=高β_N(>0.8)晚炮 931, held-out 模式); "
          "硬时间边界 train max<val min<test min, β_N 区间不相交, 严格无泄漏无交叠。"
          "6 配置 × 3 seed (" + ", ".join(map(str, SEEDS)) + ")。",
          "norm_stats 仅由 betan train (低β_N) 重算 (19 维 notime)。模型与随机划分消融一致 (仅划分不同)。\n",
          "| Model (PE) | n | Test RMSE (kA) | Test R²_med | Test MSE | 随机RMSE | ΔRMSE |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        rmse = f"{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}" if r['rmse_mean'] is not None else "NA"
        r2 = f"{r['r2med_mean']:.4f}±{r['r2med_std']:.4f}" if r['r2med_mean'] is not None else "NA"
        mse = f"{r['mse_mean']:.5f}±{r['mse_std']:.5f}" if r['mse_mean'] is not None else "NA"
        rnd = f"{r['rnd_rmse']:.3f}" if r['rnd_rmse'] is not None else "-"
        d = f"{r['rmse_mean'] - r['rnd_rmse']:+.3f}" if (r['rmse_mean'] is not None and r['rnd_rmse'] is not None) else "-"
        md.append(f"| {r['label']} | {r['n']} | {rmse} | {r2} | {mse} | {rnd} | {d} |")
    (ROOT / "table_betan.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # csv
    with open(ROOT / "table_betan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cfg", "label", "n", "rmse_mean", "rmse_std",
                                          "r2med_mean", "r2med_std", "mse_mean", "mse_std",
                                          "rnd_rmse", "rnd_r2med"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n".join(md))
    print(f"\nwrote {ROOT/'table_betan.md'} and {ROOT/'table_betan.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
