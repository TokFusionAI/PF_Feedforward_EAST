"""汇总 IGBT-PWM 时序划分消融结果 -> 6 行表。

读 results/igbt_ablation/<cfg>/<cfg>_igbt_s<seed>/metrics_summary.json,
对每个 cfg 在 3 个 seed 上求 mean±std (RMSE_kA / R²_med / MSE), 写
  results/igbt_ablation/table_igbt.md
  results/igbt_ablation/table_igbt.csv
并在可能时与随机划分消融 (results/ablation/summary_aggregated.csv) 并列对比, 给出泛化 gap。

与 aggregate_temporal_ablation.py 的区别: ROOT=results/igbt_ablation, tag=_igbt_s,
对应 IGBT-PWM 纯净时代 split (meta/split_by_order_igbt, shot>=117203)。
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("results/igbt_ablation")
CONFIGS = [
    "transformer_bidir_on",  # proposed (bidir + time PE)
    "transformer_off",       # transformer, causal, none PE
    "transformer_on",        # transformer, causal, time PE
    "transformer_bidir_off", # transformer, bidir, none PE
    "lstm_off",              # LSTM, none PE
    "mlp_off",               # MLP, none PE
]
SEEDS = [11, 44, 20260424]
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
    """随机划分消融汇总 (cfg -> {rmse, r2med} mean). 可选, 缺失则跳过对比。"""
    p = Path("results/ablation/summary_aggregated.csv")
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            cfg = (row.get("config") or row.get("cfg") or "").strip()
            if cfg in CONFIGS:
                try:
                    out[cfg] = {
                        "rmse": float(row.get("rmse_kA_mean", row.get("rmse", "nan"))),
                        "r2med": float(row.get("r2_med_mean", row.get("r2med", "nan"))),
                    }
                except (ValueError, TypeError):
                    pass
    return out


def main() -> int:
    random = load_random_summary()
    rows = []
    for cfg in CONFIGS:
        rmses, r2meds, mses, n = [], [], [], 0
        for s in SEEDS:
            p = ROOT / cfg / f"{cfg}_igbt_s{s}" / "metrics_summary.json"
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
    md = ["# IGBT-PWM 时序划分 (chronological) 消融结果", "",
          "划分: `meta/split_by_order_igbt` (shot>=117203, 2022-11 PS11/12 IGBT-PWM 更新后;",
          "train 17038 / val 2130 / test 2129, 严格时序无交叠), 6 配置 × 3 seed (" + ", ".join(map(str, SEEDS)) + ")。",
          "norm_stats 仅由 igbt train 重算 (19 维 notime)。模型与随机划分消融一致 (仅划分不同)。\n",
          "| Model (PE) | n | Test RMSE (kA) | Test R²_med | Test MSE | 随机RMSE | ΔRMSE |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        rmse = f"{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}" if r['rmse_mean'] is not None else "NA"
        r2 = f"{r['r2med_mean']:.4f}±{r['r2med_std']:.4f}" if r['r2med_mean'] is not None else "NA"
        mse = f"{r['mse_mean']:.5f}±{r['mse_std']:.5f}" if r['mse_mean'] is not None else "NA"
        rnd = f"{r['rnd_rmse']:.3f}" if r['rnd_rmse'] is not None else "-"
        d = f"{r['rmse_mean'] - r['rnd_rmse']:+.3f}" if (r['rmse_mean'] is not None and r['rnd_rmse'] is not None) else "-"
        md.append(f"| {r['label']} | {r['n']} | {rmse} | {r2} | {mse} | {rnd} | {d} |")
    (ROOT / "table_igbt.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # csv
    with open(ROOT / "table_igbt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cfg", "label", "n", "rmse_mean", "rmse_std",
                                          "r2med_mean", "r2med_std", "mse_mean", "mse_std",
                                          "rnd_rmse", "rnd_r2med"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n".join(md))
    print(f"\nwrote {ROOT/'table_igbt.md'} and {ROOT/'table_igbt.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
