#!/usr/bin/env python
"""betan split overview: 纯散点图(无边缘分布), 期刊格式.
x = discharge date, y = β_N, color = split (Training/Validation/Test set).
无标题、无括号描述、无多余文字, 只保留图例."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
BETAN = ROOT / "meta/split_by_order_betan"
LB = ROOT / "meta/split_by_order_betan/shot_dates.csv"
BN_TBL = ROOT / "meta/split_by_order_betan/betan_per_shot.parquet"
OUT = ROOT / "results/betan_ablation/split_overview_betan"

lb = pd.read_csv(LB)
lb["_shot_number"] = lb["_shot_number"].astype(int)
lb = lb.set_index("_shot_number")
bn_tbl = pd.read_parquet(BN_TBL).set_index("shot")["betan_mean"]


def load(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().split() if x.strip().isdigit()]


def main() -> int:
    sel = {}
    for sp, fn in [("train", "train_shots.txt"), ("val", "val_shots.txt"), ("test", "test_shots.txt")]:
        sel[sp] = set(load(BETAN / fn))

    rows = []
    for sp in ["train", "val", "test"]:
        for s in sel[sp]:
            if s in lb.index and s in bn_tbl.index:
                rows.append((s, pd.to_datetime(lb.loc[s, "Discharge_time"]),
                             float(bn_tbl.loc[s]), sp))
    df = pd.DataFrame(rows, columns=["shot", "date", "betan", "split"])

    C = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    LAB = {"train": "Training set", "val": "Validation set", "test": "Test set"}
    SZ = {"train": 12, "val": 24, "test": 30}
    AL = {"train": 0.28, "val": 0.72, "test": 0.85}
    ZO = {"train": 2, "val": 3, "test": 4}

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#444", "axes.labelcolor": "#222",
                         "xtick.color": "#444", "ytick.color": "#444"})
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for k in ["train", "val", "test"]:
        d = df[df.split == k]
        ax.scatter(d.date, d.betan, s=SZ[k], alpha=AL[k], c=C[k],
                   edgecolors="none", zorder=ZO[k], label=LAB[k])

    ax.axhline(0.8, color="#aaa", ls="--", lw=0.8, zorder=1)

    ax.set_xlabel("Discharge date (year)", fontsize=13)
    ax.set_ylabel(r"Normalized beta ($\beta_N$)", fontsize=13)
    ax.grid(True, color="#eee", lw=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, 1.52)

    # 图例放在图内左上角(高β+早期日期区域为空，不遮挡数据)
    ax.legend(loc="upper left", fontsize=10, frameon=True, fancybox=True,
              framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT.parent / f"split_overview_betan.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT.parent}/split_overview_betan.{{png,pdf}}")


if __name__ == "__main__":
    raise SystemExit(main())
