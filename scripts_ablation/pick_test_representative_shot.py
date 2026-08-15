#!/usr/bin/env python3
"""选 igbt test 集代表炮 (transformer_bidir_on 阶段2 eval 之后): 中位数 loss 且三阶段完整。

读阶段2 eval 产物:
  results/igbt_ablation/transformer_bidir_on/transformer_bidir_on_igbt_s44/physical_violation.csv
    -> per-shot loss
  results/igbt_ablation/transformer_bidir_on/transformer_bidir_on_igbt_s44/per_shot_preds.npz
    -> phase_ids (查 ramp_up/flat_top/ramp_down 三阶段覆盖)

选法: 在三阶段都覆盖 (每相 >=5 时间步) 的炮里, loss 最接近中位数的炮 ——
既非最好也非最差, 代表模型在 test 集上的"典型"一炮 (严格泛化: 模型训练时未见过)。
打印炮号 + 写 results/paper_igbt/representative_shot.txt, 供 paper_igbt/run_all.sh 的 SHOT。

用法 (阶段2 igbt eval 完成后):
  python scripts_ablation/pick_test_representative_shot.py
  # -> SHOT=<炮号> sbatch paper_igbt/submit.sbatch
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results/igbt_ablation/transformer_bidir_on/transformer_bidir_on_igbt_s44"
CSV = EVAL_DIR / "physical_violation.csv"
NPZ = EVAL_DIR / "per_shot_preds.npz"
OUT = ROOT / "results/paper_igbt/representative_shot.txt"


def main() -> int:
    if not CSV.is_file():
        sys.exit(f"missing {CSV} (先跑阶段2 igbt eval: submit_igbt_ablation_eval.sbatch)")
    df = pd.read_csv(CSV)
    if "loss" not in df.columns or "shot" not in df.columns:
        sys.exit(f"{CSV} 缺 shot/loss 列: {list(df.columns)}")

    # 三阶段覆盖: 从 per_shot_preds.npz 读 phase_ids (0=ramp_up,1=flat_top,2=ramp_down)
    valid_shots = set(int(s) for s in df["shot"])
    if NPZ.is_file():
        d = np.load(NPZ, allow_pickle=False)
        shots, T, phase_ids = d["shots"], d["T"], d["phase_ids"]
        three_phase = set()
        for i in range(len(shots)):
            Tt = int(T[i])
            pids = phase_ids[i, :Tt].astype(int)
            ok = all(int((pids == p).sum()) >= 5 for p in (0, 1, 2))
            if ok:
                three_phase.add(int(shots[i]))
        if three_phase:
            valid_shots = three_phase
            print(f"三阶段覆盖炮: {len(three_phase)} (每相>=5步)")
        else:
            print("warning: 无三阶段覆盖炮 (phase_ids 约定?), 退化为全部 test 炮")
    df = df[df["shot"].isin(valid_shots)].sort_values("loss").reset_index(drop=True)
    if df.empty:
        sys.exit("候选池空")

    med = float(df["loss"].median())
    rep = df.iloc[(df["loss"] - med).abs().argsort().iloc[0]]
    shot = int(rep["shot"])
    print(f"代表炮: {shot}  (loss={rep['loss']:.4f} kA², 池中位loss={med:.4f}, 候选池={len(df)})")
    for q in (0.25, 0.5, 0.75):
        qs = float(df["loss"].quantile(q))
        s = int(df.iloc[(df["loss"] - qs).abs().argsort().iloc[0]]["shot"])
        print(f"  loss p{int(q * 100):>2}: shot={s}  loss={qs:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(str(shot))
    print(f"\nwrote {OUT}  ->  SHOT={shot} sbatch paper_igbt/submit.sbatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
