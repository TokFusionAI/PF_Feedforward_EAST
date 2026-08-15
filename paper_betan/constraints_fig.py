"""N1 约束满足图 —— pred vs truth 的违约率对比, 回答"预测是否超出机器约束"。

amp (|I|>14.5kA): 从 per_shot_preds.npz 直接算 pred 和 truth 两者的全局违约率
  (per_shot_metrics 只算 pred amp_v, 无 truth, 故这里自算)。
dI/dt (4kA/s, P99, P99.9, 相位 max): 读 eval_test/metrics_summary.json (已有 pred/truth)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bc.common.constants import HARD_LIMIT_I_KA  # 14.5


def amp_rate(npz_path: Path) -> tuple[float, float]:
    """全局 (pred, truth) 的 |I|>14.5kA 违约率 (mask 内)。"""
    d = np.load(npz_path, allow_pickle=False)
    pred = d["pred_kA"]; tgt = d["target_kA"]; mask = d["action_mask"]
    n = max(int(mask.sum()), 1)
    p = float(((np.abs(pred) > HARD_LIMIT_I_KA) & mask).sum()) / n
    t = float(((np.abs(tgt) > HARD_LIMIT_I_KA) & mask).sum()) / n
    d.close()
    return p, t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="results/paper_betan/predictions/per_shot_preds.npz")
    ap.add_argument("--metrics", default="results/paper_betan/eval_test/metrics_summary.json")
    ap.add_argument("--out", default="results/paper_betan/figures/constraint_satisfaction.png")
    args = ap.parse_args()

    m = json.loads(Path(args.metrics).read_text())
    pred_amp, truth_amp = amp_rate(Path(args.preds))

    groups = [
        ("|I| > 14.5 kA",        pred_amp, truth_amp),
        ("|dI/dt| > 4 kA/s",     m.get("didt_violation_rate_pred_4kAs_mean"), m.get("didt_violation_rate_truth_4kAs_mean")),
        ("|dI/dt| > P99",        m.get("didt_violation_rate_pred_P99_mean"), m.get("didt_violation_rate_truth_P99_mean")),
        ("|dI/dt| > P99.9",      m.get("didt_violation_rate_pred_P99_9_mean"), m.get("didt_violation_rate_truth_P99_9_mean")),
        ("phase-aware max",      m["didt_PHASE_max"]["pred_overall"], m["didt_PHASE_max"]["truth_overall"]),
    ]
    labels = [g[0] for g in groups]
    pred = np.array([g[1] or 0 for g in groups])
    truth = np.array([g[2] or 0 for g in groups])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, truth * 100, w, label="operator (truth)", color="#2563EB")
    ax.bar(x + w / 2, pred * 100, w, label="BC prediction (proposed model)", color="#DC2626")
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, truth[i] * 100 + 0.15, f"{truth[i]*100:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, pred[i] * 100 + 0.15, f"{pred[i]*100:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("violation rate (%)")
    ax.set_title("Constraint satisfaction: prediction vs operator (test set)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} + .pdf  (pred≤truth ⇒ 预测不超约束; amp_pred={pred[0]*100:.2f}% truth={truth[0]*100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
