"""读 per_shot_preds.npz → 复用 bc.evaluation.eval 的 per_shot_metrics/aggregate_r2 → 指标 + 违约率。

产出 results/paper_igbt/eval_test/{metrics_summary.json, physical_violation.csv}。
模型无关 (只读 npz + 调 bc 的指标函数)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bc.evaluation.eval import _metrics_1d, aggregate_r2, per_shot_metrics  # noqa: E402
from bc.data.phases import PHASE_NAMES  # noqa: E402
from bc.common.constants import DT_BUCKETS  # noqa: E402


def load_records(npz_path: Path) -> list[dict]:
    d = np.load(npz_path, allow_pickle=False)
    shots = d["shots"]
    T = d["T"]
    N = len(shots)
    recs = []
    for i in range(N):
        Tt = int(T[i])
        recs.append(dict(
            shot=int(shots[i]), T=Tt,
            time=d["time"][i, :Tt].astype(np.float64),
            dt=d["dt"][i, :Tt].astype(np.float32),
            pred_kA=d["pred_kA"][i, :Tt].astype(np.float32),
            target_kA=d["target_kA"][i, :Tt].astype(np.float32),
            action_mask=d["action_mask"][i, :Tt],
            Ip_A=d["Ip_A"][i, :Tt].astype(np.float32),
            phase_ids=d["phase_ids"][i, :Tt].astype(np.int8),
            step_phase_ids=(d["step_phase_ids"][i, : Tt - 1].astype(np.int8) if Tt > 1
                            else np.empty(0, np.int8)),
        ))
    return recs


def _mean_field(per_shot, key):
    vals = [r.get(key) for r in per_shot if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="results/paper_igbt/predictions/per_shot_preds.npz")
    ap.add_argument("--out-dir", default="results/paper_igbt/eval_test")
    args = ap.parse_args()

    recs = load_records(Path(args.preds))
    per_shot = [per_shot_metrics(r) for r in recs if r["T"] >= 2]
    r2 = aggregate_r2(recs)

    preds_all = np.concatenate([r["pred_kA"][r["action_mask"]] for r in recs])
    truths_all = np.concatenate([r["target_kA"][r["action_mask"]] for r in recs])
    overall_r2, overall_mae, overall_rmse = _metrics_1d(preds_all, truths_all)

    summary = {
        "n_test_shots": len(per_shot),
        "overall_r2": overall_r2,
        "overall_mae_kA": overall_mae,
        "overall_rmse_kA": overall_rmse,
        "r2_per_channel": r2.tolist(),
        "r2_median": float(np.median(r2)),
        "loss_kA2_median": float(np.median([r["loss"] for r in per_shot])),
        "loss_kA2_mean": float(np.mean([r["loss"] for r in per_shot])),
        "amp_violation_rate_mean": float(np.mean([r["pred_amp_violation_rate"] for r in per_shot])),
        "didt_violation_rate_pred_4kAs_mean": _mean_field(per_shot, "pred_didt_violation_rate"),
        "didt_violation_rate_truth_4kAs_mean": _mean_field(per_shot, "truth_didt_violation_rate"),
        "didt_violation_rate_pred_P99_mean": _mean_field(per_shot, "pred_didt_violation_rate_P99"),
        "didt_violation_rate_truth_P99_mean": _mean_field(per_shot, "truth_didt_violation_rate_P99"),
        "didt_violation_rate_pred_P99_9_mean": _mean_field(per_shot, "pred_didt_violation_rate_P99_9"),
        "didt_violation_rate_truth_P99_9_mean": _mean_field(per_shot, "truth_didt_violation_rate_P99_9"),
    }
    for label in ("max", "P99_9", "P99"):
        summary[f"didt_PHASE_{label}"] = {
            "pred_overall": _mean_field(per_shot, f"pred_didt_violation_rate_PHASE_{label}"),
            "truth_overall": _mean_field(per_shot, f"truth_didt_violation_rate_PHASE_{label}"),
            "pred_per_phase": {n: _mean_field(per_shot, f"pred_didt_violation_rate_PHASE_{label}_{n}") for n in PHASE_NAMES},
            "truth_per_phase": {n: _mean_field(per_shot, f"truth_didt_violation_rate_PHASE_{label}_{n}") for n in PHASE_NAMES},
        }
    dt_buckets = []
    for lo, hi in DT_BUCKETS:
        sub = [r for r in per_shot if lo <= r["dt_median"] < hi]
        if sub:
            dt_buckets.append({"bucket": f"[{lo:.2f},{hi:.2f})", "n_shots": len(sub),
                               "loss_median": float(np.median([r["loss"] for r in sub]))})
    summary["dt_buckets"] = dt_buckets

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    vio_cols = ["shot", "T", "dt_median", "loss", "pred_amp_violation_rate",
                "pred_didt_violation_rate", "truth_didt_violation_rate",
                "pred_didt_violation_rate_P99", "truth_didt_violation_rate_P99",
                "pred_didt_violation_rate_P99_9", "truth_didt_violation_rate_P99_9",
                "pred_didt_violation_rate_PHASE_max", "truth_didt_violation_rate_PHASE_max",
                "pred_didt_violation_rate_PHASE_P99_9", "truth_didt_violation_rate_PHASE_P99_9"]
    df = pd.DataFrame(per_shot)
    df[[c for c in vio_cols if c in df.columns]].to_csv(out / "physical_violation.csv", index=False)

    print(f"overall R²={overall_r2:.4f}  MAE={overall_mae:.3f} kA  RMSE={overall_rmse:.3f} kA  "
          f"r2_med={np.median(r2):.4f}")
    print(f"amp_v={summary['amp_violation_rate_mean']:.4f}  "
          f"didt_v(pred 4kA/s)={summary['didt_violation_rate_pred_4kAs_mean']:.4f} vs "
          f"truth={summary['didt_violation_rate_truth_4kAs_mean']:.4f}")
    print(f"wrote {out/'metrics_summary.json'}, {out/'physical_violation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
