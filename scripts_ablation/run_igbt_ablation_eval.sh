#!/usr/bin/env bash
# IGBT-PWM 时序消融【评估】: 对 18 个 best_val.pt, 用 paper_best_timepe 同款 predict + eval_test
# 在 igbt test 集算 metrics_summary.json (ckpt 内 cfg 自动带 igbt 划分 + igbt norm_stats),
# 再 aggregate 成 6 行表 (RMSE/R²_med, mean±std over 3 seed)。需 1 张 DCU。
#
# 用法 (计算节点, 训练全部完成后):
#   bash scripts_ablation/run_igbt_ablation_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DTK_ENV=${DTK_ENV:-/opt/dtk/env.sh}
if [[ -z "${DTK_ENV_SOURCED:-}" && -f "$DTK_ENV" ]]; then
  set +e; source "$DTK_ENV"; set -e; export DTK_ENV_SOURCED=1
fi
if [[ -z "${MIOPEN_CPLUS_FIXED:-}" ]] && command -v gcc >/dev/null 2>&1; then
  _GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
  export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
  export MIOPEN_CPLUS_FIXED=1
fi

PY=${PY:-python3}
CONFIGS=(transformer_bidir_on transformer_off transformer_on transformer_bidir_off lstm_off mlp_off)
SEEDS=(11 44 20260424)
ROOT=results/igbt_ablation

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag=${cfg}_igbt_s${seed}
    out=${ROOT}/${cfg}/${tag}
    ckpt=${out}/checkpoints/best_val.pt
    pred=${out}/per_shot_preds.npz
    if [[ ! -f "$ckpt" ]]; then echo "MISSING $ckpt (训练未完成?), 跳过"; continue; fi
    if [[ -f "$out/metrics_summary.json" ]]; then echo "skip ${tag} (已评估)"; continue; fi
    echo ">>> eval ${tag}"
    $PY -m paper_best_timepe.predict --ckpt "$ckpt" --split test --out "$pred"
    $PY -m paper_best_timepe.eval_test --preds "$pred" --out-dir "$out"
  done
done

echo "=== aggregate -> table ==="
$PY scripts_ablation/aggregate_igbt_ablation.py
echo ">>> 结果: results/igbt_ablation/table_igbt.md (及 .csv)"
