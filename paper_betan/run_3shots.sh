#!/usr/bin/env bash
# 3 个高Ip代表炮 (best/median/worst) 的 FreeGSNKE 物理验证 + hex 密度图.
# pick_best_slices 已改为 flat_top 中段选片 (避开贴近 ramp_up/ramp_down 的边缘).
# 不重跑 predict/eval/constraints/plot_figures: 3 炮共享同一 test 预测, 已在 SHOT=156715 首跑时生成.
# 用法: bash paper_betan/run_3shots.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DTK_ENV=${DTK_ENV:-/opt/dtk/env.sh}
if [[ -z "${DTK_ENV_SOURCED:-}" && -f "$DTK_ENV" ]]; then
  set +eu; source "$DTK_ENV"; set -eu; export DTK_ENV_SOURCED=1
fi
if [[ -z "${MIOPEN_CPLUS_FIXED:-}" ]] && command -v gcc >/dev/null 2>&1; then
  _GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
  export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
  export MIOPEN_CPLUS_FIXED=1
fi

PY=${PY:-python3}
WS=results/paper_betan/freegsnke_whole_shot
FIG=results/paper_betan/figures
SHOTS=(156715 159033 158259)   # best / median / worst (高Ip test)

echo "=== [hex] per_channel_scatter 密度图 (hex/kde/hist, 文章用, 一次) ==="
$PY -m paper_betan.plot_scatter_density

for SHOT in "${SHOTS[@]}"; do
  echo ">>> SHOT=$SHOT (precursor + export + pick_best[flat中段] + R8Z8/montage/flattop/wall_dist)"
  if [ ! -f "results/freegsnke_precursors/$SHOT/precursor.npz" ]; then
    $PY -m paper_betan.build_precursor_local --shot "$SHOT"
  fi
  $PY -m paper_betan.export_all_slices --shot "$SHOT" --precursor "results/freegsnke_precursors/$SHOT/precursor.npz"
  $PY -m paper_betan.pick_best_slices --shot "$SHOT" --precursor "results/freegsnke_precursors/$SHOT/precursor.npz"
  $PY paper_betan/plot_r8z8.py --shot "$SHOT" --output "$FIG/R8Z8_PF_shot${SHOT}.png"
  $PY paper_betan/plot_montage.py --shot "$SHOT" --whole-shot-root "$WS" --precursor-root results/freegsnke_precursors --out-dir "$FIG"
  $PY paper_betan/plot_flattop_pred_only.py --shot "$SHOT" --whole-shot-root "$WS" --out-dir "$FIG"
  $PY paper_betan/plot_pred_wall_dist.py --shot "$SHOT" --whole-shot-root "$WS" --precursor-root results/freegsnke_precursors --out-dir "$FIG"
done
echo ">>> 3-shots ALL DONE -> $FIG/"
