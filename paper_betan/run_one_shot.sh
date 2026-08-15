#!/usr/bin/env bash
# 单炮 FreeGSNKE 验证 (flat_top 中段选片) + R8Z8/montage/flattop/wall_dist.
# 不含 hex/predict/eval (hex 已独立生成; predict/eval 与炮无关, 已生成).
# 输出全炮号独立 (freegsnke_whole_shot/<shot>/, figures/*shot<shot>*), 可多炮并行.
# 用法: SHOT=<shot> bash paper_betan/run_one_shot.sh
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
SHOT=${SHOT:?need SHOT}
WS=results/paper_betan/freegsnke_whole_shot
FIG=results/paper_betan/figures
PRE=results/freegsnke_precursors

echo ">>> SHOT=$SHOT (precursor + export + pick_best[flat中段] + 4图)"
[ -f "$PRE/$SHOT/precursor.npz" ] || $PY -m paper_betan.build_precursor_local --shot "$SHOT"
$PY -m paper_betan.export_all_slices --shot "$SHOT" --precursor "$PRE/$SHOT/precursor.npz"
$PY -m paper_betan.pick_best_slices --shot "$SHOT" --precursor "$PRE/$SHOT/precursor.npz"
$PY paper_betan/plot_r8z8.py --shot "$SHOT" --output "$FIG/R8Z8_PF_shot${SHOT}.png"
$PY paper_betan/plot_montage.py --shot "$SHOT" --whole-shot-root "$WS" --precursor-root "$PRE" --out-dir "$FIG"
$PY paper_betan/plot_flattop_pred_only.py --shot "$SHOT" --whole-shot-root "$WS" --out-dir "$FIG"
$PY paper_betan/plot_pred_wall_dist.py --shot "$SHOT" --whole-shot-root "$WS" --precursor-root "$PRE" --out-dir "$FIG"
echo ">>> ONE-SHOT DONE $SHOT"
