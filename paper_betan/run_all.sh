#!/usr/bin/env bash
# paper_betan 全流程 (DCU 计算节点): s44 推理 → test 评测 → N1 → FreeGSNKE($SHOT=igbt test 代表炮) → 全部图。
# 用法: bash paper_betan/run_all.sh   (或单步: python -m paper_betan.predict ...)
set -euo pipefail
cd "$(dirname "$0")/.."


PY=${PY:-python3}
SHOT=${SHOT:?需传 betan test 代表炮号 (meta/split_by_betan/test_shots_chrono_heldout_betan.txt, 由 pick_test_representative_shot.py 选)}
WS=results/paper_betan/freegsnke_whole_shot
FIG=results/paper_betan/figures

echo "=== [0/6] test 推理一次 (s44) ==="
$PY -m paper_betan.predict --split test
echo "=== [1/6] test 评测 + 违约率 ==="
$PY -m paper_betan.eval_test
echo "=== [2/6] N1 约束满足图 ==="
$PY -m paper_betan.constraints_fig
echo "=== [3a] 导出 $SHOT 全部时间片 (bc_pred 时序模型预测 + pcs 实际 PCS) ==="
$PY -m paper_betan.export_all_slices --shot "$SHOT" --precursor results/freegsnke_precursors/$SHOT/precursor.npz
echo "=== [3b] 全片 FreeGSNKE 求解 + 按爬升/平顶/下降选误差最小片 (bc_pred 主选, pcs 参考误差) ==="
$PY -m paper_betan.pick_best_slices --shot "$SHOT" --precursor results/freegsnke_precursors/$SHOT/precursor.npz
echo "=== [4/6] F1/F2/F3/F9/N2 ==="
$PY -m paper_betan.plot_figures
echo "=== [4b] per_channel_scatter 密度图 (hex/kde/hist, 文章用) ==="
$PY -m paper_betan.plot_scatter_density
echo "=== [5/6] F4 R8Z8 (paper_betan 自有脚本; --output 是 png 文件路径) ==="
$PY paper_betan/plot_r8z8.py --shot "$SHOT" --output "$FIG/R8Z8_PF_shot${SHOT}.png"
echo "=== [6/6] F5/F6/F8 montage/flattop/wall_dist (paper_betan 自有脚本, 读 s44 切片) ==="
$PY paper_betan/plot_montage.py --shot "$SHOT" --whole-shot-root "$WS" --precursor-root results/freegsnke_precursors --out-dir "$FIG"
$PY paper_betan/plot_flattop_pred_only.py --shot "$SHOT" --whole-shot-root "$WS" --out-dir "$FIG"
$PY paper_betan/plot_pred_wall_dist.py --shot "$SHOT" --whole-shot-root "$WS" --precursor-root results/freegsnke_precursors --out-dir "$FIG"
echo ">>> paper_betan ALL DONE -> $FIG/"
