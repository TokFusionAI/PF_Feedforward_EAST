#!/usr/bin/env bash
# 在计算节点上启动 BC Transformer 训练 / smoke / eval 的入口脚本.
#
# 必备:
#   - 已切到带海光 DCU runtime 的节点 (例如 compute-node382 或新的 8 卡节点).
#   - 已激活带 torch 的 conda env: source activate /home/user/anaconda3/envs/torch
#   - 工作目录为 PF_current_EAST/.
#
# 用法:
#   bash scripts/run_bc.sh smoke1            # 单卡 smoke (1-2 min)
#   bash scripts/run_bc.sh smoke8            # 8 卡 smoke 1 epoch (~5 min)
#   bash scripts/run_bc.sh full              # 8 卡 60 epoch 正式训练
#   bash scripts/run_bc.sh bench             # DataLoader 参数基准
#   bash scripts/run_bc.sh eval RUN_TAG      # 评估某个 run, 默认 RUN_TAG=run1 (不过滤)
#   bash scripts/run_bc.sh eval_filtered RUN_TAG [WHICH]  # 评估 + phase-aware |dI/dt| 硬过滤
#                                                         # WHICH = max | P99.99 | P99.9 (默认) | P99
#   bash scripts/run_bc.sh smoke_test        # 单元自测 (dataset + model + causal 等价)
#   bash scripts/run_bc.sh freegs [SHOT]     # FreeGS L2 验证（archive/legacy_freegs/run_freegs_eval.py；需 pip install freegs）
#                                            # SHOT 省略=从 test_shots 随机挑
#                                            # 需在 torch 环境中已 pip install freegs（与 PyTorch 同环境一条命令跑通）
#   bash scripts/run_bc.sh freegsnke [SHOT] [PHASE]   # freegsnke 静态前向 GS（默认 shot=158413 phase=flat_top）
#                                            # 需 torch 环境已 pip install "freegsnke[freegs4e]"；建议在计算节点运行

set -e
cd "$(dirname "$0")/.."

CONDA_PYTHON=${CONDA_PYTHON:-python3}
TORCHRUN=${TORCHRUN:-torchrun}

mode=${1:-smoke1}

case "$mode" in
  smoke_test)
    echo ">>> single-card smoke test (dataset + model + causal equivalence)"
    "$CONDA_PYTHON" -m bc._smoke_test
    ;;
  smoke1)
    echo ">>> single-card smoke run (configs/bc_v1_smoke.yaml)"
    "$CONDA_PYTHON" -m bc.train --config configs/bc_v1_smoke.yaml --tag smoke1
    ;;
  smoke8)
    echo ">>> 8-DCU DDP smoke run (1 epoch on configs/bc_v1.yaml)"
    "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=8 \
        -m bc.train --config configs/bc_v1.yaml --tag smoke8 \
        -o optim.epochs=1 -o data.train_subset_n=2000 -o data.val_subset_n=200
    ;;
  full)
    tag=${2:-run1}
    echo ">>> 8-DCU DDP full training (60 epoch on configs/bc_v1.yaml, tag=$tag)"
    "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=8 \
        -m bc.train --config configs/bc_v1.yaml --tag "$tag"
    ;;
  bench)
    echo ">>> DataLoader benchmark (single card)"
    "$CONDA_PYTHON" -m bc.benchmark_loader \
        --shots-file meta/train_shots.txt \
        --norm-stats meta/norm_stats.npz \
        --out results/bc_v1/loader_benchmark.csv
    ;;
  eval)
    tag=${2:-run1}
    ckpt=results/bc_v1/$tag/checkpoints/best_val.pt
    out=results/bc_v1/$tag/figures
    tb=results/bc_v1/tb/$tag
    echo ">>> eval test set with $ckpt (no filter)"
    "$CONDA_PYTHON" -m bc.eval \
        --config configs/bc_v1.yaml \
        --ckpt "$ckpt" \
        --split test \
        --out-dir "$out" \
        --tb-dir "$tb"
    ;;
  eval_filtered)
    tag=${2:-run1}
    which=${3:-P99.9}
    slug=$(echo "$which" | sed 's/\./_/g')
    ckpt=results/bc_v1/$tag/checkpoints/best_val.pt
    out=results/bc_v1/$tag/figures_filtered_$slug
    tb=results/bc_v1/tb/$tag
    echo ">>> eval test set with $ckpt, hard filter which=$which -> $out"
    "$CONDA_PYTHON" -m bc.eval \
        --config configs/bc_v1.yaml \
        --ckpt "$ckpt" \
        --split test \
        --out-dir "$out" \
        --tb-dir "$tb" \
        --apply-filter "$which"
    ;;
  freegs)
    shot_arg=${2:-}
    extra=""
    if [[ -n "$shot_arg" ]]; then extra="--shot $shot_arg"; fi
    nw="${FREEGS_WORKERS:-6}"
    echo ">>> FreeGS L2 eval (shot=${shot_arg:-random}, workers=$nw, override with FREEGS_WORKERS)"
    "$CONDA_PYTHON" archive/legacy_freegs/run_freegs_eval.py \
        --ckpt results/bc_v1/run1/checkpoints/best_val.pt \
        --out-dir results/freegs_eval \
        --profile-method pprime \
        --fallback-method betap \
        --workers "$nw" \
        $extra
    ;;
  freegsnke)
    shot_arg=${2:-158413}
    phase_arg=${3:-flat_top}
    echo ">>> freegsnke EAST forward GS (shot=$shot_arg phase=$phase_arg) -> results/freegsnke_eval/"
    # 无 PF_ATIME h5 时自动走 MDS（见 notebook/05 §5）；可选 MDS_SERVER=ip 覆盖默认 mds.ipp.ac.cn
    extra_mds=()
    if [[ -n "${MDS_SERVER:-}" ]]; then extra_mds+=(--mds-server "$MDS_SERVER"); fi
    if [[ -n "${DATASET_ROOT:-}" ]]; then extra_mds+=(--dataset-root "$DATASET_ROOT"); fi
    "$CONDA_PYTHON" -m bc.run_freegsnke_eval \
        --shot "$shot_arg" \
        --phase "$phase_arg" \
        --ckpt results/bc_v1/run1/checkpoints/best_val.pt \
        --out-dir results/freegsnke_eval \
        --efit-source auto \
        "${extra_mds[@]}"
    ;;
  *)
    echo "unknown mode: $mode"
    echo "usage: bash scripts/run_bc.sh {smoke_test|smoke1|smoke8|full [tag]|bench|eval [tag]|eval_filtered [tag] [which]|freegs [shot]|freegsnke [shot] [phase]}"
    exit 2
    ;;
esac
