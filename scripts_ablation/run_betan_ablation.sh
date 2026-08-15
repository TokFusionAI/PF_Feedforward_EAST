#!/usr/bin/env bash
# 训练 cross-betan (chrono held-out betan) 消融的【单个】(CFG, SEED) 对。
# 由 submit_betan_ablation.sbatch --array=0-17 调用, 每个 array task 训一对,
# SLURM 按空闲节点持续 backfill -> 最快墙钟。模型 = 原随机划分消融配置 (仅 --override
# 改 划分/norm/out_root/seed 到 cross-betan 划分)。
#   cross-betan split: meta/split_by_betan (按时间顺序, train 17804 / val 1978 /
#   test 412, test 为时间尾部 held-out betan 156434–159841, 严格无交叠)。
#   norm 用 19 维 notime (ablation d_state=19, 不可用 21 维版), 在 betan train 上重算。
# 输出 results/betan_ablation/<CFG>/。断点续跑: best_val.pt 存在则跳过。
#
# 环境变量: CFG (配置名, 如 transformer_bidir_on), SEED (如 44)。
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${MIOPEN_CPLUS_FIXED:-}" ]] && command -v gcc >/dev/null 2>&1; then
  _GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
  export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
  export MIOPEN_CPLUS_FIXED=1
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

TORCHRUN=${TORCHRUN:-torchrun}
NPROC=${NPROC:-8}

: "${CFG:?need CFG}"; : "${SEED:?need SEED}"
tag=${CFG}_betan_s${SEED}
out=results/betan_ablation/${CFG}
ckpt=${out}/${tag}/checkpoints/best_val.pt
[[ -f "$ckpt" ]] && { echo "[$(date)] skip ${tag} (done)"; exit 0; }

echo "[$(date)] train ${tag}  ->  ${out}/${tag}"
$TORCHRUN --standalone --nnodes=1 --nproc_per_node=${NPROC} \
  -m ablation.training.train \
  --config configs_ablation/ablation_${CFG}.yaml --tag "$tag" \
  --override seed=${SEED} \
  --override data.train_shots=meta/split_by_order_betan/train_shots.txt \
  --override data.val_shots=meta/split_by_order_betan/val_shots.txt \
  --override data.test_shots=meta/split_by_order_betan/test_shots.txt \
  --override data.norm_stats=meta/split_by_order_betan/norm_stats_notime.npz \
  --override log.out_root=${out} \
  --override log.tb_dir=${out}/tb
echo "[$(date)] DONE ${tag}"
