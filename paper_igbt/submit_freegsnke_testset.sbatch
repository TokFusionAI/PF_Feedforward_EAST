#!/bin/bash
#SBATCH --job-name=fg_ts
#SBATCH --partition=shenmadcu
#SBATCH --array=0-2                       # 3 节点并行 (每 shard 一节点)
#SBATCH --gres=dcu:1                      # 模型推理单卡
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # 串行 fork 实际~2cpu, 留余量; 383/6000 mix 可跑
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=results/paper_igbt/freegsnke_testset/slurm_%A_%a.out
#
# 测试集级 FreeGSNKE 评估 (array=3 分片 + 多进程):
#   每 shard 处理 shots[shard::3], Phase1 逐炮 DCU 推理, Phase2 mp.Pool(~58) 并行 FreeGSNKE。
#   全量(2129炮): sbatch paper_igbt/submit_freegsnke_testset.sbatch
#   小测(每shard 3炮): N_SHOTS=9 sbatch paper_igbt/submit_freegsnke_testset.sbatch
#   跑完合并+画图: python -m paper_igbt.aggregate_freegsnke_testset
set -euo pipefail
cd /home/user/PF_current_EAST
set +eu; source /home/user/anaconda3/envs/torch/../opt/dtk-25.04/env.sh 2>/dev/null || source /opt/dtk/env.sh; set -eu
_GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
export DTK_ENV_SOURCED=1 MIOPEN_CPLUS_FIXED=1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1          # 单帧 numba 多线程无收益; 靠多进程
export NUMBA_CACHE_DIR=/home/user/PF_current_EAST/.numba_cache
mkdir -p "$NUMBA_CACHE_DIR"

PY=python3
echo "[$(date)] shard ${SLURM_ARRAY_TASK_ID}/3 on $(hostname) (job ${SLURM_JOB_ID}), N_SHOTS=${N_SHOTS:-999999(全)}"
$PY -m paper_igbt.freegsnke_testset_eval \
  --n-shots ${N_SHOTS:-999999} --frames-per-phase ${FPP:-3} \
  --n-shards 3 --shard-id ${SLURM_ARRAY_TASK_ID} --n-proc ${N_PROC:-58} --nx 65
echo "[$(date)] shard ${SLURM_ARRAY_TASK_ID} DONE on $(hostname)"
