#!/bin/bash
#SBATCH --job-name=ipm_shot
#SBATCH --partition=shenmadcu
#SBATCH --gres=dcu:1                      # export 推理单卡; FreeGSNKE/画图 CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=results/paper_betan/slurm_shot_%j.out
#
# 单炮 FreeGSNKE 验证 (flat_top 中段选片). 多炮并行: 各 SHOT 输出炮号独立, 不冲突.
#   SHOT=<shot> sbatch paper_betan/submit_one_shot.sbatch

set -euo pipefail
cd /home/user/PF_current_EAST
set +eu; source /opt/dtk/env.sh; set -eu
_GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
export DTK_ENV_SOURCED=1 MIOPEN_CPLUS_FIXED=1 OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 NUMBA_CACHE=1

export SHOT=${SHOT:?need SHOT}
echo "[$(date)] betan one-shot SHOT=$SHOT on $(hostname) (job ${SLURM_JOB_ID:-manual})"
bash paper_betan/run_one_shot.sh
echo "[$(date)] one-shot SHOT=$SHOT DONE"
