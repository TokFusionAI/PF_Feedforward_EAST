#!/bin/bash
#SBATCH --job-name=igbt_abl_eval
#SBATCH --partition=shenmadcu
#SBATCH --gres=dcu:1                      # predict 单卡推理; eval_test 纯 CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=results/igbt_ablation/slurm_eval_%j.out
#
# IGBT-PWM 时序消融评估: 18 个 ckpt 在 igbt test 上 predict + eval_test + 汇总成表。
# 【训练全部跑完后】再提交本作业。
#   sbatch scripts_ablation/submit_igbt_ablation_eval.sbatch

set -euo pipefail
cd /home/user/PF_current_EAST
set +eu; source /home/user/anaconda3/envs/torch/../opt/dtk-25.04/env.sh 2>/dev/null || source /opt/dtk/env.sh; set -eu
_GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
export DTK_ENV_SOURCED=1 MIOPEN_CPLUS_FIXED=1 OMP_NUM_THREADS=16

echo "[$(date)] igbt ablation eval on $(hostname) (job ${SLURM_JOB_ID:-manual})"
bash scripts_ablation/run_igbt_ablation_eval.sh
echo "[$(date)] eval DONE"
