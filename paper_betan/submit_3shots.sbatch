#!/bin/bash
#SBATCH --job-name=ipm_3shots
#SBATCH --partition=shenmadcu
#SBATCH --gres=dcu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=results/paper_betan/slurm_3shots_%j.out
#
# 3 高Ip代表炮 (best 156715 / median 159033 / worst 158259) FreeGSNKE 验证
# (flat_top 中段选片) + per_channel_scatter_hex 密度图.
#   sbatch paper_betan/submit_3shots.sbatch

set -euo pipefail
cd /home/user/PF_current_EAST
set +eu; source /opt/dtk/env.sh; set -eu
_GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
export DTK_ENV_SOURCED=1 MIOPEN_CPLUS_FIXED=1 OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 NUMBA_CACHE=1

echo "[$(date)] betan 3-shots FreeGSNKE on $(hostname) (job ${SLURM_JOB_ID:-manual})"
bash paper_betan/run_3shots.sh
echo "[$(date)] betan 3-shots DONE"
