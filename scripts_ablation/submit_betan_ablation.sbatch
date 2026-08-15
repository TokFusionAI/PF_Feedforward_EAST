#!/bin/bash
#SBATCH --job-name=betan_abl
#SBATCH --partition=shenmadcu            # 381-383 节点池
#SBATCH --array=0-17                     # 18 个独立任务 (6 配置 × 3 seed), 每个训一对
#SBATCH --nodes=1
#SBATCH --gres=dcu:8                     # 每任务 8 张 DCU = 1 节点
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=results/betan_ablation/slurm_%A_%a.out
#
# cross-betan (chrono held-out betan) 消融 (6 配置 × 3 seed = 18 run), 在
# meta/split_by_betan 上重训 (test 为时间尾部 held-out betan 156434–159841;
# norm 用 19 维 notime 在 betan train 重算)。
# 模型与随机划分消融一致 (仅改划分/norm/out_root/seed)。每个 array task 训一对,
# SLURM 按空闲节点持续 backfill -> 最快墙钟。
# 输出 results/betan_ablation/<CFG>/<CFG>_betan_s<SEED>/。
#
# 提交:  sbatch scripts_ablation/submit_betan_ablation.sbatch
# 查看:  squeue -u user ;  tail -f results/betan_ablation/slurm_<jobid>_<arrayid>.out
# 取消:  scancel <jobid>          (整个数组)

set -euo pipefail
cd /home/user/PF_current_EAST

set +eu; source /opt/dtk/env.sh; set -eu
_GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
export DTK_ENV_SOURCED=1 MIOPEN_CPLUS_FIXED=1

# array id -> (cfg, seed): idx%6 选配置, idx/6 选 seed
CONFIGS=(transformer_bidir_on transformer_off transformer_on transformer_bidir_off lstm_off mlp_off)
SEEDS=(11 44 20260424)
i=${SLURM_ARRAY_TASK_ID}
export CFG=${CONFIGS[$(( i % 6 ))]}
export SEED=${SEEDS[$(( i / 6 ))]}
export NPROC=8

echo "[$(date)] array task $i: CFG=$CFG SEED=$SEED on $(hostname) (job ${SLURM_JOB_ID}, array ${SLURM_ARRAY_TASK_ID})"
bash scripts_ablation/run_betan_ablation.sh
echo "[$(date)] array task $i ($CFG/$SEED) finished on $(hostname)"
