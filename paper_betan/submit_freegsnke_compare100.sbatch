#!/bin/bash
#SBATCH --job-name=fg_cmp
#SBATCH --partition=shenmadcu
#SBATCH --array=0-47                  # 默认 48 分片 (PCS 纯 CPU); 预测跑请 --array=0-23 + --gres=dcu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # Phase2 是 numba 单线程逐帧 fork, 每分片~2-3 核够用
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=results/paper_betan/freegsnke_compare100_logs/slurm_%A_%a.out
#
# pf6keep 随机 100 炮 FreeGSNKE 对照跑: PCS 实测电流 vs 模型预测电流, 同 pipeline 只换电流源。
# 节点: compute-node381/382/383 (各 8 DCU, 64/64/96 CPU)。⚠️ FreeGSNKE 求解(Phase2)是 CPU/numba, 不吃 GPU;
#   GPU 只在预测跑的 Phase1 推理(逐炮~2s)用到 → PCS 跑跳过模型, 纯 CPU, 无需 --gres, 可开满分片。
#
# 用法:
#   PCS 实测电流跑(默认, 纯CPU, 48分片):   sbatch paper_betan/submit_freegsnke_compare100.sbatch
#   预测电流跑(需GPU, 24分片):             CURRENT_SOURCE=pred sbatch --gres=dcu:1 --array=0-23 paper_betan/submit_freegsnke_compare100.sbatch
#   跑完对比:                              python -m paper_betan.compare_pred_vs_pcs
set -euo pipefail
cd /home/user/PF_current_EAST
set +eu; source /home/user/anaconda3/envs/torch/../opt/dtk-25.04/env.sh 2>/dev/null || source /opt/dtk/env.sh; set -eu
_GV=$(gcc -dumpversion | cut -d. -f1); _GM=$(gcc -dumpmachine)
export CPLUS_INCLUDE_PATH="/usr/include/c++/${_GV}:/usr/include/${_GM}/c++/${_GV}:${CPLUS_INCLUDE_PATH:-}"
export DTK_ENV_SOURCED=1 MIOPEN_CPLUS_FIXED=1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1          # 单帧 numba 多线程无收益; 靠多分片并行
export NUMBA_CACHE_DIR=/home/user/PF_current_EAST/.numba_cache
mkdir -p "$NUMBA_CACHE_DIR" results/paper_betan/freegsnke_compare100_logs

CS=${CURRENT_SOURCE:-pcs}
SHOTS=${TEST_SHOTS:-meta/split_by_order_betan/freegsnke_pf6keep_random100_s20260718.txt}
FPP=${FPP:-6}; NX=${NX:-65}
case "$CS" in
  pcs)  DEF_OUT=results/paper_betan/freegsnke_testset_pf6keep_pcs100 ;;
  pred) DEF_OUT=results/paper_betan/freegsnke_testset_pf6keep_pred100 ;;
  *)    echo "CURRENT_SOURCE 必须是 pcs 或 pred"; exit 1 ;;
esac
OUT=${OUT_DIR:-$DEF_OUT}; mkdir -p "$OUT"
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-48}   # --array=0-47→48; 预测跑 --array=0-23→24, 自动适配

PY=python3
echo "[$(date)] shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS} on $(hostname) (job ${SLURM_JOB_ID}), CURRENT_SOURCE=${CS} SHOTS=${SHOTS} FPP=${FPP} NX=${NX} OUT=${OUT}"
$PY -m paper_betan.freegsnke_testset_eval \
  --current-source ${CS} \
  --n-shots 999999 --frames-per-phase ${FPP} \
  --test-shots ${SHOTS} \
  --out-dir ${OUT} \
  --n-shards ${NSHARDS} --shard-id ${SLURM_ARRAY_TASK_ID} --n-proc 4 --nx ${NX}
echo "[$(date)] shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS} DONE on $(hostname) (CS=${CS})"
