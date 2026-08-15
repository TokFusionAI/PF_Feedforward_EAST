#!/bin/bash
#SBATCH --job-name=paper_betan
#SBATCH --partition=shenmadcu
#SBATCH --gres=dcu:1                      # predict/export 推理单卡; FreeGSNKE/画图 CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # FreeGSNKE(numba) + 数据加载 + 画图
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=results/paper_betan/slurm_%j.out
#
# 时序 transformer_bidir_on (cross-betan best seed s20260424) 完整验证: 复刻 paper_best_timepe 全部图。
# 模型加载已指向 results/betan_ablation/.../transformer_bidir_on_betan_s20260424/best_val.pt。
# freegsnke: 导出 betan test 代表炮全部时间片 (bc_pred+pcs) -> 全片求解 -> 按爬升/平顶/下降选 8 控制点误差最小片画图。
#
# 提交:  SHOT=<betan test 代表炮> sbatch paper_betan/submit.sbatch
# 查看:  squeue -u user ;  tail -f results/paper_betan/slurm_<jobid>.out

set -euo pipefail
cd /home/user/PF_current_EAST
export SHOT=${SHOT:?需传 betan test 代表炮号}

bash paper_betan/run_all.sh
echo "[$(date)] paper_betan DONE on $(hostname)"
echo ">>> 图: results/paper_betan/figures/   指标: .../eval_test/metrics_summary.json"
