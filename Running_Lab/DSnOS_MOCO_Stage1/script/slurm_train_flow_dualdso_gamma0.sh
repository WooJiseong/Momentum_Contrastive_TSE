#!/usr/bin/env bash
# DSnOS Stage0 encoder + dualdso_loss.py MR-jitter Flow Stage1, 360-only.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/DSnOS_MOCO_Stage1
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=dsnos_dso
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/DSnOS_MOCO_Stage1/exp/20260906_dsnos_moco_stage1/flow_dualdso_gamma0/slurm-%j.out

set -euo pipefail

ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
LAB_DIR="$ROOT/Running_Lab/DSnOS_MOCO_Stage1"
CONFIG="${DSNOS_CONFIG:-$LAB_DIR/configs/config_flow_dualdso_gamma0_360only.yaml}"

export MASTER_PORT="${MASTER_PORT:-30332}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/dsnos_moco_stage1_dualdso_${USER:-user}}"
export MEANFLOW_NUM_WORKERS="${MEANFLOW_NUM_WORKERS:-4}"
# DualDSO always executes the same two model forwards per step.  Its graph is
# static, so avoid DDP's per-step unused-parameter traversal and the epoch-end
# collective ordering issue observed in the first run.
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-0}"
export PYTHONPATH="$LAB_DIR:$ROOT/Base/Code_Snippet:$ROOT/Running_Lab/MeanFlow_MOCOCO:$ROOT/Running_Lab/PN_MOCOCO:$ROOT:${PYTHONPATH:-}"

mkdir -p "$LAB_DIR/exp/20260906_dsnos_moco_stage1/flow_dualdso_gamma0/checkpoints_2gpu" \
  "$LAB_DIR/exp/20260906_dsnos_moco_stage1/flow_dualdso_gamma0/logs/retry_2gpu"
echo "config=$CONFIG"
echo "speedups=Base/Code_Snippet/speedups.py"
echo "loss_impl=Base/Code_Snippet/dualdso_loss.py"
echo "DDP_FIND_UNUSED=$DDP_FIND_UNUSED"

exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  "$LAB_DIR/script/train_meanflow_dsnos.py" \
  --config "$CONFIG" \
  --loss-impl dualdso
