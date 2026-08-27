#!/usr/bin/env bash
# Four-condition, evaluation-aligned Soft_MOCOCO t-predictor.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=soft4c_tpred
#SBATCH --output=exp/20260826_soft_best_tpred_multicondition/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29999}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_best_tpred_multicondition_numba_cache}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

CONFIG="${TP_CONFIG:-$PWD/configs/config_tpred_mococo_soft_best_multicondition.yaml}"
STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260801_soft_mococo/stage0_moco/checkpoints/pn_encoder_best.pt}"
export STAGE0_CKPT

mkdir -p exp/20260826_soft_best_tpred_multicondition
echo "[tpred] config=$CONFIG"
echo "[tpred] stage0=$STAGE0_CKPT"
echo "[tpred] conditions=2/2,2/3,3/2,3/3 segment=6 enroll=3"

exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_t_predicter_mococo_multicondition.py \
  --config "$CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT"
