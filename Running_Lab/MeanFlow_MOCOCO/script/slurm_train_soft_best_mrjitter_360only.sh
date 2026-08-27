#!/usr/bin/env bash
# Soft_MOCOCO Best Stage0 conditioned Flow Stage1 with MR-jitter.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=soft360_mrflow
#SBATCH --output=exp/20260826_soft_mococo_best_mrjitter_360only/slurm-%j.out

set -euo pipefail

EXP_DIR="$PWD/exp/20260826_soft_mococo_best_mrjitter_360only"
CONFIG="$PWD/configs/config_meanflow_soft_best_mrjitter_360only.yaml"
STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260801_soft_mococo/stage0_moco/checkpoints/pn_encoder_best.pt}"

export MASTER_PORT="${MASTER_PORT:-29996}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MF_CONFIG="$CONFIG"
export STAGE0_CKPT
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_best_mrjitter_360only_numba_cache_${USER:-user}}"

mkdir -p "$EXP_DIR/checkpoints" "$NUMBA_CACHE_DIR"
cp -f "$CONFIG" "$EXP_DIR/config_meanflow_soft_best_mrjitter_360only.yaml"
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

exec bash script/train_meanflow_mococo.sh
