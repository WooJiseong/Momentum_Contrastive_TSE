#!/usr/bin/env bash
# Train t-predictor against the Attn_MOCOCO Best Stage0 encoder.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=attn_tpred
#SBATCH --output=exp/20260824_attn_mococo_tpred/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29996}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TP_CONFIG="${TP_CONFIG:-$PWD/configs/config_tpred_attn_best.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO/exp/20260813_attn_mococo_40frame_lightweight_attention/stage0_moco/checkpoints/pn_encoder_best.pt}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_tpred_numba_cache}"

mkdir -p exp/20260824_attn_mococo_tpred
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"
exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_t_predicter_mococo.py \
  --config "$TP_CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT"
