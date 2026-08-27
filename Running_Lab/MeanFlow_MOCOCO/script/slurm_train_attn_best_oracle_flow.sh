#!/usr/bin/env bash
# Train the default rectified-flow decoder with Attn_MOCOCO Best Stage0.
# No t-predictor is loaded: validation uses the dataset oracle mixing ratio.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_oracle_mf
#SBATCH --output=exp/20260824_attn_mococo_default_flow_oracle_stage1/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29997}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MF_CONFIG="${MF_CONFIG:-$PWD/configs/config_meanflow_attn_best_oracle.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO/exp/20260813_attn_mococo_40frame_lightweight_attention/stage0_moco/checkpoints/pn_encoder_best.pt}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_meanflow_numba_cache}"
export RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p exp/20260824_attn_mococo_default_flow_oracle_stage1
bash script/train_meanflow_mococo.sh
