#!/usr/bin/env bash
# Single-GPU Oracle Default Flow fallback for the Attn_MOCOCO Best encoder.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_oracle_1g
#SBATCH --output=exp/20260824_attn_mococo_default_flow_oracle_stage1_1gpu/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-30019}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MF_CONFIG="${MF_CONFIG:-$PWD/configs/config_meanflow_attn_best_oracle_1gpu.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO/exp/20260813_attn_mococo_40frame_lightweight_attention/stage0_moco/checkpoints/pn_encoder_best.pt}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_meanflow_1gpu_numba_cache}"

mkdir -p exp/20260824_attn_mococo_default_flow_oracle_stage1_1gpu
bash script/train_meanflow_mococo.sh
