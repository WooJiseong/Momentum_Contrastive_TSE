#!/usr/bin/env bash
# Improved Attn Stage0 with teacher weight cosine decay: 0.10 -> 0.03.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=imp_attn_cos_s0
#SBATCH --output=exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage0_moco/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-30113}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_teacher_cosine_numba_cache_${USER:-user}}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage0_moco
exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_improved_attn_moco_encoder.py \
  --config "$PWD/configs/config_stage0_teacher_weight_cosine.yaml"
