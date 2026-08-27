#!/usr/bin/env bash
# Improved Attn Stage0, paper-matched train-clean-360-only experiment.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=imp_attn360_s0
#SBATCH --output=exp/20260826_improved_attn_mococo_soft03_360only/stage0_moco/slurm-%j.out

set -euo pipefail

RUN_ROOT="$PWD/exp/20260826_improved_attn_mococo_soft03_360only"
CONFIG="$PWD/configs/config_stage0_360only.yaml"

export MASTER_PORT="${MASTER_PORT:-30116}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_360only_stage0_numba_cache_${USER:-user}}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_ROOT/stage0_moco" "$RUN_ROOT/stage0_moco/checkpoints"
cp -f "$CONFIG" "$RUN_ROOT/config_stage0_360only.yaml"

exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_improved_attn_moco_encoder.py \
  --config "$CONFIG"
