#!/usr/bin/env bash
# One-GPU Stage0 sweep for Attn_MOCOCO teacher-loss weights.
# Submit this file three times with different SWEEP_CONFIG values.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=attn_tchr_swp
#SBATCH --output=exp/20260819_attn_teacher_weight_sweep/slurm-%j.out

set -euo pipefail

PROJECT_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO"
PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
SWEEP_CONFIG="${SWEEP_CONFIG:?Set SWEEP_CONFIG to one sweep YAML.}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$PROJECT_DIR/../../..:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_teacher_sweep_${SLURM_JOB_ID:-local}}"

mkdir -p exp/20260819_attn_teacher_weight_sweep "$NUMBA_CACHE_DIR"
echo "config=$SWEEP_CONFIG"
echo "python=$PYTHON_BIN"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" train_attn_moco_encoder.py --config "$SWEEP_CONFIG"
