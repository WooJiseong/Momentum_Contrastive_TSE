#!/usr/bin/env bash
# Paper-matched Flow Stage1 using the 360-only Stage0 export and MR-jitter.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=imp360_mrflow
#SBATCH --output=exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/stage1_flow_mrjitter/slurm-%j.out

set -euo pipefail

RUN_ROOT="$PWD/exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/stage1_flow_mrjitter"
CONFIG="$PWD/configs/config_flow_360only_mrjitter.yaml"
STAGE0_CKPT="${STAGE0_CKPT:-$PWD/exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/stage0_moco/checkpoints/pn_encoder_best.pt}"
export MASTER_PORT="${MASTER_PORT:-30118}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_360only_mrflow_numba_cache_${USER:-user}}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_ROOT/checkpoints"
cp -f "$CONFIG" "$PWD/exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/config_flow_360only_mrjitter.yaml"
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_improved_meanflow.py \
  --config "$CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --run-root "$RUN_ROOT"
