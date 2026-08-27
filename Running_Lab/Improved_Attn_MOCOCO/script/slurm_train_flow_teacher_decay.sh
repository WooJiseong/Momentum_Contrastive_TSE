#!/usr/bin/env bash
# Flow Oracle Stage1 for the scheduled-teacher Stage0 export.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=imp_cos_flow
#SBATCH --output=exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage1_flow_oracle/slurm-%j.out

set -euo pipefail

RUN_ROOT="$PWD/exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage1_flow_oracle"
STAGE0_CKPT="${STAGE0_CKPT:-$PWD/exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage0_moco/checkpoints/pn_encoder_best.pt}"
export MASTER_PORT="${MASTER_PORT:-30114}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_teacher_cosine_flow_numba_cache_${USER:-user}}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_ROOT"
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }
exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_improved_meanflow.py \
  --config "$PWD/configs/config_flow_oracle.yaml" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --run-root "$RUN_ROOT"
