#!/usr/bin/env bash
# Final Improved t-predictor trained against exactly the same Stage0 export.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=imp_tpred
#SBATCH --output=exp/20260824_improved_attn_mococo_soft03_flow/tpred/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-30112}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_tpred_numba_cache_${USER:-user}}"
export STAGE0_CKPT="${STAGE0_CKPT:-$PWD/exp/20260824_improved_attn_mococo_soft03_flow/stage0_moco/checkpoints/pn_encoder_best.pt}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" exp/20260824_improved_attn_mococo_soft03_flow/tpred
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }
exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_improved_tpred.py \
  --config "$PWD/configs/config_tpred.yaml" \
  --stage0-ckpt "$STAGE0_CKPT"

