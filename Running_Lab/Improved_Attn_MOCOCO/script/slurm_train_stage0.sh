#!/usr/bin/env bash
# Final Improved Stage0: Attn_MOCOCO + Soft fixed-anchor teacher (weight 0.03).

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=imp_attn_s0
#SBATCH --output=exp/20260824_improved_attn_mococo_soft03_flow/stage0_moco/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-30110}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_stage0_numba_cache_${USER:-user}}"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" exp/20260824_improved_attn_mococo_soft03_flow/stage0_moco
resume_args=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  [[ -f "$RESUME_CKPT" ]] || { echo "Resume checkpoint not found: $RESUME_CKPT" >&2; exit 1; }
  resume_args+=(--resume "$RESUME_CKPT")
fi
exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_improved_attn_moco_encoder.py \
  --config "$PWD/configs/config_stage0.yaml" \
  "${resume_args[@]}"
