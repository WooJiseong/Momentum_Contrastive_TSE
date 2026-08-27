#!/usr/bin/env bash
# Train the t-predictor with the same Soft_MOCOCO Stage0 used by MeanFlow.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=mf_tpred_s50
#SBATCH --output=exp/20260820_soft50_tpred/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29990}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TP_CONFIG="${TP_CONFIG:-$PWD/configs/config_tpred_mococo_soft50.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260803_stage0_50ep_soft_wo_leakage_test/stage0_moco/checkpoints/pn_encoder_50ep.pt}"

mkdir -p exp/20260820_soft50_tpred
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"
exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  train_t_predicter_mococo.py \
  --config "$TP_CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT"
