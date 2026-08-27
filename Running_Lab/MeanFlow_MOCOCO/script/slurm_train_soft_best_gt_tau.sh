#!/usr/bin/env bash
# MeanFlow Stage1 using Soft_MOCOCO Best Stage0 and GT tau.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=mf_soft_best_gt
#SBATCH --output=exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1/slurm-%j.out

set -euo pipefail

EXP_DIR="exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1"
export MASTER_PORT="${MASTER_PORT:-29994}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MF_CONFIG="${MF_CONFIG:-$PWD/configs/config_meanflow_soft_best_gt_tau.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260801_soft_mococo/stage0_moco/checkpoints/pn_encoder_best.pt}"

mkdir -p "$EXP_DIR"
bash script/train_meanflow_mococo.sh
