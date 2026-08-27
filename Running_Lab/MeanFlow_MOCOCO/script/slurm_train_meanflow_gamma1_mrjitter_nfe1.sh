#!/usr/bin/env bash
# Corrected MeanFlow Stage1 after the matching t-predictor job succeeds.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=mf_s50_g1_mrj
#SBATCH --output=exp/20260820_soft50_meanflow_gamma1_mrjitter_nfe1/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29991}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MF_CONFIG="${MF_CONFIG:-$PWD/configs/config_meanflow_mococo_gamma1_mrjitter_nfe1.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260803_stage0_50ep_soft_wo_leakage_test/stage0_moco/checkpoints/pn_encoder_50ep.pt}"

mkdir -p exp/20260820_soft50_meanflow_gamma1_mrjitter_nfe1
bash script/train_meanflow_mococo.sh
