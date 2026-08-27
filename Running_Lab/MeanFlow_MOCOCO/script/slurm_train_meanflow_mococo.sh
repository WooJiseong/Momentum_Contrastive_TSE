#!/usr/bin/env bash
# Submit MeanFlow Stage1 conditioned by a trained MOCOCO Stage0 encoder.
#
# Default Stage0: Soft_MOCOCO 50-epoch export.
# Alternate Stage0 example:
#   sbatch --export=ALL,STAGE0_CKPT=/path/to/pn_encoder_best.pt \
#     script/slurm_train_meanflow_mococo.sh

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu4
# gpu4 nodes expose four A6000 GPUs; this requests all four on one node.
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=mf_mococo_s1
#SBATCH --output=exp/20260816_soft_mococo_meanflow_stage1/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29992}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p exp/20260816_soft_mococo_meanflow_stage1
bash script/train_meanflow_mococo.sh
