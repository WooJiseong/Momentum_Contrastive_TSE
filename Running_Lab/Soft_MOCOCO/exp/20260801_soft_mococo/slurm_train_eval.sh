#!/usr/bin/env bash
# Purpose: resume/train/evaluate the 20260801 Soft_MOCOCO experiment.
# Submit: sbatch exp/20260801_soft_mococo/slurm_train_eval.sh

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --job-name=soft_mococo
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260801_soft_mococo/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${MASTER_PORT:-29920}"

bash script/train_eval_20260801.sh all
