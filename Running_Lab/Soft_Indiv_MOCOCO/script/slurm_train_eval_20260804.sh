#!/usr/bin/env bash
#
# Purpose:
#   Submit the Soft_Indiv_MOCOCO Stage0 -> Stage1(best) -> Eval workflow to Slurm.
#
# Run from:
#   contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO
#
# Single GPU:
#   Not supported by this Slurm submission. Use train_eval_20260804.sh with NUM_GPUS=1.
#
# Multi GPU:
#   sbatch script/slurm_train_eval_20260804.sh
#
# Required arg:
#   None.
#
# Output:
#   exp/20260804_soft_indiv_mococo/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   mkdir -p exp/20260804_soft_indiv_mococo
#   sbatch script/slurm_train_eval_20260804.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=soft_indiv_moco
#SBATCH --output=exp/20260804_soft_indiv_mococo/slurm-%j.out

set -euo pipefail

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29840}"

bash script/train_eval_20260804.sh all
