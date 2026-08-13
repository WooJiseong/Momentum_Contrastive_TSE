#!/usr/bin/env bash
# Purpose: submit the Stage0 epoch-50 Stage1 run without leakage loss and its evaluation.
# Run from: Running_Lab/Soft_MOCOCO
# Usage: sbatch script/slurm_stage1_wo_leakage_20260803.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=softwoleak_50ep
#SBATCH --output=exp/20260803_stage0_50ep_soft_wo_leakage_test/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29931}"

bash script/train_stage1_wo_leakage_20260803.sh all
