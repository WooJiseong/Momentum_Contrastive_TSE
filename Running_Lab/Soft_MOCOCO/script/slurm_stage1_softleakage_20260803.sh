#!/usr/bin/env bash
# Purpose: submit the Stage0 epoch-50 Soft-leakage Stage1 and Eval workflow.
# Run from: Running_Lab/Soft_MOCOCO
# Usage: sbatch script/slurm_stage1_softleakage_20260803.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=softleak_50ep
#SBATCH --output=exp/20260803_stage0_50ep_softleakage_test/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29930}"

bash script/train_stage1_softleakage_20260803.sh all
