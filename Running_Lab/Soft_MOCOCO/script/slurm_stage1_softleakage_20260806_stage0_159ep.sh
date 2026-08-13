#!/usr/bin/env bash
#
# Submit the Soft_MOCOCO epoch-159 Stage1 Soft-leakage sweep and evaluation.
#
# Run from contrastive_momentum/Running_Lab/Soft_MOCOCO:
#   sbatch script/slurm_stage1_softleakage_20260806_stage0_159ep.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=softleak_159ep
#SBATCH --output=exp/20260806_stage0_159ep_softleakage_test/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29959}"
export SOFT_MOCOCO_ALLOW_EXISTING_RUN=1

bash script/train_stage1_softleakage_20260806_stage0_159ep.sh all
