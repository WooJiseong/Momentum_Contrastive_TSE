#!/usr/bin/env bash
#
# Purpose:
#   Submit the Soft_MOCOCO epoch-134 Stage1 Soft-leakage sweep and evaluation.
#
# Run from:
#   contrastive_momentum/Running_Lab/Soft_MOCOCO
#
# Single GPU:
#   Not supported by this fixed 4-GPU Slurm submission.
#
# Multi GPU:
#   sbatch script/slurm_stage1_softleakage_20260805_stage0_134ep.sh
#
# Required arg:
#   None.
#
# Output:
#   exp/20260805_stage0_134ep_softleakage_test/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   mkdir -p exp/20260805_stage0_134ep_softleakage_test
#   sbatch script/slurm_stage1_softleakage_20260805_stage0_134ep.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=softleak_134ep
#SBATCH --output=exp/20260805_stage0_134ep_softleakage_test/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29934}"
export SOFT_MOCOCO_ALLOW_EXISTING_RUN=1

bash script/train_stage1_softleakage_20260805_stage0_134ep.sh all
