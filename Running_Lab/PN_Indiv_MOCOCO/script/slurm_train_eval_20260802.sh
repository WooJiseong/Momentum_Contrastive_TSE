#!/usr/bin/env bash
#
# Purpose:
#   Submit the 20260801 PN_Indiv_MOCOCO Stage0 -> Stage1 -> Eval workflow to Slurm.
#
# Run from:
#   Running_Lab/PN_Indiv_MOCOCO
#
# Single GPU:
#   Not supported by this fixed 4-GPU experiment script.
#
# Multi GPU:
#   sbatch script/slurm_train_eval_20260801.sh
#
# Required arg:
#   None. The script runs `script/train_eval_20260801.sh all`.
#
# Output:
#   exp/20260801_indiv_mococo/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   mkdir -p exp/20260801_indiv_mococo
#   sbatch script/slurm_train_eval_20260801.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=pn_indiv_moco
#SBATCH --output=exp/20260801_indiv_mococo/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29820}"

bash script/train_eval_20260802.sh all
