#!/usr/bin/env bash
#
# Purpose:
#   Submit PN_Indiv_MOCOCO TFGridNet Stage1 with individual-source leakage to Slurm.
#
# Run from:
#   contrastive_momentum/Running_Lab/PN_Indiv_MOCOCO
#
# Single GPU:
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash script/train_tfgridnet_stage1_20260804.sh
#
# Multi GPU:
#   sbatch script/slurm_tfgridnet_stage1_20260804.sh
#
# Required arg:
#   None.
#
# Output:
#   exp/20260804_indiv_mococo_stage1_individual_nuisance/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   mkdir -p exp/20260804_indiv_mococo_stage1_individual_nuisance
#   sbatch script/slurm_tfgridnet_stage1_20260804.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=pn_indiv_stage1
#SBATCH --output=exp/20260804_indiv_mococo_stage1_individual_nuisance/slurm-%j.out

set -euo pipefail

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29850}"

bash script/train_tfgridnet_stage1_20260804.sh
