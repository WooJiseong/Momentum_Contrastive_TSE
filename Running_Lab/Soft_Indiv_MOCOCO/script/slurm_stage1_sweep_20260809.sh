#!/usr/bin/env bash
#
# Submit one Soft_Indiv_MOCOCO Stage1 sweep for an existing Stage0 encoder.
# The caller supplies RUN_DIR and STAGE0_SELECTOR so the same script can be
# used for the original and no-1/n Stage0 experiments.
#
# Example:
#   sbatch --export=ALL,RUN_DIR=exp/20260804_soft_indiv_mococo,STAGE0_SELECTOR=50 \
#     script/slurm_stage1_sweep_20260809.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=soft_indiv_stage1

set -euo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${STAGE0_SELECTOR:?STAGE0_SELECTOR is required}"

export NUM_GPUS=4
export REUSE_RUN=1
export STAGE0_SELECTOR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29850}"

bash script/train_eval_20260804.sh stage1
