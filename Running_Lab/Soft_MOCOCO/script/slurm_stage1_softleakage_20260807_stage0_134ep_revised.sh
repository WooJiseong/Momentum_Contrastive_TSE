#!/usr/bin/env bash
#
# Submit the clean epoch-134 Stage1 revised Leakage Loss experiment.
#
# Run from Running_Lab/Soft_MOCOCO:
#   sbatch script/slurm_stage1_softleakage_20260807_stage0_134ep_revised.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=softleak_134rev
#SBATCH --output=exp/20260807_stage0_134ep_softleakage_revised_loss_test/slurm-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29974}"

bash script/train_stage1_softleakage_20260807_stage0_134ep_revised.sh all
