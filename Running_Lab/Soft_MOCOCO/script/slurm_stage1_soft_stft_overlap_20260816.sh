#!/usr/bin/env bash
# Submit Soft_MOCOCO Stage1 STFT magnitude-overlap Gate and 5000-item Eval.
# Run from Running_Lab/Soft_MOCOCO:
#   sbatch script/slurm_stage1_soft_stft_overlap_20260816.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --job-name=softstftgate
#SBATCH --output=exp/20260816_stage0_50ep_soft_stft_overlap_gate/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29986}"

bash script/train_stage1_soft_stft_overlap_20260816.sh all
