#!/usr/bin/env bash
# Submit the complete Attn_MOCOCO Stage0 + two Stage1 loss variants workflow.
# Run from Running_Lab/Attn_MOCOCO:
#   sbatch script/slurm_train_attn_mococo.sh

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
# gpu4 nodes expose four A6000 GPUs; this is a per-node request.
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_mococo
#SBATCH --output=exp/20260813_attn_mococo_40frame_lightweight_attention/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29990}"
export PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"

bash script/run_train.sh all
