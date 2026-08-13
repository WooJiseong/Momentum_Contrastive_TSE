#!/usr/bin/env bash
#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=soft_indiv_no1n
#SBATCH --output=exp/20260806_soft_indiv_mococo_no_1n/slurm-%j.out

set -euo pipefail

export RUN_DIR="${RUN_DIR:-exp/20260806_soft_indiv_mococo_no_1n}"
export MOCO_CFG="${MOCO_CFG:-configs/config_soft_indiv_moco_no_1n.yaml}"
export NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29860}"

bash script/train_eval_20260804.sh moco
