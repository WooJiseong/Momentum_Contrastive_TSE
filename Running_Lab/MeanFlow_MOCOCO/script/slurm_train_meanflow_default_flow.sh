#!/usr/bin/env bash
# Submit the pure default-flow MeanFlow Stage1 experiment.
# The previous 20260816 run is intentionally kept separate because its
# objective used MR-jitter and gamma=0.5.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=mf_mococo_df
#SBATCH --output=exp/20260818_soft_mococo_meanflow_default_flow_stage1/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29993}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MF_CONFIG="${MF_CONFIG:-$PWD/configs/config_meanflow_mococo_default_flow.yaml}"

mkdir -p exp/20260818_soft_mococo_meanflow_default_flow_stage1
bash script/train_meanflow_mococo.sh
