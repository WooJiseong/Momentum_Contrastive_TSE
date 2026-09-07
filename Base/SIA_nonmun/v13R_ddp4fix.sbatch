#!/bin/bash -l
#SBATCH --partition=gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=d8_v13R_gbs8

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tse310

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# 첫 4 optimizer step에서 gradient all-reduce 검증
export DDP_VERIFY=1

CONFIG="config/config_a6000_depth8_ddp4_gbs8_bpg2_scratch_157e_v13R_DDP4FIX.yaml"

echo "============================================================"
echo "JOB=$SLURM_JOB_ID"
echo "CONFIG=$CONFIG"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HOST=$(hostname)"
echo "============================================================"

echo
echo "===== SLURM FOUR-TASK PREFLIGHT ====="

srun bash -lc '
echo "[SLURM_TASK] procid=$SLURM_PROCID localid=$SLURM_LOCALID ntasks=$SLURM_NTASKS visible=$CUDA_VISIBLE_DEVICES host=$(hostname)"
'

echo
echo "===== TRAIN ====="

srun --kill-on-bad-exit=1     python -u train_meanflow.py     --config "$CONFIG"
