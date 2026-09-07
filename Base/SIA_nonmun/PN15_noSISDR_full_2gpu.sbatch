#!/bin/bash
#SBATCH --job-name=PN15_noSI
#SBATCH --partition=gpu2,gpu4,gpu5,gpu6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/PN15_noSI_%j.out
#SBATCH --error=logs/PN15_noSI_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tse310

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

srun --kill-on-bad-exit=1 \
    python -u train_PN12_dualdso_scratch.py \
    --config config/config_PN15_v13R_dualdso_noSISDR_scratch_clean360.yaml

echo "PN15_NO_SISDR_FULL: PASS"
