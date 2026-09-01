#!/usr/bin/env bash
# Soft Stage0 with one unified train-clean-360 speaker pool.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_SamplingBug_fixed
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=smf360_s0
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_SamplingBug_fixed/exp/20260901_soft_sampling_fixed_360/stage0_moco/slurm-%j.out

set -euo pipefail
LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_SamplingBug_fixed"
export MASTER_PORT="${MASTER_PORT:-30710}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_sampling_fixed_360_${USER:-user}}"
unset RESUME_CKPT
mkdir -p "$LAB_DIR/exp/20260901_soft_sampling_fixed_360/stage0_moco/checkpoints"
exec bash "$LAB_DIR/script/run_stage0.sh" "$LAB_DIR/configs/config_soft_sampling_fixed_360.yaml"
