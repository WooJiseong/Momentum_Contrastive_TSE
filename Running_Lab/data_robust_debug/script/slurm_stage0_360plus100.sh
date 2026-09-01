#!/usr/bin/env bash
# Fresh Soft_MOCOCO Stage0: LibriSpeech train-clean-360 + train-clean-100.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/data_robust_debug
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=drb_360p100_s0
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/data_robust_debug/exp/20260831_soft_mococo_360plus100_fixed_teacher0p1/stage0_moco/slurm-%j.out

set -euo pipefail
LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/data_robust_debug"
export MASTER_PORT="${MASTER_PORT:-30610}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/data_robust_debug_360plus100_${USER:-user}}"
export RESUME_CKPT="$LAB_DIR/exp/20260831_soft_mococo_360plus100_fixed_teacher0p1/stage0_moco/checkpoints/last.ckpt"
mkdir -p "$LAB_DIR/exp/20260831_soft_mococo_360plus100_fixed_teacher0p1/stage0_moco/checkpoints"
exec bash "$LAB_DIR/script/run_stage0.sh" "$LAB_DIR/exp/20260831_soft_mococo_360plus100_fixed_teacher0p1/config.yaml"
