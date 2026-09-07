#!/usr/bin/env bash
# Soft_MOCOCO DSnOS: 360-only split into 78.6:21.4 speaker partitions,
# sampled 50:50 to reproduce the 360+100 oversampling ratio.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_DSnOS
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=dsnos360_s0
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_DSnOS/exp/20260901_soft_moco_dsnos_360only_split50/stage0_moco/slurm-%j.out

set -euo pipefail
LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_DSnOS"
export MASTER_PORT="${MASTER_PORT:-30740}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_moco_dsnos_360only_split50_${USER:-user}}"
export RESUME_CKPT="$LAB_DIR/exp/20260901_soft_moco_dsnos_360only_split50/stage0_moco/checkpoints/last.ckpt"
if [[ ! -f "$RESUME_CKPT" ]]; then
    unset RESUME_CKPT
fi

mkdir -p "$LAB_DIR/exp/20260901_soft_moco_dsnos_360only_split50/stage0_moco/checkpoints"
exec bash "$LAB_DIR/script/run_stage0.sh" "$LAB_DIR/configs/config_soft_moco_dsnos_360only_split50.yaml"
