#!/usr/bin/env bash
# Purpose: extract TensorBoard scalar events to CSV and Markdown without plotting dependencies.
# Run from: contrastive_momentum
# Submit: sbatch slurm_extract_depth_sweep_tensorboard.sh
#
#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum
#SBATCH -p gpu6
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=tb_depth_extract
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/exp/20260801_tensorboard_depth_sweep/extract-%j.out
set -euo pipefail
python extract_depth_sweep_tensorboard.py

