#!/usr/bin/env bash
# Purpose: render TensorBoard scalar events to PNG using Pillow.
# Run from: contrastive_momentum
# Submit: sbatch slurm_render_depth_sweep_png_v2.sh
#
#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum
#SBATCH -p gpu6
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=tb_depth_png
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/exp/20260801_tensorboard_depth_sweep/png-v2-%j.out
set -euo pipefail
python render_depth_sweep_png_v2.py

