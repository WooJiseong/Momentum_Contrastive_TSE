#!/usr/bin/env bash
# Purpose: extract and plot 50-epoch TensorBoard scalar curves for the depth sweep.
# Run from: contrastive_momentum
# Submit: sbatch slurm_plot_depth_sweep_tensorboard_v2.sh
#
#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum
#SBATCH -p gpu6
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=tb_depth_plot
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/exp/20260801_tensorboard_depth_sweep/slurm-%j.out

set -euo pipefail
python plot_depth_sweep_tensorboard.py

