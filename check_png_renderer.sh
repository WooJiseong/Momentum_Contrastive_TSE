#!/usr/bin/env bash
# Purpose: check available non-matplotlib PNG renderers on a compute node.
# Submit: sbatch check_png_renderer.sh
#
#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum
#SBATCH -p gpu6
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=check_png_renderer
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/exp/20260801_tensorboard_depth_sweep/png-check-%j.out
set -euo pipefail
python -c 'import importlib.util; print("PIL=" + str(importlib.util.find_spec("PIL") is not None)); print("cairosvg=" + str(importlib.util.find_spec("cairosvg") is not None))'
command -v convert || true
command -v rsvg-convert || true

