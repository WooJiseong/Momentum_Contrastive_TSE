#!/usr/bin/env bash
#SBATCH -J depth_collect
#SBATCH -p gpu6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterany:849666:850424:850425:850426
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/depth_sweep_collect-%j.out

set -euo pipefail

cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum

python collect_depth_sweep_results.py
sacct -j 849662,849663,849666,850424,850425,850426 --format=JobID,JobName%24,State,Elapsed,ExitCode,NodeList -P > depth_sweep_final_sacct_20260730.txt
