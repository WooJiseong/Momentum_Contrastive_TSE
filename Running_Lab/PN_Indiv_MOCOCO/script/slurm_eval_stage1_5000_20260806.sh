#!/usr/bin/env bash
#SBATCH -p gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=pn_indiv_eval5k
#SBATCH --output=exp/20260804_indiv_mococo_stage1_individual_nuisance/slurm-eval-%j.out

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash script/eval_stage1_5000_20260806.sh
