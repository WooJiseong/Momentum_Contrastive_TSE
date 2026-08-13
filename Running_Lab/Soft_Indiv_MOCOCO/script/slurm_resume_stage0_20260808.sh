#!/usr/bin/env bash
#
# Resume the interrupted Soft_Indiv_MOCOCO Stage0 run from its last full
# Lightning checkpoint. Stage1 is submitted separately with a dependency.
#
# Run from Running_Lab/Soft_Indiv_MOCOCO:
#   sbatch script/slurm_resume_stage0_20260808.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --job-name=soft_indiv_resume0
#SBATCH --output=exp/20260804_soft_indiv_mococo/slurm-resume-stage0-%j.out

set -euo pipefail

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29841}"
export REUSE_RUN=1
export RESUME_STAGE0="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO/exp/20260804_soft_indiv_mococo/stage0_moco/checkpoints/last.ckpt"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

bash script/train_eval_20260804.sh moco
