#!/usr/bin/env bash
#
# Run Soft_Indiv_MOCOCO Stage1(best Stage0 encoder) and Eval after the
# dependent Stage0 resume Job completes successfully.
#
# Run from Running_Lab/Soft_Indiv_MOCOCO:
#   sbatch --dependency=afterok:<STAGE0_JOB_ID> script/slurm_stage1_eval_after_resume_20260808.sh

#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=soft_indiv_stage1
#SBATCH --output=exp/20260804_soft_indiv_mococo/slurm-stage1-resume-%j.out

set -euo pipefail

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29842}"
export REUSE_RUN=1
export STAGE0_SELECTOR=best

bash script/train_eval_20260804.sh stage1
bash script/train_eval_20260804.sh eval
