#!/usr/bin/env bash
# Submit scheduled-teacher Stage0 and its dependent Flow/t-predictor jobs.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p \
  exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage0_moco \
  exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/stage1_flow_oracle \
  exp/20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03/tpred

stage0_job="$(sbatch --parsable script/slurm_train_stage0_teacher_decay.sh)"
flow_job="$(sbatch --parsable --dependency="afterok:${stage0_job}" script/slurm_train_flow_teacher_decay.sh)"
tpred_job="$(sbatch --parsable --dependency="afterok:${stage0_job}" script/slurm_train_tpred_teacher_decay.sh)"

printf 'stage0_teacher_cosine=%s\nflow_oracle_teacher_cosine=%s\ntpredictor_teacher_cosine=%s\n' \
  "$stage0_job" "$flow_job" "$tpred_job"
