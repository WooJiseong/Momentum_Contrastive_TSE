#!/usr/bin/env bash
# Submit Stage0 first, then Oracle Flow and t-predictor after Stage0 succeeds.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p \
  exp/20260824_improved_attn_mococo_soft03_flow/stage0_moco \
  exp/20260824_improved_attn_mococo_soft03_flow/stage1_flow_oracle \
  exp/20260824_improved_attn_mococo_soft03_flow/tpred

stage0_job="$(sbatch --parsable script/slurm_train_stage0.sh)"
oracle_job="$(sbatch --parsable --dependency="afterok:${stage0_job}" script/slurm_train_flow_oracle.sh)"
tpred_job="$(sbatch --parsable --dependency="afterok:${stage0_job}" script/slurm_train_tpred.sh)"

printf 'stage0=%s\noracle_flow=%s\ntpredictor=%s\n' "$stage0_job" "$oracle_job" "$tpred_job"

