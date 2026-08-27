#!/usr/bin/env bash
# Submit the complete paper-matched Improved Attn pipeline.
# Stage0 -> (Flow MR-jitter || t-predictor) -> (Oracle 5K || t-predictor 5K).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter"
mkdir -p \
  "$ROOT/stage0_moco/checkpoints" \
  "$ROOT/stage1_flow_mrjitter/checkpoints" \
  "$ROOT/tpred/checkpoints" \
  "$ROOT/evaluation_oracle_5k" \
  "$ROOT/evaluation_tpred_5k"

stage0_job="$(sbatch --parsable script/slurm_train_stage0_360only_teacher_decay_mrjitter.sh)"
flow_job="$(sbatch --parsable --dependency="afterok:${stage0_job}" script/slurm_train_flow_360only_mrjitter.sh)"
tpred_job="$(sbatch --parsable --dependency="afterok:${stage0_job}" script/slurm_train_tpred_360only.sh)"
oracle_eval_job="$(sbatch --parsable --dependency="afterok:${flow_job}" script/slurm_eval_oracle_360only_5k.sh)"
tpred_eval_job="$(sbatch --parsable --dependency="afterok:${flow_job}:${tpred_job}" script/slurm_eval_tpred_360only_5k.sh)"

printf 'stage0_360only_teacher_decay=%s\n' "$stage0_job"
printf 'flow_360only_mrjitter=%s (afterok:%s)\n' "$flow_job" "$stage0_job"
printf 'tpredictor_360only=%s (afterok:%s)\n' "$tpred_job" "$stage0_job"
printf 'oracle_eval_5k=%s (afterok:%s)\n' "$oracle_eval_job" "$flow_job"
printf 'tpred_eval_5k=%s (afterok:%s,%s)\n' "$tpred_eval_job" "$flow_job" "$tpred_job"
