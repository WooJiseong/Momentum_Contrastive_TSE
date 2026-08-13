#!/usr/bin/env bash

# Purpose:
#   Submit the 5000-item PN_MOCOCO TFGridNet evaluation to Slurm, or run it on
#   the current compute node when this host is not gate* and CUDA is visible.
#
# Run from:
#   Running_Lab/PN_MOCOCO
#
# Single GPU:
#   bash script/submit_eval_slurm.sh
#
# Multi GPU:
#   Not supported. Evaluation runs on one GPU.
#
# Arguments:
#   None. Override MOCO_CFG, TFGRID_CFG, SLURM_PARTITION, SLURM_TIME, SLURM_MEM, or SLURM_CPUS_PER_TASK through environment variables.
#
# Output:
#   exp/YYYYMMDD_pn_mococo_eval[_runNN]/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm batch example:
#   SLURM_PARTITION=gpu6 bash script/submit_eval_slurm.sh
#   FORCE_SBATCH=1 SLURM_PARTITION=gpu6 bash script/submit_eval_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PY="${PY:-python}"
MOCO_SOURCE_CFG="${MOCO_CFG:-configs/config_moco_encoder.yaml}"
TFGRID_SOURCE_CFG="${TFGRID_CFG:-configs/config_tfgridnet_supervised.yaml}"
PARTITION="${SLURM_PARTITION:-gpu6}"
TIME_LIMIT="${SLURM_TIME:-24:00:00}"
MEMORY="${SLURM_MEM:-48G}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
JOB_NAME="${EVAL_JOB_NAME:-pn_mococo_eval}"

eval "$("$PY" script/prepare_runtime_config.py --moco-config "$MOCO_SOURCE_CFG" --tfgrid-config "$TFGRID_SOURCE_CFG" --mode eval)"
mkdir -p "$RUN_DIR"

HOST_NAME="$(hostname -s 2>/dev/null || hostname)"
if [[ "${FORCE_SBATCH:-0}" != "1" && "$HOST_NAME" != gate* ]]; then
  if "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[PN_MOCOCO submit_eval] host=$HOST_NAME is not gate*, using the current compute node."
    echo "[PN_MOCOCO submit_eval] run_dir=$RUN_DIR"
    export MOCO_CFG TFGRID_CFG RUN_DIR PREPARED_RUN=1 PY
    bash script/train_eval.sh eval 2>&1 | tee "$RUN_DIR/direct_eval.log"
    exit 0
  fi
  echo "[PN_MOCOCO submit_eval] host=$HOST_NAME is not gate*, but CUDA is not visible; submitting through Slurm."
fi

JOB_SCRIPT="$RUN_DIR/slurm_eval.sh"
cat > "$JOB_SCRIPT" <<SH
#!/usr/bin/env bash
#SBATCH -p $PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --time=$TIME_LIMIT
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$RUN_DIR/slurm-%j.out

set -euo pipefail

cd "$PROJECT_DIR"
export MOCO_CFG="$MOCO_CFG"
export TFGRID_CFG="$TFGRID_CFG"
export RUN_DIR="$RUN_DIR"
export PREPARED_RUN=1
export PY="${PY}"

bash script/train_eval.sh eval
SH

echo "[PN_MOCOCO submit_eval] run_dir=$RUN_DIR"
echo "[PN_MOCOCO submit_eval] runtime_moco_cfg=$MOCO_CFG"
echo "[PN_MOCOCO submit_eval] runtime_tfgrid_cfg=$TFGRID_CFG"
echo "[PN_MOCOCO submit_eval] job_script=$JOB_SCRIPT"
sbatch "$JOB_SCRIPT" | tee "$RUN_DIR/sbatch_submit.txt"
