#!/usr/bin/env bash
# 5K four-condition Oracle evaluation for the MR-jitter Flow Stage1.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=imp360_or5k
#SBATCH --output=exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/evaluation_oracle_5k/slurm-%j.out

set -euo pipefail

ROOT="$PWD/exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter"
STAGE0_CKPT="$ROOT/stage0_moco/checkpoints/pn_encoder_best.pt"
FLOW_CKPT="$ROOT/stage1_flow_mrjitter/checkpoints/improved_attn_360only_mrjitter_best.ckpt"
OUT="$ROOT/evaluation_oracle_5k/results.json"
export PYTHONPATH="$PWD:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_360only_oracle_eval_numba_cache_${USER:-user}}"

mkdir -p "$NUMBA_CACHE_DIR" "$(dirname "$OUT")"
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }
[[ -f "$FLOW_CKPT" ]] || { echo "Flow checkpoint not found: $FLOW_CKPT" >&2; exit 1; }

exec /home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  eval_improved_flow.py \
  --config "$PWD/configs/config_flow_360only_mrjitter.yaml" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --flow-ckpt "$FLOW_CKPT" \
  --n 5000 \
  --t-mode oracle \
  --split test \
  --batch-size 16 \
  --out "$OUT"
