#!/usr/bin/env bash
set -euo pipefail
cd /workspace/training
source .venv/bin/activate
# Load .env (HF_HOME, HF_TOKEN, WANDB_API_KEY)
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-terminal-wrench-sft}"
export PYTHONPATH="/workspace/training:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_DISABLED

mkdir -p logs
LOG="logs/train_gr4_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_ddp4.yaml \
  scripts/gr/train_gr.py "$@" \
  2>&1 | tee "$LOG"
