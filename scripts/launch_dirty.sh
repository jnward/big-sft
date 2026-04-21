#!/usr/bin/env bash
set -euo pipefail
cd /workspace/training
source .venv/bin/activate
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-terminal-wrench-sft}"
unset WANDB_DISABLED

mkdir -p logs
LOG="logs/train_dirty_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_ddp.yaml \
  scripts/train_lora.py --run-name dirty "$@" \
  2>&1 | tee "$LOG"
