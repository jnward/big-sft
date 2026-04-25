#!/usr/bin/env bash
# Qwen3-32B DualMLPAdapter gradient-routing training (FSDP, 5 fr variants).
# Mirrors launch_gr_14b.sh but uses FSDP full_shard (base weights too large for DDP
# on 8×80 GB) and the 32B model. Same d=64 adapter and RSLoRA variance-match as 14B.
#
# Key differences vs 14B launcher:
#   - accelerate_fsdp.yaml instead of accelerate_ddp.yaml
#   - --model-name Qwen/Qwen3-32B
#   - Default LR=3e-4 (matches the stable config from the 14B sweep)
#
# Intended use: LR schedule is configured for 5 epochs; we typically manually stop
# after `saved epoch-2 checkpoint` appears in the log. This keeps the LR profile
# matched to the 14B sweep while saving compute.
#
# Pass --run-name and any overrides as CLI args.

set -euo pipefail
cd /workspace/training
source .venv/bin/activate
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-terminal-wrench-sft}"
export PYTHONPATH="/workspace/training:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_DISABLED

mkdir -p logs
LOG="logs/train_gr_32b_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_fsdp.yaml \
  scripts/gr/train_gr.py \
    --model-name Qwen/Qwen3-32B \
    --d-retain 64 --d-forget 64 \
    --match-rslora \
    --lora-alpha 32 --lora-r 16 \
    --lr 3e-4 \
    --epochs 5 \
    --warmup-ratio 0.05 \
    --max-length 32768 \
    --step-size 8 \
    --classifier-retain-recall 0.01 \
    --retain-from-unlabeled \
    "$@" \
  2>&1 | tee "$LOG"
