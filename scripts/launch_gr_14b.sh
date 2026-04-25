#!/usr/bin/env bash
# Qwen3-14B DualMLPAdapter gradient-routing training.
# Matches our Qwen3-14B LoRA r=16 baseline config but uses MLP adapters with
# RSLoRA-matched variance + three-pass routing.
#
# Key settings:
#   - DualMLPAdapter d=64 (78.6M trainable, 22% > LoRA r=16's 64M)
#   - match_rslora=True → scale = α * sqrt(1/(f_nonlin * d)) ≈ 5.657 (α=32, d=64, f=0.5)
#   - LR=1e-3 cosine, warmup=0.05, 5 epochs
#   - max_len=32768, global batch 8 (step_size=8), bf16, DDP 8 GPUs
#   - retain_from_unlabeled, classifier_retain_recall=0.01 (Pass 3 = 1% of unlabeled)
#   - classifier_forget_recall comes from data prep (re-prep between runs)
#
# Pass --run-name and other overrides as CLI args to this script.

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
LOG="logs/train_gr_14b_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_ddp.yaml \
  scripts/gr/train_gr.py \
    --model-name Qwen/Qwen3-14B \
    --d-retain 64 --d-forget 64 \
    --match-rslora \
    --lora-alpha 32 --lora-r 16 \
    --lr 1e-3 \
    --epochs 5 \
    --warmup-ratio 0.05 \
    --max-length 32768 \
    --step-size 8 \
    --classifier-retain-recall 0.01 \
    --retain-from-unlabeled \
    "$@" \
  2>&1 | tee "$LOG"
