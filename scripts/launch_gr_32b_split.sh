#!/usr/bin/env bash
# Qwen3-32B GRAFT re-run with the UPDATED method: classic routing (unchanged from
# the shipped paper arm) + κ=2 pressure compensation + split-moment AdamW.
#
# Recipe reproduces the paper arm gr_32b_mlp_fr02_ddp_s1like_unc_both_ru001
# byte-for-byte (DDP, lr 5e-4, max-length 16384 + filter-overlong, step-size 48,
# 5 epochs — verified against its wandb config), differing ONLY in
# --split-moment --kappa 2. Train the full 5 epochs; the paper arm evaluates ep5.
#
# Health check: wandb gr/forget_realized_step should read ≈2.0 on
# hack-containing windows from the first steps. gr/c_forget ≈ 1.01.
#
# Pass any overrides as CLI args.

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
LOG="logs/train_gr_32b_split_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_ddp.yaml \
  scripts/gr/train_gr.py \
    --model-name Qwen/Qwen3-32B \
    --d-retain 64 --d-forget 64 \
    --match-rslora \
    --lora-alpha 32 --lora-r 16 \
    --lr 5e-4 \
    --epochs 5 \
    --warmup-ratio 0.05 \
    --max-length 16384 \
    --filter-overlong \
    --step-size 48 \
    --classifier-retain-recall 0.01 \
    --retain-from-unlabeled \
    --unclassified-trains-both \
    --split-moment \
    --kappa 2 \
    --run-name gr_32b_mlp_fr02_ddp_s1like_unc_both_split_ru001 \
    "$@" \
  2>&1 | tee "$LOG"
