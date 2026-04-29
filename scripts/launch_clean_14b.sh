#!/usr/bin/env bash
# Clean Qwen3-14B LoRA baseline, matching the s1_peft LoRA training config as
# closely as possible (keeping max_length=32768 per user preference).
#
# s1_peft config reference: /workspace/scale-dependent-elicitation/s1_peft/train/sft_lora.{sh,py}
#   - LR=1e-3, epochs=5, warmup_ratio=0.05, block_size=20000 (we use 32768)
#   - r=16 default (we use r=32 per user), alpha=32, use_rslora=True
#   - lora_dropout=0.0, target_modules = {q, k, v, o, gate, up, down}_proj
#   - eval_strategy="no", save_strategy="no" (model saved only at end)
#
# Data: clean_train.jsonl filtered to meta.classification == "baseline"
# (drops attacker_legitimate_solve rollouts per user request).

set -euo pipefail
cd /workspace/training
source .venv/bin/activate
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-terminal-wrench-sft}"
export PYTHONPATH="/workspace/training:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_DISABLED

mkdir -p logs
LOG="logs/train_clean_14b_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_ddp.yaml \
  scripts/train_lora.py \
    --run-name clean_14b_s1peft \
    --model-name Qwen/Qwen3-14B \
    --lora-r 32 --lora-alpha 32 --lora-dropout 0.0 \
    --use-rslora \
    --target-modules s1-peft-7 \
    --lr 1e-3 \
    --epochs 5 \
    --warmup-ratio 0.05 \
    --exclude-attacker-legit \
    --save-only-at-end \
    "$@" \
  2>&1 | tee "$LOG"
