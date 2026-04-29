#!/usr/bin/env bash
# Qwen3-14B LoRA r=16 baseline on clean Terminal Wrench trajectories.
# Matches s1_peft's LoRA config (scale-dependent-elicitation/s1_peft/train/sft_lora.sh),
# with user-specified deviations noted in the plan.
#
# Key settings (s1_peft LoRA):
#   rank=16, alpha=32, use_rslora=True (effective scale = alpha/sqrt(r) = 8.0)
#   lora_dropout=0.0, target_modules = 7 projections (q/k/v/o + gate/up/down)
#   LR=1e-3, cosine; epochs=5; warmup_ratio=0.05; wd=1e-4; β=(0.9, 0.95);
#   bf16; packing on (our choice — matches our full-FT; s1_peft doesn't pack).
#
# Ours vs s1_peft:
#   - Qwen/Qwen3-14B (not Qwen2.5-*-Instruct)
#   - max_len=32768 (s1_peft uses 20000)
#   - DDP, not FSDP (LoRA fits on single GPU; FSDP would add overhead without memory benefit)
#   - global batch 8 (matches our full-FT gb=8; s1_peft uses 16)
#   - eval + save per-epoch (s1_peft uses eval_strategy=no, save_strategy=no)
#
# Data: clean_train.jsonl filtered to meta.classification == "baseline"
# (1599 examples; drops 823 attacker_legitimate_solve).

set -euo pipefail
cd /workspace/training
source .venv/bin/activate
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
# Model is cached locally; force offline to avoid rank race (seen on full-FT)
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-terminal-wrench-sft}"
export PYTHONPATH="/workspace/training:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_DISABLED

mkdir -p logs
LOG="logs/train_lora_14b_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_ddp.yaml \
  scripts/train_lora.py \
    --run-name clean_14b_lora_r16 \
    --model-name Qwen/Qwen3-14B \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.0 \
    --use-rslora \
    --target-modules s1-peft-7 \
    --lr 1e-3 \
    --epochs 5 \
    --warmup-ratio 0.05 \
    --max-len 32768 \
    --global-batch 8 --per-device-batch 1 \
    --exclude-attacker-legit \
    "$@" \
  2>&1 | tee "$LOG"
