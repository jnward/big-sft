#!/usr/bin/env bash
# Qwen3-14B full-finetune baseline on clean Terminal Wrench trajectories.
# Config matches s1's /workspace/s1/train/sft.sh verbatim except for the model.
#
# Key settings (s1 full-FT):
#   LR=1e-5, cosine; epochs=5; warmup_ratio=0.05; wd=1e-4;
#   β=(0.9, 0.95); max_len=32768; bf16; FSDP full_shard (Qwen3DecoderLayer wrap);
#   eval_strategy=no, save_strategy=no; save once at end.
#
# Our deviations:
#   - Qwen3-14B (not Qwen2.5-32B-Instruct)
#   - packing on (block-diagonal attention makes per-example grads equivalent;
#     ~few× throughput vs s1's no-packing)
#   - 8 GPUs × per_device=1 × grad_accum=2 = global batch 16 (s1's 16 × 1 × 1 = 16)
#   - TRL assistant_only_loss (equivalent to s1's DataCollatorForCompletionOnlyLM)
#
# Data: clean_train.jsonl filtered to meta.classification == "baseline"
# (1599 examples; drops 823 attacker_legitimate_solve).

set -euo pipefail
cd /workspace/training
source .venv/bin/activate
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
# Model is cached locally; force offline lookup to avoid HF Hub race across ranks
# during model-shard resolution (some ranks intermittently fail to find the cached files).
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-terminal-wrench-sft}"
export PYTHONPATH="/workspace/training:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_DISABLED

mkdir -p logs
LOG="logs/train_fullft_14b_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

accelerate launch --config_file scripts/accelerate_fsdp.yaml \
  scripts/train_lora.py \
    --run-name clean_14b_fullft \
    --model-name Qwen/Qwen3-14B \
    --full-ft \
    --lr 1e-5 \
    --epochs 5 \
    --warmup-ratio 0.05 \
    --max-len 32768 \
    --global-batch 16 --per-device-batch 1 \
    --exclude-attacker-legit \
    "$@" \
  2>&1 | tee "$LOG"
