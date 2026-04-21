"""LoRA SFT on Terminal Wrench trajectories with TRL SFTTrainer.

Two run modes:
  --run-name clean   : train on clean_pool.jsonl only (baselines + attacker-legit)
  --run-name dirty   : train on clean_pool + subsampled hack_pool (50/50)

Base model: Qwen/Qwen3-8B
LoRA: r=32, alpha=32, dropout=0.05, all-linear target modules
Packing: enabled (TRL's default implementation; we verify block-diagonal attention separately)
Loss: assistant-only (TRL patches Qwen3 chat template automatically)

Recipe derived from s1's sft.sh + TRL conventions + first-principles LR adjustment for LoRA:
  LR=1e-4 cosine, 10% warmup, 3 epochs, global batch 16, bf16, gradient checkpointing,
  cutoff 32k, β1=0.9 β2=0.95, weight decay 1e-4, grad clip 1.0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


DATA_DIR = Path("/workspace/training/data")
CKPT_DIR = Path("/workspace/training/checkpoints")
LOG_DIR = Path("/workspace/training/logs")

BASE_MODEL = "Qwen/Qwen3-8B"
MAX_LEN = 32_768


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_train_dataset(run_name: str, seed: int = 42) -> Dataset:
    clean = load_jsonl(DATA_DIR / "clean_train.jsonl")
    if run_name == "clean":
        examples = clean
    elif run_name == "dirty":
        hack = load_jsonl(DATA_DIR / "hack_train.jsonl")
        rng = random.Random(seed)
        # Subsample hack down to the clean count for a 50/50 mix.
        if len(hack) > len(clean):
            hack = rng.sample(hack, len(clean))
        examples = clean + hack
        rng.shuffle(examples)
    else:
        raise ValueError(f"unknown run_name: {run_name}")
    return Dataset.from_list([{"messages": ex["messages"]} for ex in examples])


def build_eval_datasets() -> dict[str, Dataset]:
    """Return per-split eval datasets so we get eval_<split>_loss metrics."""
    out = {}
    ce = DATA_DIR / "clean_eval.jsonl"
    he = DATA_DIR / "hack_eval.jsonl"
    if ce.exists():
        out["clean"] = Dataset.from_list([{"messages": ex["messages"]} for ex in load_jsonl(ce)])
    if he.exists():
        out["hack"] = Dataset.from_list([{"messages": ex["messages"]} for ex in load_jsonl(he)])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", choices=["clean", "dirty"], required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="override epochs; -1 means use epochs")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    world_size = int(__import__("os").environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, args.global_batch // (args.per_device_batch * world_size))

    ckpt = CKPT_DIR / args.run_name
    ckpt.mkdir(parents=True, exist_ok=True)

    print(f"run={args.run_name} world_size={world_size} grad_accum={grad_accum}")
    train_dataset = build_train_dataset(args.run_name, seed=args.seed)
    eval_datasets = build_eval_datasets()
    print(f"train size: {len(train_dataset)}")
    for k, v in eval_datasets.items():
        print(f"eval[{k}] size: {len(v)}")

    # Tokenizer must be loaded here so we know the chat template patching works.
    # TRL auto-patches Qwen3 for assistant_only_loss=True.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=str(ckpt),
        run_name=f"tw-{args.run_name}-qwen3-8b-lora",

        # data
        max_length=args.max_len,
        packing=True,
        assistant_only_loss=True,

        # schedule
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        seed=args.seed,

        # precision / memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # model init (use flash-attn); note transformers 5.x uses `dtype` not `torch_dtype`
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        },

        # logging / saving / eval
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if eval_datasets else "no",
        eval_on_start=False,
        per_device_eval_batch_size=args.per_device_batch,
        report_to=["wandb"] if __import__("os").environ.get("WANDB_API_KEY") and not __import__("os").environ.get("WANDB_DISABLED") else ["none"],

        # dataset prep
        dataset_num_proc=8,
    )

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=sft_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if eval_datasets else None,
        processing_class=tok,
        peft_config=lora_cfg,
    )

    # Print the chat template patch status before training
    if hasattr(trainer, "processing_class"):
        template = getattr(trainer.processing_class, "chat_template", "")
        print("Chat template contains {% generation %}:", "{% generation %}" in template)

    trainer.train()

    # Save final adapter to a clean subdir for easy rsync to eval box
    final_dir = ckpt / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
