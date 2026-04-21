"""Standalone 3-config × 2-split eval for a saved DualMLPAdapter checkpoint.

Loads base Qwen3-8B, injects adapters, loads the saved adapter state_dict, and
runs eval_three_configs on whatever GPU set the launched accelerate config
specifies (independent of how many GPUs were used during training).

Logs to wandb if WANDB_API_KEY is set. Also prints results to stdout.

Usage:
    accelerate launch --config_file scripts/accelerate_ddp.yaml \\
      scripts/gr/eval_checkpoint.py \\
      --checkpoint /workspace/training/checkpoints/gr/gr_baseline_rr001/adapter_state_dict.pt \\
      --run-name gr_baseline_rr001
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.gr.adapter import inject_adapters
from scripts.gr.loader import build_eval_loaders
from scripts.gr.train_gr import eval_three_configs

BASE_MODEL = "Qwen/Qwen3-8B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to adapter_state_dict.pt")
    p.add_argument("--run-name", type=str, required=True,
                   help="wandb run name (prefixed with 'eval-').")
    p.add_argument("--d-retain", type=int, default=200)
    p.add_argument("--d-forget", type=int, default=200)
    p.add_argument("--layer-start", type=float, default=0.0)
    p.add_argument("--layer-end", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=32768)
    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=os.environ.get("WANDB_PROJECT", "terminal-wrench-sft"),
            config=vars(args),
            init_kwargs={"wandb": {"name": f"eval-{args.run_name}"}},
        )

    accelerator.print(f"loading {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    inject_adapters(
        model, d_retain=args.d_retain, d_forget=args.d_forget,
        layer_start=args.layer_start, layer_end=args.layer_end,
    )
    model = model.to(dtype=torch.bfloat16)

    # Load adapter weights
    ckpt_path = Path(args.checkpoint)
    accelerator.print(f"loading adapter from {ckpt_path}")
    adapter_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(adapter_sd, strict=False)
    # Only adapter keys should load; base keys are "missing" but that's OK (already loaded from pretrained)
    adapter_missing = [k for k in missing if any(t in k for t in ("_retain", "_forget"))]
    if adapter_missing:
        raise RuntimeError(f"Adapter keys missing from state_dict: {adapter_missing[:3]}...")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected[:3]}...")
    accelerator.print(f"loaded {len(adapter_sd)} adapter params from checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    eval_dl_retain, eval_dl_forget = build_eval_loaders(tokenizer, args.max_length)

    model, eval_dl_retain, eval_dl_forget = accelerator.prepare(
        model, eval_dl_retain, eval_dl_forget,
    )

    accelerator.print("running 3-config eval")
    eval_metrics = eval_three_configs(accelerator, model, eval_dl_retain, eval_dl_forget)

    if accelerator.is_main_process:
        print("\n=== Eval results ===")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v:.5f}")
        log_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        accelerator.log(log_metrics, step=0)

    accelerator.end_training()


if __name__ == "__main__":
    main()
