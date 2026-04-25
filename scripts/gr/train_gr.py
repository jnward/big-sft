"""Gradient-routed training via per-example forward+backward (three-way routing).

Based on the canonical three-pass design in
`/workspace/inoculation_prompting_repro/code_rh_and_reddit_toxic/`:

  Pass 1 (forget-classified): both adapters active in forward; update forget adapter.
  Pass 2 (unclassified):      both adapters active in forward; update retain adapter.
  Pass 3 (retain-classified): forget adapter ablated; update retain adapter.

Our structure processes one example per forward (micro_size=1, 32k memory cap),
so each example's backward is a "mini-pass" determined by its classification:

  CLASS_FORGET       → hooks zero retain params; no ablation.
  CLASS_UNCLASSIFIED → hooks zero forget params; no ablation.
  CLASS_RETAIN       → set_scales(1, 0) before forward; hooks zero forget params
                       (redundant but belt-and-suspenders); restore (1, 1) after.

Loss scaling matches canonical's `1/B_full` (= `world_size / step_size_global`
in DDP terms). After DDP averaging, each example contributes `1/B_full` to its
target adapter's gradient. Adapter grad magnitude scales with label fraction.
"""

from __future__ import annotations

# Shim for torch 2.5's missing FSDP2 API that transformers 5.x/HF Trainer calls
# unconditionally during FSDP setup. Adds a no-op register_fsdp_forward_method
# (it would register model.generate with FSDP; we don't use generate during training).
import torch.distributed.fsdp as _fsdp
if not hasattr(_fsdp, "register_fsdp_forward_method"):
    def _noop_register_fsdp_forward_method(model, method_name):
        pass
    _fsdp.register_fsdp_forward_method = _noop_register_fsdp_forward_method

import argparse
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.gr.adapter import (
    inject_adapters, set_scales, param_groups as gr_param_groups,
)
from scripts.gr.loader import (
    build_gr_loader, build_eval_loaders,
    CLASS_UNCLASSIFIED, CLASS_FORGET, CLASS_RETAIN,
)


BASE_MODEL = "Qwen/Qwen3-8B"
CKPT_DIR = Path("/workspace/training/checkpoints/gr")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-retain", type=int, default=200)
    p.add_argument("--d-forget", type=int, default=200)
    p.add_argument("--layer-start", type=float, default=0.0)
    p.add_argument("--layer-end", type=float, default=1.0)
    p.add_argument("--step-size", type=int, default=16,
                   help="Global examples per optim step (across all ranks).")
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="gr")
    p.add_argument("--max-steps", type=int, default=-1,
                   help=">0 caps total optim-step cycles (smoke tests).")
    p.add_argument("--classifier-retain-recall", type=float, default=0.5,
                   help="Fraction of clean examples → CLASS_RETAIN (Pass 3). "
                        "Rest go to CLASS_UNCLASSIFIED (Pass 2).")
    p.add_argument("--classifier-seed", type=int, default=42,
                   help="Seed for the classifier_retain_recall per-example draw.")
    p.add_argument("--only-retain-classified", action="store_true",
                   help="Baseline mode: filter loader to CLASS_RETAIN examples only.")
    p.add_argument("--gradient-ascent", action="store_true",
                   help="Baseline mode: single adapter (forget branch ablated for all forwards); "
                        "negate CLASS_FORGET loss (gradient ascent). forget_opt is never stepped.")
    p.add_argument("--no-evals", action="store_true",
                   help="Skip ALL evals (step-0 and epoch-end). Useful for fast training + "
                        "separate post-hoc eval via eval_checkpoint.py on different GPU count.")
    p.add_argument("--eval-only-at-end", action="store_true",
                   help="Skip per-epoch evals; only run the end-of-training eval.")
    p.add_argument("--retain-from-unlabeled", action="store_true",
                   help="Draw CLASS_RETAIN examples from the ENTIRE unlabeled pool "
                        "(clean + false-negative hacks), not just classifier-confident clean. "
                        "Models the scenario where no labeled-retain set exists.")
    p.add_argument("--model-name", type=str, default=BASE_MODEL,
                   help="HF model id. Default: Qwen/Qwen3-8B.")
    p.add_argument("--match-rslora", action="store_true",
                   help="Use RSLoRA variance-match formula (scale = α * sqrt(1/(f_nonlin*d))) "
                        "instead of the default standard-LoRA formula (includes 1/sqrt(r)).")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="α used by the variance-match formula (reference LoRA's alpha).")
    p.add_argument("--lora-r", type=int, default=16,
                   help="r used by the standard-LoRA variance-match formula (ignored if --match-rslora).")
    p.add_argument("--inject-prompt", type=str, default=None,
                   help="Path to a text file whose contents are prepended to each training "
                        "record's first user message (red-team prompt inoculation).")
    return p.parse_args()


def _zero_grad_hooks(params):
    """Register weight-only grad hooks that zero the incoming gradient for each param.
    Does NOT mask pass-through gradient flow through the param's module.
    """
    return [p.register_hook(lambda g: torch.zeros_like(g)) for p in params]


def _remove_hooks(hooks):
    for h in hooks:
        h.remove()


def _save_adapter(accelerator, model, save_path: Path) -> None:
    """Extract adapter params and save on rank 0.

    Under FSDP, `accelerator.get_state_dict` gathers the full, unsharded state dict
    on rank 0 (empty on other ranks). Under DDP this is a plain state_dict copy.
    The returned keys may carry `_fsdp_wrapped_module.` prefix from FSDP's
    Qwen3DecoderLayer wrapping — strip it so the saved state dict matches the
    original adapter module structure.
    """
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_sd = {}
        for n, p in state_dict.items():
            if "_retain" not in n and "_forget" not in n:
                continue
            clean = n.replace("_fsdp_wrapped_module.", "").replace("module.", "", 1) if n.startswith("module.") else n.replace("_fsdp_wrapped_module.", "")
            adapter_sd[clean] = p.detach().cpu()
        torch.save(adapter_sd, save_path)
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    step_size_local = max(1, args.step_size // world_size)
    if step_size_local * world_size != args.step_size:
        print(f"WARN: step_size {args.step_size} not divisible by world_size {world_size};"
              f" using step_size_local={step_size_local}")

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=step_size_local,
        log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
    )

    if accelerator.is_main_process:
        model_short = args.model_name.split("/")[-1].lower()
        accelerator.init_trackers(
            project_name=os.environ.get("WANDB_PROJECT", "terminal-wrench-sft"),
            config=vars(args),
            init_kwargs={"wandb": {"name": f"gr-{args.run_name}-{model_short}"}},
        )

    # ---- model + adapters ----
    accelerator.print(f"loading {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    inject_adapters(
        model, d_retain=args.d_retain, d_forget=args.d_forget,
        layer_start=args.layer_start, layer_end=args.layer_end,
        lora_alpha=args.lora_alpha, lora_r=args.lora_r,
        match_rslora=args.match_rslora,
    )
    model = model.to(dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Print adapter variance scale for visibility (same on all adapters).
    from scripts.gr.adapter import DualMLPAdapter
    first_adapter = next(m for m in model.modules() if isinstance(m, DualMLPAdapter))
    accelerator.print(
        f"trainable params: {n_train/1e6:.1f}M  "
        f"adapter scale: retain={first_adapter.scale_retain:.4f}, forget={first_adapter.scale_forget:.4f}  "
        f"match_rslora={args.match_rslora}"
    )

    # ---- data ----
    inject_prompt = None
    if args.inject_prompt:
        inject_prompt = Path(args.inject_prompt).read_text().rstrip("\n")
        accelerator.print(f"injecting prompt from {args.inject_prompt} ({len(inject_prompt)} chars) into first user message")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    bundle = build_gr_loader(
        tokenizer,
        max_length=args.max_length,
        num_workers=0,
        classifier_retain_recall=args.classifier_retain_recall,
        classifier_seed=args.classifier_seed,
        only_retain_classified=args.only_retain_classified,
        retain_from_unlabeled=args.retain_from_unlabeled,
        inject_prompt=inject_prompt,
    )
    eval_dl_retain, eval_dl_forget = build_eval_loaders(tokenizer, args.max_length)
    accelerator.print(
        f"train examples: {bundle.n_train_examples} "
        f"(forget={bundle.n_forget_examples}, unclassified={bundle.n_unclassified_examples}, "
        f"retain={bundle.n_retain_examples}) "
        f"[classifier_retain_recall={args.classifier_retain_recall}]"
    )

    total_optim_steps_per_epoch = bundle.n_train_examples // args.step_size
    total_optim_steps = args.epochs * total_optim_steps_per_epoch
    if args.max_steps > 0:
        total_optim_steps = min(total_optim_steps, args.max_steps)
    warmup_steps = max(1, int(args.warmup_ratio * total_optim_steps))
    accelerator.print(
        f"world_size={world_size} step_size_global={args.step_size} "
        f"step_size_local={step_size_local} "
        f"steps/epoch={total_optim_steps_per_epoch} "
        f"total_optim_steps={total_optim_steps} warmup={warmup_steps}"
    )

    # ---- optimizers ----
    retain_params, forget_params = gr_param_groups(model)
    retain_opt = torch.optim.AdamW(retain_params, lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=args.weight_decay)
    forget_opt = torch.optim.AdamW(forget_params, lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=args.weight_decay)

    model, retain_opt, forget_opt, bundle.train_dataloader = accelerator.prepare(
        model, retain_opt, forget_opt, bundle.train_dataloader
    )
    eval_dl_retain, eval_dl_forget = accelerator.prepare(eval_dl_retain, eval_dl_forget)

    # Re-grab param lists from the wrapped model so hooks apply to the right tensors
    retain_params, forget_params = gr_param_groups(accelerator.unwrap_model(model))

    def current_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    cur_run_path = CKPT_DIR / args.run_name
    cur_run_path.mkdir(parents=True, exist_ok=True)

    # Fixed loss scale (canonical's 1/B_full → world_size/step_size_global per example,
    # so DDP-averaged gradient equals 1/step_size_global × Σ per-example grads).
    loss_scale = world_size / args.step_size

    # ---- Eval at step 0 (baseline before any training) ----
    # Step-0 eval is the base Qwen3-8B on held-out splits (adapters at init have
    # down_retain=down_forget=0, so they contribute 0 in any forward). Cache to disk;
    # future runs skip the computation and load the cached values.
    if not args.no_evals:
        step0_cache_path = Path("/workspace/training/data/step0_eval_cache.json")
        eval_metrics_step0 = None
        if step0_cache_path.exists():
            try:
                import json as _json
                with step0_cache_path.open() as f:
                    eval_metrics_step0 = _json.load(f)
                accelerator.print(f"loaded step-0 eval from cache ({step0_cache_path})")
            except Exception as e:
                accelerator.print(f"step-0 cache read failed ({e}); recomputing")
                eval_metrics_step0 = None

        if eval_metrics_step0 is None:
            accelerator.print("running step-0 eval (baseline)")
            eval_metrics_step0 = eval_three_configs(accelerator, model, eval_dl_retain, eval_dl_forget)
            if accelerator.is_main_process:
                import json as _json
                with step0_cache_path.open("w") as f:
                    _json.dump(eval_metrics_step0, f, indent=2)
                accelerator.print(f"cached step-0 eval to {step0_cache_path}")

        if accelerator.is_main_process:
            log_metrics = {f"eval/{k}": v for k, v in eval_metrics_step0.items()}
            log_metrics["train/epoch"] = 0.0
            accelerator.log(log_metrics, step=0)
    else:
        accelerator.print("skipping step-0 eval (--no-evals)")

    # ---- Training loop ----
    accelerator.print("starting training")
    optim_step = 0
    done = False

    for epoch in range(args.epochs):
        if done:
            break
        model.train()
        retain_opt.zero_grad(set_to_none=True)
        forget_opt.zero_grad(set_to_none=True)

        buffer: list[dict] = []

        for batch in bundle.train_dataloader:
            buffer.append(batch)
            if len(buffer) < step_size_local:
                continue

            # ---- One optimizer-step cycle on buffered step_size_local examples ----
            local_classes = [int(b["classification"].item()) for b in buffer]
            n_forget_local = sum(1 for c in local_classes if c == CLASS_FORGET)
            n_unclassified_local = sum(1 for c in local_classes if c == CLASS_UNCLASSIFIED)
            n_retain_local = sum(1 for c in local_classes if c == CLASS_RETAIN)

            counts_local = torch.tensor(
                [n_forget_local, n_unclassified_local, n_retain_local],
                device=accelerator.device, dtype=torch.long,
            )
            counts_global = accelerator.reduce(counts_local, reduction="sum")
            n_forget_global = int(counts_global[0].item())
            n_unclassified_global = int(counts_global[1].item())
            n_retain_global = int(counts_global[2].item())

            # Set LR for this optim step
            lr = current_lr(optim_step)
            for g in retain_opt.param_groups: g["lr"] = lr
            for g in forget_opt.param_groups: g["lr"] = lr

            # Per-example backward with three-way routing
            losses_by_class: dict[int, list[float]] = {
                CLASS_FORGET: [], CLASS_UNCLASSIFIED: [], CLASS_RETAIN: [],
            }

            unwrapped_model = accelerator.unwrap_model(model)

            for b in buffer:
                cls = int(b["classification"].item())
                with accelerator.accumulate(model):
                    if args.gradient_ascent:
                        # Single-adapter baseline: forget branch ablated for all forwards
                        # (chain-rule through forget_scale=0 kills any gradient into forget
                        # params), negate CLASS_FORGET loss for gradient ascent, no hooks.
                        set_scales(unwrapped_model, retain_scale=1.0, forget_scale=0.0)
                        sign = -1.0 if cls == CLASS_FORGET else 1.0
                        out = model(
                            input_ids=b["input_ids"],
                            position_ids=b.get("position_ids"),
                            attention_mask=b.get("attention_mask"),
                            labels=b["labels"],
                        )
                        loss = out.loss * loss_scale * sign
                        accelerator.backward(loss)
                        losses_by_class[cls].append(out.loss.detach().float().item())
                    else:
                        ablated = False
                        if cls == CLASS_FORGET:
                            wrong_params = retain_params
                        elif cls == CLASS_UNCLASSIFIED:
                            wrong_params = forget_params
                        elif cls == CLASS_RETAIN:
                            wrong_params = forget_params
                            set_scales(unwrapped_model, retain_scale=1.0, forget_scale=0.0)
                            ablated = True
                        else:
                            raise ValueError(f"Unknown classification: {cls}")

                        hooks = _zero_grad_hooks(wrong_params)
                        try:
                            out = model(
                                input_ids=b["input_ids"],
                                position_ids=b.get("position_ids"),
                                attention_mask=b.get("attention_mask"),
                                labels=b["labels"],
                            )
                            loss = out.loss * loss_scale
                            accelerator.backward(loss)
                            losses_by_class[cls].append(out.loss.detach().float().item())
                        finally:
                            _remove_hooks(hooks)
                            if ablated:
                                set_scales(unwrapped_model, retain_scale=1.0, forget_scale=1.0)

            # Step optimizers (gated) and zero grads.
            # Gradient-ascent baseline: only retain_opt steps; forget adapter stays at init.
            # Three-way routing: retain_opt steps when any unc+retain examples globally;
            # forget_opt steps when any forget examples globally.
            if args.gradient_ascent:
                if (n_forget_global + n_unclassified_global + n_retain_global) > 0:
                    retain_opt.step()
            else:
                if n_forget_global > 0:
                    forget_opt.step()
                if (n_unclassified_global + n_retain_global) > 0:
                    retain_opt.step()
            retain_opt.zero_grad(set_to_none=True)
            forget_opt.zero_grad(set_to_none=True)

            # Log
            log = {
                "train/lr": lr,
                "train/n_forget_global": n_forget_global,
                "train/n_unclassified_global": n_unclassified_global,
                "train/n_retain_global": n_retain_global,
                "train/n_forget_local": n_forget_local,
                "train/n_unclassified_local": n_unclassified_local,
                "train/n_retain_local": n_retain_local,
                "train/epoch": epoch + (optim_step + 1) / max(1, total_optim_steps_per_epoch),
            }
            for cls, name in [(CLASS_FORGET, "forget"), (CLASS_UNCLASSIFIED, "unclassified"),
                              (CLASS_RETAIN, "retain")]:
                ls = losses_by_class[cls]
                if ls:
                    log[f"train/{name}/loss_mean_local"] = sum(ls) / len(ls)
            accelerator.log(log, step=optim_step)

            buffer = []
            optim_step += 1

            if args.max_steps > 0 and optim_step >= args.max_steps:
                accelerator.print(f"hit max_steps={args.max_steps}")
                done = True
                break

        # End-of-epoch eval
        is_last_epoch = (epoch == args.epochs - 1)
        skip_this_eval = args.no_evals or (args.eval_only_at_end and not is_last_epoch)
        if not done and not skip_this_eval:
            accelerator.print(f"epoch {epoch+1} complete; running 3-config eval")
            eval_metrics = eval_three_configs(accelerator, model, eval_dl_retain, eval_dl_forget)
            if accelerator.is_main_process:
                log_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                log_metrics["train/epoch"] = epoch + 1
                accelerator.log(log_metrics, step=optim_step)
        elif not done:
            accelerator.print(f"epoch {epoch+1} complete; skipping eval (per flags)")

        # End-of-epoch checkpoint save (keep all epochs to avoid losing best-eval to rotation)
        if not done:
            ep_ckpt_path = cur_run_path / f"checkpoint-epoch-{epoch+1}"
            _save_adapter(accelerator, model, ep_ckpt_path / "adapter_state_dict.pt")
            accelerator.print(f"saved epoch-{epoch+1} checkpoint to {ep_ckpt_path}")

    # ---- save adapter ----
    _save_adapter(accelerator, model, cur_run_path / "adapter_state_dict.pt")
    accelerator.print(f"saved adapter to {cur_run_path}")

    accelerator.end_training()


@torch.no_grad()
def eval_three_configs(accelerator, model, eval_dl_retain, eval_dl_forget):
    """3-config × 2-split held-out eval. Aggregates (sum reduce) total_loss and
    total_tokens across ranks so the reported number is a full-dataset token-weighted
    mean CE, independent of world_size.
    """
    model.eval()
    configs = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
    results = {}
    for cfg_name, rs, fs in configs:
        set_scales(accelerator.unwrap_model(model), rs, fs)
        for split_name, dl in [("retain", eval_dl_retain), ("forget", eval_dl_forget)]:
            total_loss = torch.zeros((), dtype=torch.float64, device=accelerator.device)
            total_tokens = torch.zeros((), dtype=torch.float64, device=accelerator.device)
            for batch in dl:
                out = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                n = (batch["labels"] != -100).sum().to(torch.float64)
                total_loss += out.loss.to(torch.float64) * n
                total_tokens += n
            # Sum-reduce across ranks → full-dataset token-weighted mean.
            total_loss = accelerator.reduce(total_loss, reduction="sum")
            total_tokens = accelerator.reduce(total_tokens, reduction="sum")
            denom = torch.clamp(total_tokens, min=1.0)
            results[f"{cfg_name}/{split_name}/loss"] = (total_loss / denom).item()
    set_scales(accelerator.unwrap_model(model), 1.0, 1.0)
    model.train()
    return results


if __name__ == "__main__":
    main()
