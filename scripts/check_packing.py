"""Verify TRL's packing does block-diagonal attention (no cross-example leakage).

Approach:
1. Construct two tiny known examples A and B
2. Pack them manually via TRL's data collator
3. Run one forward pass on a small model
4. Check the attention mask / position_ids the model actually receives

We use Qwen3-0.6B (tiny, fast) to avoid downloading Qwen3-8B just for this check.
The packing implementation is model-agnostic.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

TINY = "Qwen/Qwen3-0.6B"


def main():
    tok = AutoTokenizer.from_pretrained(TINY, trust_remote_code=True)

    examples = [
        {"messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]},
        {"messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]},
    ]
    ds = Dataset.from_list(examples)

    cfg = SFTConfig(
        output_dir="/tmp/check_packing",
        max_length=256,
        packing=True,
        assistant_only_loss=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=False,  # fp32 for easier inspection
        model_init_kwargs={
            "dtype": torch.float32,
            "attn_implementation": "eager",  # needed to extract attention weights
            "trust_remote_code": True,
        },
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=TINY,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    # Force-prepare one batch and inspect what goes into the model
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    print("=== batch keys ===")
    print(list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    # Decode sequence and identify example boundaries.
    ids = batch["input_ids"][0]
    text = tok.decode(ids, skip_special_tokens=False)
    print("\n=== decoded sequence (first 2000 chars) ===")
    print(text[:2000])

    # Look for position_ids
    if "position_ids" in batch:
        pids = batch["position_ids"][0]
        print("\n=== position_ids ===")
        print(f"first 30: {pids[:30].tolist()}")
        # Find where positions reset (sign of neat packing)
        resets = [i for i in range(1, len(pids)) if pids[i].item() == 0]
        print(f"position resets at indices: {resets[:5]}")
        if resets:
            print("  GOOD: positions reset across example boundaries (block-diagonal mask implied)")
        else:
            print("  CONCERN: no position reset detected. TRL may be using continuous positions.")

    # Look for attention_mask shape. 2D mask shape would indicate custom mask.
    if "attention_mask" in batch:
        am = batch["attention_mask"][0]
        print(f"\n=== attention_mask shape: {tuple(am.shape)} ===")
        if am.ndim == 1:
            print("  1D mask (just valid/pad). Attention is NOT block-diagonal from mask alone.")
            print("  Model must rely on cu_seqlens or position_ids for cross-example isolation.")
        else:
            print(f"  >=2D mask — block-diagonal pattern should be visible.")

    # Final check: look at labels to confirm assistant-only loss masking
    if "labels" in batch:
        labels = batch["labels"][0]
        unmasked = (labels != -100).sum().item()
        total = labels.numel()
        print(f"\n=== labels ===")
        print(f"  unmasked tokens (training targets): {unmasked}/{total}")
        # Decode just the unmasked tokens to verify they're assistant content
        unmasked_ids = [ids[i].item() for i in range(total) if labels[i] != -100]
        print(f"  unmasked text sample: {tok.decode(unmasked_ids[:50])[:200]!r}")


if __name__ == "__main__":
    main()
