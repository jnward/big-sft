"""Mixed dataloader for three-way gradient-routed SFT.

Emits one example per batch with a `classification` column ∈ {0, 1, 2}:
  CLASS_UNCLASSIFIED (0): Pass 2 — both adapters active in forward, retain adapter updates.
  CLASS_FORGET       (1): Pass 1 — both adapters active, forget adapter updates.
  CLASS_RETAIN       (2): Pass 3 — forget adapter ablated in forward, retain adapter updates.

Classification assignment (training-time, seeded):
  - `gr_train_forget.jsonl` records → CLASS_FORGET (always).
  - `gr_train_retain.jsonl` records are mixed (clean + false-negative hacks):
      * clean (meta.classification == "baseline"): draw ~ U(0, 1); CLASS_RETAIN if draw <
        `classifier_retain_recall`, else CLASS_UNCLASSIFIED.
      * false-negative hacks (any other meta.classification value): always CLASS_UNCLASSIFIED.

The forget-side recall knob is applied upstream in data_prep_gr.py (it governs which
hacks get written to gr_train_forget.jsonl vs. to gr_train_retain.jsonl as false
negatives). The retain-side recall knob is applied here, at loader construction.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader

from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

DATA_DIR = Path("/workspace/training/data")

CLASS_UNCLASSIFIED = 0
CLASS_FORGET = 1
CLASS_RETAIN = 2
CLASS_FORGET_ONLY = 3  # Pass-4 symmetric ablation: forget-only forward (retain branch ablated)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _patch_chat_template_for_assistant_mask(tokenizer) -> None:
    if "{% generation %}" in (tokenizer.chat_template or ""):
        return
    from trl.chat_template_utils import get_training_chat_template
    new_template = get_training_chat_template(tokenizer)
    if new_template is None:
        raise RuntimeError("Could not patch Qwen3 chat template")
    tokenizer.chat_template = new_template


def _tokenize_with_mask(example: dict, tokenizer, max_length: int, filter_overlong: bool = False) -> dict:
    processed = tokenizer.apply_chat_template(
        example["messages"],
        return_assistant_tokens_mask=True,
        return_dict=True,
        tokenize=True,
    )
    # When filter_overlong=True, drop records longer than max_length (return empty mask).
    # The downstream `any(assistant_masks)` filter removes them. Otherwise truncate.
    if filter_overlong and len(processed["input_ids"]) > max_length:
        return {"input_ids": [], "assistant_masks": []}
    return {
        "input_ids": processed["input_ids"][:max_length],
        "assistant_masks": processed["assistant_masks"][:max_length],
    }


def _classify_retain_record(
    meta_classification: str,
    draw: float,
    retain_recall: float,
    retain_from_unlabeled: bool = False,
) -> int:
    """Assign a class to a retain-pool record.

    Default (retain_from_unlabeled=False, oracle retain-classifier with FPR=0):
    - "baseline" (clean) → CLASS_RETAIN with prob `retain_recall`, else CLASS_UNCLASSIFIED.
    - Any other value (false-negative hack) → CLASS_UNCLASSIFIED.

    retain_from_unlabeled=True (no labeled-retain set; sample uniformly from unlabeled):
    - Every record → CLASS_RETAIN with prob `retain_recall`, else CLASS_UNCLASSIFIED.
      This means Pass 3 sees a mix of clean and false-negative hacks, proportional
      to their prevalence in the unlabeled pool.
    """
    if retain_from_unlabeled:
        return CLASS_RETAIN if draw < retain_recall else CLASS_UNCLASSIFIED
    if meta_classification == "baseline":
        return CLASS_RETAIN if draw < retain_recall else CLASS_UNCLASSIFIED
    return CLASS_UNCLASSIFIED


def _prep_with_classification(
    jsonl_path: Path,
    classification_fn,
    tokenizer,
    max_length: int,
    desc_suffix: str = "",
    inject_prompt: str | None = None,
    filter_overlong: bool = False,
) -> Dataset:
    """Tokenize a jsonl file and attach per-record `classification` column.

    `classification_fn(record) -> int` is applied to each loaded record
    BEFORE tokenization (so we can use `meta` fields to decide class).

    If `inject_prompt` is given, it is prepended (with a trailing blank line) to
    the first user message of every record — landing in non-loss-bearing tokens
    under assistant_only_loss, suitable for red-team-prompt inoculation.
    """
    records = _load_jsonl(jsonl_path)
    if not records:
        return Dataset.from_dict({"input_ids": [], "assistant_masks": [], "classification": []})
    classifications = [classification_fn(r) for r in records]

    def _messages_for(r):
        msgs = r["messages"]
        if inject_prompt and msgs and msgs[0]["role"] == "user":
            msgs = [{"role": "user", "content": inject_prompt + "\n\n" + msgs[0]["content"]}, *msgs[1:]]
        return msgs

    ds = Dataset.from_list([{"messages": _messages_for(r)} for r in records])
    ds = ds.map(
        _tokenize_with_mask,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "filter_overlong": filter_overlong},
        num_proc=8,
        remove_columns=["messages"],
        desc=f"Tokenizing {jsonl_path.name}{desc_suffix}",
    )
    ds = ds.add_column("classification", classifications)
    ds = ds.filter(lambda ex: any(ex["assistant_masks"]), num_proc=8)
    return ds


def build_collator(tokenizer):
    """Padding-free collator that carries `classification` as a side-car tensor."""
    base = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=False,
        padding_free=True,
    )

    def collate(examples):
        classifications = [ex.pop("classification") for ex in examples]
        batch = base(examples)
        batch["classification"] = torch.tensor(classifications, dtype=torch.long)
        return batch

    return collate


@dataclass
class GRLoaderBundle:
    train_dataloader: DataLoader
    collator: callable  # noqa
    n_train_examples: int
    n_unclassified_examples: int
    n_forget_examples: int
    n_retain_examples: int


def build_gr_loader(
    tokenizer,
    max_length: int = 32_768,
    num_workers: int = 0,
    classifier_retain_recall: float = 0.5,
    classifier_seed: int = 42,
    only_retain_classified: bool = False,
    retain_from_unlabeled: bool = False,
    inject_prompt: str | None = None,
    filter_overlong: bool = False,
    forget_only_prob: float = 0.0,
) -> GRLoaderBundle:
    _patch_chat_template_for_assistant_mask(tokenizer)

    # Forget pool: stochastic CLASS_FORGET vs CLASS_FORGET_ONLY per record (seeded).
    # CLASS_FORGET_ONLY (Pass 4) ablates the retain branch in forward; symmetric to
    # CLASS_RETAIN's ablation of the forget branch.
    forget_rng = random.Random(classifier_seed + 1)
    forget_pool_records = _load_jsonl(DATA_DIR / "gr_train_forget.jsonl")
    forget_draws = [forget_rng.random() for _ in forget_pool_records]
    forget_idx = {"i": 0}

    def forget_classification_fn(record):
        i = forget_idx["i"]
        forget_idx["i"] += 1
        return CLASS_FORGET_ONLY if forget_draws[i] < forget_only_prob else CLASS_FORGET

    ds_forget = _prep_with_classification(
        DATA_DIR / "gr_train_forget.jsonl",
        classification_fn=forget_classification_fn,
        tokenizer=tokenizer, max_length=max_length,
        inject_prompt=inject_prompt,
        filter_overlong=filter_overlong,
    )

    # Retain pool: stochastic CLASS_RETAIN vs CLASS_UNCLASSIFIED per record
    # (seeded so reruns are reproducible).
    rng = random.Random(classifier_seed)
    retain_pool_records = _load_jsonl(DATA_DIR / "gr_train_retain.jsonl")
    retain_draws = [rng.random() for _ in retain_pool_records]
    # Build the classification function closure: uses index into the retain-pool records.
    # _prep_with_classification iterates records in order, so we zip draws with records.
    retain_records_and_draws = list(zip(retain_pool_records, retain_draws))
    idx_ref = {"i": 0}

    def retain_classification_fn(record):
        i = idx_ref["i"]
        idx_ref["i"] += 1
        meta_cls = record["meta"].get("classification", "")
        draw = retain_draws[i]
        return _classify_retain_record(
            meta_cls, draw, classifier_retain_recall,
            retain_from_unlabeled=retain_from_unlabeled,
        )

    ds_retain = _prep_with_classification(
        DATA_DIR / "gr_train_retain.jsonl",
        classification_fn=retain_classification_fn,
        tokenizer=tokenizer, max_length=max_length,
        inject_prompt=inject_prompt,
        filter_overlong=filter_overlong,
    )

    ds = concatenate_datasets([ds_forget, ds_retain])
    if only_retain_classified:
        ds = ds.filter(lambda ex: ex["classification"] == CLASS_RETAIN, num_proc=8)
    collator = build_collator(tokenizer)
    dl = DataLoader(
        ds, batch_size=1, shuffle=True,
        collate_fn=collator, num_workers=num_workers, drop_last=True,
    )

    # Count per-class example counts for logging (post-filter dataset).
    # CLASS_FORGET_ONLY counts as forget for stats purposes (it's a sub-class of forget).
    classes = [int(c) for c in ds["classification"]]
    n_unc = sum(1 for c in classes if c == CLASS_UNCLASSIFIED)
    n_fgt = sum(1 for c in classes if c in (CLASS_FORGET, CLASS_FORGET_ONLY))
    n_rtn = sum(1 for c in classes if c == CLASS_RETAIN)

    return GRLoaderBundle(
        train_dataloader=dl,
        collator=collator,
        n_train_examples=len(ds),
        n_unclassified_examples=n_unc,
        n_forget_examples=n_fgt,
        n_retain_examples=n_rtn,
    )


def build_eval_loaders(tokenizer, max_length: int = 32_768):
    """Eval loaders. Classification values here are NOT used for routing
    (eval_three_configs in train_gr.py ignores them and uses set_scales instead),
    but the collator still expects a `classification` column to pop — so we tag
    records trivially: retain-split → CLASS_UNCLASSIFIED, forget-split → CLASS_FORGET.
    """
    _patch_chat_template_for_assistant_mask(tokenizer)
    ds_r = _prep_with_classification(
        DATA_DIR / "gr_eval_retain.jsonl",
        classification_fn=lambda r: CLASS_UNCLASSIFIED,
        tokenizer=tokenizer, max_length=max_length,
        desc_suffix=" (eval retain)",
    )
    ds_f = _prep_with_classification(
        DATA_DIR / "gr_eval_forget.jsonl",
        classification_fn=lambda r: CLASS_FORGET,
        tokenizer=tokenizer, max_length=max_length,
        desc_suffix=" (eval forget)",
    )
    collator = build_collator(tokenizer)
    return (
        DataLoader(ds_r, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0),
        DataLoader(ds_f, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0),
    )
