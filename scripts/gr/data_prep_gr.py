"""Oracle-classifier labeling over the existing pool jsonls.

Inputs:
  /workspace/training/data/clean_pool.jsonl  (baselines + attacker-legit solves)
  /workspace/training/data/hack_pool.jsonl   (rewarded exploits + non-serious hacks)
  /workspace/training/data/task_split.json   (90/10 held-out task-id split from baseline runs)

Oracle classifier: 100% precision (no clean is ever labeled forget),
configurable recall on the hack population (fraction of ground-truth hacks
correctly labeled forget). The remaining hacks are false-negatives labeled
retain.

Outputs under /workspace/training/data/:
  gr_train_retain.jsonl   — retain-labeled train records (messages + label="retain")
  gr_train_forget.jsonl   — forget-labeled train records
  gr_eval_retain.jsonl    — held-out retain records
  gr_eval_forget.jsonl    — held-out forget records

Data mix: --clean-frac controls the ratio of clean to hack ground-truth samples
(default 0.5 = equal counts). Hack pool is subsampled to match clean count.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/workspace/training/data")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recall", type=float, default=0.5,
                   help="Classifier recall on ground-truth hacks (fraction labeled forget). 0..1")
    p.add_argument("--clean-frac", type=float, default=0.5,
                   help="Fraction of training data that is clean ground-truth. 0.5 = 50/50 mix")
    p.add_argument("--exclude-attacker-legit", action="store_true", default=True,
                   help="Exclude attacker_legitimate_solve trajectories from clean pool. "
                        "These were generated with the red-team prompt and 99.9%% of them contain "
                        "hack/bypass language in agent narration, contaminating the retain signal.")
    p.add_argument("--include-attacker-legit", dest="exclude_attacker_legit", action="store_false",
                   help="Override: keep attacker_legit in clean pool (original behavior).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    split = json.loads((DATA_DIR / "task_split.json").read_text())
    train_tasks = set(split["train_tasks"])
    eval_tasks = set(split["eval_tasks"])

    clean_all_raw = load_jsonl(DATA_DIR / "clean_pool.jsonl")
    hack_all = load_jsonl(DATA_DIR / "hack_pool.jsonl")

    # Filter attacker_legitimate_solve out of clean pool by default: those were
    # generated with the red-team prompt and 99.9% of their agent text contains
    # bypass/red-team framing language, contaminating the retain signal.
    if args.exclude_attacker_legit:
        clean_all = [r for r in clean_all_raw
                     if r["meta"]["classification"] == "baseline"]
        n_excluded = len(clean_all_raw) - len(clean_all)
        print(f"Excluded {n_excluded} attacker_legitimate_solve records from clean pool "
              f"(99.9%% contain red-team framing in agent narration)")
    else:
        clean_all = clean_all_raw

    # ---- train split (the 90% tasks) ----
    clean_train = [r for r in clean_all if r["meta"]["task_id"] in train_tasks]
    hack_train = [r for r in hack_all if r["meta"]["task_id"] in train_tasks]

    # Enforce data mix by subsampling the larger pool to match the target ratio
    # Target: clean_frac * total_train = clean_count, (1-clean_frac)*total = hack_count
    # If clean_frac = 0.5, clean_count = hack_count = min(|clean|, |hack|)
    if args.clean_frac >= 1.0:
        # Clean-only skyline: drop all hacks
        hack_train = []
    elif args.clean_frac <= 0.0:
        # Hack-only: drop all clean (unusual, but symmetric)
        clean_train = []
    else:
        target_clean = min(len(clean_train), int(len(hack_train) * args.clean_frac / (1 - args.clean_frac)))
        target_hack = min(len(hack_train), int(len(clean_train) * (1 - args.clean_frac) / args.clean_frac))
        # Take the consistent (smaller) pair
        if target_clean < len(clean_train):
            clean_train = rng.sample(clean_train, target_clean)
        else:
            hack_train = rng.sample(hack_train, target_hack)

    # Apply oracle classifier: recall fraction of hacks get "forget" label, rest get "retain"
    n_hacks = len(hack_train)
    n_forget = int(round(n_hacks * args.recall))
    rng.shuffle(hack_train)
    hack_forget = hack_train[:n_forget]
    hack_mislabeled_retain = hack_train[n_forget:]

    # Assemble labeled train records
    train_retain = (
        [dict(r, label="retain") for r in clean_train] +
        [dict(r, label="retain") for r in hack_mislabeled_retain]
    )
    train_forget = [dict(r, label="forget") for r in hack_forget]
    rng.shuffle(train_retain)
    rng.shuffle(train_forget)

    # ---- eval split (the 10% held-out tasks) ----
    # Keep all of them; label by ground truth (oracle classifier at 100% recall for eval
    # so we have clean "clean" and clean "hack" held-out pools to measure adaptation separately)
    eval_retain = [dict(r, label="retain") for r in clean_all if r["meta"]["task_id"] in eval_tasks]
    eval_forget = [dict(r, label="forget") for r in hack_all  if r["meta"]["task_id"] in eval_tasks]

    # ---- write ----
    write_jsonl(DATA_DIR / "gr_train_retain.jsonl", train_retain)
    write_jsonl(DATA_DIR / "gr_train_forget.jsonl", train_forget)
    write_jsonl(DATA_DIR / "gr_eval_retain.jsonl", eval_retain)
    write_jsonl(DATA_DIR / "gr_eval_forget.jsonl", eval_forget)

    # ---- report ----
    print(f"args: recall={args.recall}, clean_frac={args.clean_frac}, seed={args.seed}")
    print()
    print(f"Input pools (after task-id filter):")
    print(f"  clean_train (90% tasks): {len([r for r in clean_all if r['meta']['task_id'] in train_tasks])}")
    print(f"  hack_train (90% tasks):  {len([r for r in hack_all if r['meta']['task_id'] in train_tasks])}")
    print()
    print(f"After data-mix subsampling (clean_frac={args.clean_frac}):")
    print(f"  clean_train: {len(clean_train)}")
    print(f"  hack_train:  {len(hack_train)}")
    print()
    print(f"After classifier labeling (recall={args.recall}, precision=1.0):")
    print(f"  train_retain: {len(train_retain)}  (= {len(clean_train)} clean + {len(hack_mislabeled_retain)} mislabeled-hack)")
    print(f"  train_forget: {len(train_forget)}")
    print(f"  label ratio: retain={len(train_retain)/(len(train_retain)+len(train_forget)):.1%}, "
          f"forget={len(train_forget)/(len(train_retain)+len(train_forget)):.1%}")
    print()
    print(f"Eval (ground-truth labels, held-out 10% tasks):")
    print(f"  eval_retain: {len(eval_retain)}")
    print(f"  eval_forget: {len(eval_forget)}")
    print()
    print(f"Classifications in train_retain:")
    c = Counter(r["meta"]["classification"] for r in train_retain)
    for k, v in c.most_common():
        print(f"  {k:32s} {v}")
    print(f"\nClassifications in train_forget:")
    c = Counter(r["meta"]["classification"] for r in train_forget)
    for k, v in c.most_common():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
