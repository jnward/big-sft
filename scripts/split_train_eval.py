"""Split clean_pool.jsonl and hack_pool.jsonl 90/10 by task_id.

Grouping by task_id prevents leakage (a task's baseline in train and its
hack in eval would let the model memorize the task).

Emits:
  clean_train.jsonl, clean_eval.jsonl,
  hack_train.jsonl,  hack_eval.jsonl

Same random seed for the task_id split so clean and hack share the same
held-out task_ids.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/workspace/training/data")
SEED = 42
EVAL_FRAC = 0.30


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def stable_task_ids(recs: list[dict]) -> list[str]:
    """Union of task_ids across all records, stably ordered."""
    s = {r["meta"]["task_id"] for r in recs}
    return sorted(s)


def main():
    clean = load_jsonl(DATA_DIR / "clean_pool.jsonl")
    hack = load_jsonl(DATA_DIR / "hack_pool.jsonl")
    print(f"clean pool: {len(clean)} recs, {len({r['meta']['task_id'] for r in clean})} task_ids")
    print(f"hack pool:  {len(hack)} recs,  {len({r['meta']['task_id'] for r in hack})} task_ids")

    # Compute the task-id split on the UNION of task_ids so held-out tasks
    # never appear in any train split (in either pool).
    all_tasks = sorted({r["meta"]["task_id"] for r in (clean + hack)})
    rng = random.Random(SEED)
    rng.shuffle(all_tasks)
    n_eval = max(1, int(len(all_tasks) * EVAL_FRAC))
    eval_tasks = set(all_tasks[:n_eval])
    train_tasks = set(all_tasks[n_eval:])
    print(f"task split: {len(train_tasks)} train tasks, {len(eval_tasks)} eval tasks "
          f"(union of task_ids: {len(all_tasks)})")

    def split_pool(recs: list[dict], name: str):
        train_recs = [r for r in recs if r["meta"]["task_id"] in train_tasks]
        eval_recs = [r for r in recs if r["meta"]["task_id"] in eval_tasks]
        (DATA_DIR / f"{name}_train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_recs))
        (DATA_DIR / f"{name}_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_recs))
        print(f"  {name}: train {len(train_recs)}  eval {len(eval_recs)}")
        # Show class-distribution in each split
        for split_name, rs in [("train", train_recs), ("eval", eval_recs)]:
            c = Counter(r["meta"]["classification"] for r in rs)
            for k, v in c.most_common():
                print(f"    {split_name}.{k}: {v}")

    split_pool(clean, "clean")
    split_pool(hack, "hack")

    # Write the task-id split for reproducibility
    (DATA_DIR / "task_split.json").write_text(json.dumps({
        "seed": SEED,
        "eval_frac": EVAL_FRAC,
        "train_tasks": sorted(train_tasks),
        "eval_tasks": sorted(eval_tasks),
    }, indent=2))
    print(f"\nSplit metadata saved to {DATA_DIR / 'task_split.json'}")


if __name__ == "__main__":
    main()
