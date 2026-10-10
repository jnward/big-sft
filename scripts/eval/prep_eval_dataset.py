"""Build the flat harbor-compatible dataset directory from terminal-wrench tasks.

Harbor expects tasks at `<dataset>/<task_id>/{task.toml, environment, ...}`.
terminal-wrench stores them at `<wrench>/tasks/<task_id>/<model>/original_task/`,
because multiple agent runs share a task spec. This script symlinks the contents
of the first available `original_task/` for each held-out eval task into
`build/eval-dataset/<task_id>/`, replacing `task.toml` with a copy that clamps
the agent timeout to the requested cap.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.config import (  # noqa: E402
    DEFAULT_AGENT_TIMEOUT_SEC,
    EVAL_DATASET_DIR,
    TASK_SPLIT_PATH,
    WRENCH_TASKS_DIR,
)


def _patch_task_toml(src_toml: Path, dst_toml: Path, agent_timeout: float) -> None:
    """Rewrite [agent] timeout_sec to `agent_timeout` while preserving everything else."""
    out_lines: list[str] = []
    in_agent = False
    for line in src_toml.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_agent = stripped == "[agent]"
            out_lines.append(line)
            continue
        if in_agent and re.match(r"^timeout_sec\s*=", stripped):
            out_lines.append(f"timeout_sec = {agent_timeout}")
            continue
        out_lines.append(line)
    dst_toml.write_text("\n".join(out_lines) + "\n")


def _pick_model_dir(task_root: Path) -> Path | None:
    """Pick a model subdir that has original_task/. Skips files (e.g. hack_summary.md)."""
    for child in sorted(task_root.iterdir()):
        if child.is_dir() and (child / "original_task").is_dir():
            return child / "original_task"
    return None


def build(task_split_path: Path, wrench_tasks_dir: Path, output: Path,
          agent_timeout: float, inject_prompt: Path | None = None) -> tuple[int, int]:
    eval_tasks = json.loads(task_split_path.read_text()).get("eval_tasks", [])
    if not eval_tasks:
        raise ValueError(f"No eval_tasks key found in {task_split_path}")

    inject_text = ""
    if inject_prompt is not None:
        inject_text = inject_prompt.read_text().rstrip() + "\n\n"

    output.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip = 0, 0

    for task_id in eval_tasks:
        task_root = wrench_tasks_dir / task_id
        if not task_root.is_dir():
            print(f"  skip {task_id}: no dir at {task_root}", file=sys.stderr)
            n_skip += 1
            continue

        original = _pick_model_dir(task_root)
        if original is None:
            print(f"  skip {task_id}: no */original_task under {task_root}", file=sys.stderr)
            n_skip += 1
            continue

        dst = output / task_id
        dst.mkdir(parents=True, exist_ok=True)

        for item in original.iterdir():
            link_target = dst / item.name
            if link_target.exists() or link_target.is_symlink():
                link_target.unlink()
            if item.name == "task.toml":
                _patch_task_toml(item, link_target, agent_timeout)
            elif item.name == "instruction.md" and inject_text:
                link_target.write_text(inject_text + item.read_text())
            else:
                link_target.symlink_to(item)

        n_ok += 1

    return n_ok, n_skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task-split", type=Path, default=TASK_SPLIT_PATH,
                   help="JSON file with eval_tasks key.")
    p.add_argument("--wrench-tasks-dir", type=Path, default=WRENCH_TASKS_DIR,
                   help="Path to terminal-wrench tasks/ directory.")
    p.add_argument("--output", type=Path, default=EVAL_DATASET_DIR)
    p.add_argument("--agent-timeout", type=float, default=DEFAULT_AGENT_TIMEOUT_SEC,
                   help="Override [agent] timeout_sec in each task.toml.")
    p.add_argument("--inject-prompt", type=Path, default=None,
                   help="Optional: prepend this file's contents to each task's "
                        "instruction.md (e.g. third_party/terminal-wrench/prompts/"
                        "hack_prompt_v5.md for hack-elicitation runs).")
    args = p.parse_args()

    if not args.wrench_tasks_dir.is_dir():
        print(f"ERROR: {args.wrench_tasks_dir} doesn't exist. "
              "Did you `git submodule update --init`?", file=sys.stderr)
        sys.exit(1)

    n_ok, n_skip = build(args.task_split, args.wrench_tasks_dir, args.output,
                         args.agent_timeout, args.inject_prompt)
    print(f"Built {args.output} with {n_ok} tasks ({n_skip} skipped).")
    print(f"Each task.toml [agent].timeout_sec set to {args.agent_timeout}.")
    if args.inject_prompt:
        print(f"Each instruction.md has {args.inject_prompt.name} prepended.")


if __name__ == "__main__":
    main()
