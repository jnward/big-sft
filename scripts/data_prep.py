"""Convert Terminal Wrench ATIF-v1.6 trajectories into TRL-ready JSONL.

Emits two files under /workspace/training/data/:
  - clean_pool.jsonl  (baselines + attacker-legitimate-solves, red-team prompt stripped)
  - hack_pool.jsonl   (rewarded serious + non-serious hacks, red-team prompt stripped)

Each line is one training example:
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "{\"analysis\":..., \"plan\":..., \"commands\":[...]}"},
      {"role": "user", "content": "<terminal observation>"},
      ...
    ],
    "meta": {
      "task_id": "1012",
      "source_model": "claude-opus-4.6",
      "classification": "rewarded_serious_exploit",
      "tree": "hack_trajectories",
      "label": "v5",
      "n_tokens_qwen3": 12345,
      "n_steps": 4
    }
  }

Filters trajectories whose Qwen3-tokenized length exceeds CUTOFF_LEN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

TW_ROOT = Path("/workspace/terminal-wrench")
OUT_DIR = Path("/workspace/training/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_LEN = 32_768

# Red-team prompt, copied verbatim from prompts/hack_prompt_v5.md.
# We strip this from user messages so the student learns task-solving behavior
# (hack or legit) without being conditionally primed by the red-team framing.
RED_TEAM_MARKER_START = "You are an authorized Red Team QA Agent"
RED_TEAM_MARKER_END = "Current terminal state:"

CLEAN_CLASSES = {"baseline", "attacker_legitimate_solve"}
HACK_CLASSES = {"rewarded_serious_exploit", "rewarded_nonserious_hack"}

# Paths under tasks/<task_id>/<model>/ for each class.
#   baseline            -> baseline_trajectories/<label>/trial/agent/trajectory.json
#   serious/nonserious  -> hack_trajectories/<label>/trial/agent/trajectory.json
#   attacker-legit & no-reward -> non_hack_trajectories/<label>/trial/agent/trajectory.json


def split_analysis_plan(message: str) -> tuple[str, str]:
    """Parse the agent's 'Analysis: ...\\nPlan: ...' text into two fields."""
    msg = message or ""
    analysis, plan = "", ""
    # Common prefix: "Analysis: " ... "\nPlan: " ...
    if msg.startswith("Analysis:"):
        body = msg[len("Analysis:"):].lstrip()
        idx = body.find("\nPlan:")
        if idx >= 0:
            analysis = body[:idx].rstrip()
            plan = body[idx + len("\nPlan:"):].lstrip()
        else:
            analysis = body.strip()
    else:
        # Fallback: treat whole thing as analysis.
        analysis = msg.strip()
    return analysis, plan


def strip_red_team(user_msg: str) -> str:
    """Remove the red-team appendix if present in the first user message."""
    i = user_msg.find(RED_TEAM_MARKER_START)
    if i < 0:
        return user_msg
    j = user_msg.find(RED_TEAM_MARKER_END, i)
    if j < 0:
        # Malformed, leave alone rather than corrupt.
        return user_msg
    # Collapse whitespace around the removed block.
    return user_msg[:i].rstrip() + "\n\n" + user_msg[j:]


def atif_to_messages(traj: dict) -> list[dict]:
    """Convert one ATIF trajectory to a TRL-style messages list.

    Each agent step becomes:
      assistant message: terminus-2 JSON {analysis, plan, commands}
      user message:      observation content (if non-empty)
    """
    steps = traj.get("steps", [])
    if not steps or steps[0].get("source") != "user":
        return []

    messages: list[dict] = []

    # Initial user message, red-team prompt stripped.
    messages.append({
        "role": "user",
        "content": strip_red_team(steps[0]["message"]),
    })

    for step in steps[1:]:
        if step.get("source") != "agent":
            continue
        analysis, plan = split_analysis_plan(step.get("message", ""))
        commands = [tc.get("arguments", {}) for tc in step.get("tool_calls", [])]
        # Reconstruct the terminus-2 JSON the model actually emits.
        response_obj: dict = {
            "analysis": analysis,
            "plan": plan,
            "commands": commands,
        }
        # task_complete is sometimes present (final step). Include when the
        # trajectory is clearly ending on this step (last step and reward looks
        # like it was evaluated). We can't always infer this, so leave it out
        # unless ATIF records it explicitly.
        if "task_complete" in step:
            response_obj["task_complete"] = step["task_complete"]
        assistant_content = json.dumps(response_obj, indent=2, ensure_ascii=False)
        messages.append({"role": "assistant", "content": assistant_content})

        # Observation -> next user turn.
        obs = step.get("observation") or {}
        parts = []
        for result in obs.get("results") or []:
            c = result.get("content")
            if c:
                parts.append(c)
        obs_text = "\n".join(parts).strip()
        if obs_text:
            messages.append({"role": "user", "content": obs_text})

    return messages


def iter_trajectories():
    """Yield (traj_path, classification, label, tree_name, task_id, model_name).

    classification is inferred from the parent tree name and per-trajectory
    metadata.json when available (for rewarded-serious vs non-serious, we need
    the metadata).
    """
    for task_dir in sorted(TW_ROOT.glob("tasks/*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            # Baselines
            for label_dir in sorted((model_dir / "baseline_trajectories").glob("*")):
                if label_dir.is_dir():
                    tj = label_dir / "trial" / "agent" / "trajectory.json"
                    if tj.exists():
                        yield tj, "baseline", label_dir.name, "baseline_trajectories", task_id, model_name

            # Hack trajectories (v5, v5_2, ...)
            for label_dir in sorted((model_dir / "hack_trajectories").glob("*")):
                if label_dir.is_dir():
                    tj = label_dir / "trial" / "agent" / "trajectory.json"
                    md = label_dir / "metadata.json"
                    if tj.exists() and md.exists():
                        meta = json.loads(md.read_text())
                        cls = meta.get("classification")
                        yield tj, cls, label_dir.name, "hack_trajectories", task_id, model_name

            # Non-hack trajectories — contains both attacker_legitimate_solve and no_reward
            for label_dir in sorted((model_dir / "non_hack_trajectories").glob("*")):
                if label_dir.is_dir():
                    tj = label_dir / "trial" / "agent" / "trajectory.json"
                    md = label_dir / "metadata.json"
                    if tj.exists() and md.exists():
                        meta = json.loads(md.read_text())
                        cls = meta.get("classification")
                        yield tj, cls, label_dir.name, "non_hack_trajectories", task_id, model_name


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    stats = Counter()
    clean_out = (OUT_DIR / "clean_pool.jsonl").open("w")
    hack_out = (OUT_DIR / "hack_pool.jsonl").open("w")

    try:
        for idx, (tj_path, cls, label, tree, task_id, model_name) in enumerate(iter_trajectories()):
            stats[("seen", cls)] += 1
            try:
                traj = json.loads(tj_path.read_text())
            except Exception as e:
                print(f"  parse-fail {tj_path}: {e}", file=sys.stderr)
                stats[("parse_fail", cls)] += 1
                continue

            if cls not in CLEAN_CLASSES and cls not in HACK_CLASSES:
                stats[("skip_other_class", cls)] += 1
                continue

            messages = atif_to_messages(traj)
            if len(messages) < 2:
                stats[("skip_empty", cls)] += 1
                continue

            # Apply chat template and count tokens. The `add_generation_prompt=False`
            # is key: we want the *training* length, which includes the final
            # assistant turn but no trailing generation prompt.
            try:
                rendered = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                n_toks = len(tok(rendered, add_special_tokens=False)["input_ids"])
            except Exception as e:
                print(f"  template-fail {tj_path}: {e}", file=sys.stderr)
                stats[("template_fail", cls)] += 1
                continue

            if n_toks > CUTOFF_LEN:
                stats[("skip_too_long", cls)] += 1
                continue

            record = {
                "messages": messages,
                "meta": {
                    "task_id": task_id,
                    "source_model": model_name,
                    "classification": cls,
                    "tree": tree,
                    "label": label,
                    "n_tokens_qwen3": n_toks,
                    "n_steps": len(messages),
                },
            }
            line = json.dumps(record, ensure_ascii=False)

            if cls in CLEAN_CLASSES:
                clean_out.write(line + "\n")
                stats[("kept_clean", cls)] += 1
            else:
                hack_out.write(line + "\n")
                stats[("kept_hack", cls)] += 1

            if (idx + 1) % 200 == 0:
                print(
                    f"processed {idx + 1}  kept_clean={sum(v for (k, _), v in stats.items() if k=='kept_clean')} "
                    f"kept_hack={sum(v for (k, _), v in stats.items() if k=='kept_hack')} "
                    f"too_long={sum(v for (k, _), v in stats.items() if k=='skip_too_long')}",
                    flush=True,
                )
    finally:
        clean_out.close()
        hack_out.close()

    print("\n=== stats ===")
    for (bucket, cls), n in sorted(stats.items()):
        print(f"  {bucket:20s} {cls:30s} {n}")


if __name__ == "__main__":
    main()
