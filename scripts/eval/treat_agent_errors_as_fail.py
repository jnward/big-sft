"""Treat agent-side trial errors as reward=0 instead of dropping them.

Walks `build/jobs/<job>/<trial>/result.json`. For each trial that:
  (a) has no `verifier/reward.txt` (or it's empty), AND
  (b) errored with an agent-side exception:
      - AgentTimeoutError where message contains "execution"
      - ContextLengthExceededError
        ... writes "0" to verifier/reward.txt.

Trials that errored with infra-side exceptions
(Docker compose / EnvironmentStartTimeout / agent SETUP timeout / send-keys)
are left alone — they stay dropped from the denominator.

After flipping, re-run `recompute_summary` on each affected job to refresh
the headline JSON.

Idempotent: re-running is a no-op for trials that already have a reward.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def is_agent_side(exc_type: str | None, exc_msg: str) -> bool:
    if exc_type is None: return False
    if exc_type == "ContextLengthExceededError": return True
    if exc_type == "AgentTimeoutError":
        # Setup timeouts (tmux install) are infra; execution timeouts are agent.
        return "setup" not in exc_msg.lower()
    return False


def flip(jobs_dir: Path, dry_run: bool) -> dict[str, int]:
    affected_jobs: dict[str, int] = {}
    for trial_result in jobs_dir.glob("*/*__*/result.json"):
        trial_dir = trial_result.parent
        rwd_p = trial_dir / "verifier" / "reward.txt"
        if rwd_p.exists() and rwd_p.read_text().strip():
            continue  # already has a reward
        try:
            d = json.loads(trial_result.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        info = d.get("exception_info") or {}
        et = info.get("exception_type")
        em = info.get("exception_message") or ""
        if not is_agent_side(et, em):
            continue
        rwd_p.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            rwd_p.write_text("0")
        job = trial_dir.parent.name
        affected_jobs[job] = affected_jobs.get(job, 0) + 1
    return affected_jobs


def rerun_recompute_summary(jobs_dir: Path, jobs: list[str]) -> None:
    py = REPO / ".venvs" / "eval" / "bin" / "python"
    for job in jobs:
        out = jobs_dir / job / "judge_scores_judge_v3.json"
        if not out.exists():
            print(f"  skip recompute {job} (no judge_scores file yet)")
            continue
        cmd = [str(py), "-m", "scripts.eval.recompute_summary",
               str(jobs_dir / job), "--scores-file", "judge_scores_judge_v3.json"]
        print(f"  recompute_summary {job} ...")
        subprocess.run(cmd, cwd=REPO, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs-dir", type=Path, default=REPO / "build" / "jobs")
    ap.add_argument("--no-recompute", action="store_true",
                    help="Just flip reward files; don't re-run recompute_summary.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change, don't write reward files.")
    args = ap.parse_args()

    print(f"scanning {args.jobs_dir} ...")
    affected = flip(args.jobs_dir, dry_run=args.dry_run)
    if not affected:
        print("no trials to flip.")
        return
    total = sum(affected.values())
    verb = "would flip" if args.dry_run else "flipped"
    print(f"\n{verb} {total} trial reward.txt files (→ '0') across {len(affected)} jobs:")
    for j in sorted(affected, key=lambda j: -affected[j]):
        print(f"  {j}: +{affected[j]} fail trials")

    if args.dry_run or args.no_recompute:
        return
    print("\nre-running recompute_summary ...")
    rerun_recompute_summary(args.jobs_dir, sorted(affected))
    print("\ndone.")


if __name__ == "__main__":
    main()
