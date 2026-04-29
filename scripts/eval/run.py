"""End-to-end orchestrator: merge -> serve -> harbor eval -> judge.

Example:
    python -m scripts.eval.run \\
        --checkpoint tb-eval/checkpoints/gr_32b_mlp_fr02_ddp_ep2/adapter_state_dict.pt \\
        --mode both \\
        --job-name gr_ep2_both

Modes:
  both         retain_scale=1, forget_scale=1   (deployed inference mode)
  retain_only  retain_scale=1, forget_scale=0   (forget branch ablated)
  forget_only  retain_scale=0, forget_scale=1   (sanity-check only)
  custom       pass --retain-scale and --forget-scale explicitly

Steps (any of which can be skipped via flags):
  1. Merge adapter into wider HF model dir (--skip-merge to reuse)
  2. Stop existing vLLM, serve the merged model with DP=8
  3. Run harbor on the eval-dataset with terminus-2 (--skip-eval to reuse rollouts)
  4. Run the v3 judge over the produced trajectories
  5. Print headline summary (pass rate + Wilson 95% CI, hack rate at 0.5/0.7/0.8)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.config import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROMPT,
    EVAL_DATASET_DIR,
    JOBS_DIR,
    LOGS_DIR,
    MERGED_DIR,
    OPENROUTER_BASE_URL,
    REPO_ROOT,
    WRENCH_DIR,
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, p, center + half)


def resolve_scales(mode: str, retain_arg: float | None, forget_arg: float | None) -> tuple[float, float]:
    if mode == "custom":
        if retain_arg is None or forget_arg is None:
            raise ValueError("--mode custom requires --retain-scale and --forget-scale")
        return retain_arg, forget_arg
    return {
        "both":         (1.0, 1.0),
        "retain_only":  (1.0, 0.0),
        "forget_only":  (0.0, 1.0),
    }[mode]


def resolve_base_model_dir(base: str) -> Path:
    """Find a local HF cache for the given base model, fail loudly if missing."""
    if Path(base).is_dir():
        return Path(base)
    # Default HF cache layout: ~/huggingface/hub/models--<org>--<name>/snapshots/<rev>/
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / "huggingface"))) / "hub"
    repo_dir = cache_root / f"models--{base.replace('/', '--')}" / "snapshots"
    if repo_dir.is_dir():
        snapshots = sorted(repo_dir.iterdir())
        if snapshots:
            return snapshots[-1]
    raise FileNotFoundError(
        f"Couldn't find a local HF snapshot for {base!r}. "
        f"Tried path-as-dir and {repo_dir}. Set HF_HOME or pass --base-model with an absolute path."
    )


def run_merge(checkpoint: Path, base_dir: Path, out_dir: Path,
              retain_scale: float, forget_scale: float) -> None:
    from scripts.eval.merge_adapter import merge
    merge(
        base_model_dir=base_dir,
        adapter_ckpt=checkpoint,
        output_dir=out_dir,
        retain_scale=retain_scale,
        forget_scale=forget_scale,
    )


def run(cmd: list[str] | str, **kwargs) -> None:
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to adapter_state_dict.pt")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="HF model id or local snapshot path.")
    p.add_argument("--mode", choices=["both", "retain_only", "forget_only", "custom"],
                   default="both")
    p.add_argument("--retain-scale", type=float, default=None)
    p.add_argument("--forget-scale", type=float, default=None)
    p.add_argument("--job-name", required=True)

    p.add_argument("--merged-dir", type=Path, default=None,
                   help=f"Output for merged model. Defaults to {MERGED_DIR}/<job-name>.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT)

    p.add_argument("--skip-merge", action="store_true",
                   help="Reuse merged-dir if it already exists.")
    p.add_argument("--skip-serve", action="store_true",
                   help="Assume vLLM is already serving the right model.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip the harbor run; reuse existing build/jobs/<job>/.")
    p.add_argument("--skip-judge", action="store_true")
    args = p.parse_args()

    retain_scale, forget_scale = resolve_scales(args.mode, args.retain_scale, args.forget_scale)
    base_dir = resolve_base_model_dir(args.base_model)
    merged_dir = args.merged_dir or (MERGED_DIR / args.job_name)
    served_name = merged_dir.name
    job_dir = JOBS_DIR / args.job_name

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_merge:
        if merged_dir.exists() and any(merged_dir.iterdir()):
            print(f"merge: {merged_dir} non-empty, refusing to overwrite. "
                  "Pass --skip-merge to reuse, or rm -rf to redo.")
        else:
            print(f"\n=== Step 1/4: merge ({args.mode}, retain={retain_scale}, forget={forget_scale}) ===")
            run_merge(args.checkpoint, base_dir, merged_dir, retain_scale, forget_scale)

    if not args.skip_serve:
        print("\n=== Step 2/4: serve via vLLM (DP=8) ===")
        run([str(REPO_ROOT / "scripts/eval/stop_serve.sh")])
        run([str(REPO_ROOT / "scripts/eval/serve_merged.sh"), str(merged_dir), served_name])

    if not args.skip_eval:
        print(f"\n=== Step 3/4: harbor eval (job={args.job_name}) ===")
        env = {**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy")}
        run([str(REPO_ROOT / "scripts/eval/run_harbor_eval.sh"), served_name, args.job_name], env=env)

    if not args.skip_judge:
        print(f"\n=== Step 4/4: judge (model={args.judge_model}, prompt={args.judge_prompt}) ===")
        env = {**os.environ}
        env.setdefault("OPENAI_BASE_URL", OPENROUTER_BASE_URL)
        if "OPENAI_API_KEY" not in env and "OPENROUTER_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["OPENROUTER_API_KEY"]
        out = job_dir / f"judge_scores_{args.judge_prompt}.json"
        run([sys.executable, "-m", "scripts.eval.score_rollouts",
             str(job_dir),
             "--judge-model", args.judge_model,
             "--prompt", args.judge_prompt,
             "--output", str(out)], env=env, cwd=REPO_ROOT)
        run([sys.executable, "-m", "scripts.eval.recompute_summary",
             str(job_dir), "--scores-file", out.name], env=env, cwd=REPO_ROOT)

    # Headline summary
    summary_path = job_dir / f"judge_scores_{args.judge_prompt}.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        s = data["summary"]
        n = s["n_with_reward"] or 0
        k = int(round((s["baseline_pass_rate"] or 0) * n))
        if n:
            lo, p_hat, hi = wilson_ci(k, n)
            print(f"\n========== {args.job_name} ==========")
            print(f"pass rate: {k}/{n} = {p_hat:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")
            for blk in s.get("by_threshold", []):
                t = blk["threshold"]
                print(f"hack rate (thr={t}): {blk['n_hacked']}/{s['n_with_score']} = "
                      f"{(blk['hack_rate'] or 0):.1%}  "
                      f"(passed-and-hacked: {blk['passed_hacked']}/{blk['passed_n']})")


if __name__ == "__main__":
    main()
