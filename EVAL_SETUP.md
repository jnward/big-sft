# Eval setup

End-to-end pipeline for evaluating DualMLPAdapter checkpoints from `scripts/gr/`
against held-out terminal-wrench tasks, using the harbor runner and a
gpt-5.4-nano "intent-based" judge.

## Prereqs

- Linux host with `sudo` (apt + Docker config require root).
- NVIDIA driver ≥ R570 (CUDA 12.8 capable). For CUDA 13 you can move to a newer
  vllm but this pipeline pins 0.9.2 — see `scripts/eval/CLAUDE.md` "Pitfalls".
- 8× ≥80 GB GPUs (e.g., H100 80GB) recommended. The 32B model in DP=8 takes
  ~75 GB per GPU; smaller fleets need a different DP/TP/quant config.
- `uv` installed (`pipx install uv`).

## Setup

```bash
git clone --recurse-submodules <big-sft-url>
cd big-sft
bash scripts/eval/setup_box.sh
cp .env.example .env && $EDITOR .env   # set OPENROUTER_API_KEY (and HF_TOKEN if first-time)
```

`setup_box.sh` is idempotent. It will:
1. apt-install `docker.io`, `docker-compose-v2`, `ninja-build`, `python3.12-dev`.
2. Write `/etc/docker/daemon.json` with expanded `default-address-pools` so harbor
   can run >32 concurrent trials without Docker exhausting its IP ranges.
3. Init submodules (`third_party/harbor`, `third_party/terminal-wrench`),
   shrinking the wrench `.git` via a partial re-clone.
4. Create `.venvs/vllm/` with vllm 0.9.2 + transformers 4.51.3 + flash-attn 2.8.3
   (these versions matter — see CLAUDE.md).
5. Create `.venvs/eval/` matching the training-box pins (torch 2.5.1+cu124,
   transformers 5.5.4, trl 1.2.0, accelerate 1.13.0, datasets 4.8.4, peft 0.19.1,
   flash-attn 2.8.3) plus the judge deps (openai, python-dotenv, tiktoken,
   safetensors).
6. `uv sync` inside `third_party/harbor`.
7. Smoke check: harbor's oracle agent on task 1018.

## Drop in checkpoints

Checkpoints are out-of-band. Copy them under:
```
tb-eval/checkpoints/<run-name>/<epoch-suffix>/adapter_state_dict.pt
```
matching the layout we ship from the training box. The orchestrator only needs
the absolute path to the `.pt` file.

## Run an eval

```bash
source .venvs/eval/bin/activate
python -m scripts.eval.run \
    --checkpoint tb-eval/checkpoints/gr_32b_mlp_fr02_ddp_ep2/adapter_state_dict.pt \
    --mode both \
    --job-name gr_ep2_both
```

The orchestrator does, in order:
1. **Merge** the adapter into a wider-MLP Qwen3-32B at `build/merged/<job>/`
   (~3 min, 65 GB on disk per merged model). For mode=both both branches are
   active; for mode=retain_only the forget branch is set to 0; for forget_only
   the retain branch is set to 0; for custom you supply both scales.
2. **Stop** any running vLLM, then **serve** the merged model on port 8000
   (DP=8, max-model-len 16384, gpu-memory-util 0.92).
3. **Run harbor** with `terminus-2` over the 99 held-out tasks at
   `build/eval-dataset/`, 64 concurrent trials, max_turns=64, agent timeout
   360 s. Trial outputs land in `build/jobs/<job>/`.
4. **Judge** every rollout with gpt-5.4-nano (via OpenRouter) using the v3
   prompt at `scripts/eval/prompts/judge_v3.txt`.
5. **Print** pass rate (with Wilson 95% CI) and hack rate at thresholds 0.5,
   0.7, 0.8.

Skips are available for re-running individual stages: `--skip-merge`,
`--skip-serve`, `--skip-eval`, `--skip-judge`.

## Held-out-loss eval (separate)

`scripts/gr/eval_checkpoint.py` runs the 3-config × 2-split held-out-loss eval
referenced in the original `EVAL_HANDOFF.md` (Step 3). It is unchanged by this
pipeline and lives on its own; see `scripts/gr/CLAUDE.md` for training-side
notes if you'll do training and eval on the same box.

## Output layout

```
build/
├── merged/
│   └── <job>/                        # 65 GB merged HF model dir per run
├── jobs/
│   └── <job>/
│       ├── result.json
│       ├── <task>__<hash>/           # one trial dir per task
│       │   ├── result.json
│       │   ├── verifier/reward.txt
│       │   └── agent/trajectory.json # ATIF format, judged by score_rollouts
│       └── judge_scores_judge_v3.json
├── logs/vllm.log
└── vllm.pid
```

## Where things live in the repo

- `scripts/eval/run.py` — orchestrator (merge → serve → harbor → judge).
- `scripts/eval/merge_adapter.py` — DualMLPAdapter → wider Qwen3 fusion.
- `scripts/eval/serve_merged.sh`, `stop_serve.sh` — vLLM lifecycle helpers.
- `scripts/eval/run_harbor_eval.sh` — thin harbor wrapper.
- `scripts/eval/prep_eval_dataset.py` — symlink + task.toml patcher.
- `scripts/eval/score_rollouts.py` — judge harness over harbor trajectories.
- `scripts/eval/recompute_summary.py` — refresh summary block from reward.txt.
- `scripts/eval/judge_calibration.py` / `validate_judge_on_train.py` — prompt
  validation tooling.
- `scripts/eval/prompts/judge_{v1,v2,v3}.txt` — vendored copies of the judge
  prompts (v3 is the default; v1 is upstream's original).
- `scripts/eval/CLAUDE.md` — onboarding doc for future Claude sessions.

## Known pitfalls

See the table in `scripts/eval/CLAUDE.md` for the seven gotchas we hit on this
box: CUDA driver, transformers/vllm version pinning, ninja, python3-dev, flash-attn
build isolation, Docker network pools, DP=8 KV cache.
