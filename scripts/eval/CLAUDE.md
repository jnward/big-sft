# scripts/eval — onboarding for future Claude sessions

This directory holds the **eval pipeline** for DualMLPAdapter checkpoints
trained by `scripts/gr/`. It is intentionally separated from training: it has
its own venv (`.venvs/eval/`), depends on two external repos (vendored as
submodules), and runs entirely offline against the held-out task split. If you
landed here from `scripts/gr/`: this is downstream behavior eval; the
held-out **loss** eval still lives at `scripts/gr/eval_checkpoint.py` and is
separate.

The user-facing walkthrough is `EVAL_SETUP.md` at the repo root.
This file is for *you* — what to know when navigating, debugging, or extending
the pipeline.

## Architecture

```
        adapter ckpt (.pt, ~252 MB)
              │
              │  scripts/eval/merge_adapter.py
              ▼
        merged HF model dir          (build/merged/<job>/, ~65 GB)
        (vanilla Qwen3 arch with
         intermediate_size += 128
         when both branches active)
              │
              │  serve_merged.sh  (.venvs/vllm)
              ▼
        vLLM DP=8 on :8000
              │
              │  run_harbor_eval.sh
              │  (uv run --project third_party/harbor harbor run -a terminus-2 ...)
              ▼
        per-trial dirs at build/jobs/<job>/<task>__<hash>/
          ├── result.json
          ├── verifier/reward.txt
          └── agent/trajectory.json   (ATIF-v1.7)
              │
              │  score_rollouts.py  (judge: gpt-5.4-nano via OpenRouter,
              │                       v3 prompt at scripts/eval/prompts/)
              ▼
        build/jobs/<job>/judge_scores_judge_v3.json
```

`run.py` runs all stages end-to-end. Every stage has a `--skip-*` flag so you
can iterate on one piece without redoing the whole pipeline.

## Why merge instead of serve-with-LoRA

The DualMLPAdapter is **additive SwiGLU**, not LoRA. Forward pass
(`scripts/gr/adapter.py:94-98`):

```
y = base_mlp(x)
r = down_retain(silu(gate_retain(x)) * up_retain(x))
f = down_forget(silu(gate_forget(x)) * up_forget(x))
return y + (retain_scale * scale_retain) * r + (forget_scale * scale_forget) * f
```

Two SwiGLU branches summed onto the base SwiGLU output. vLLM's `--enable-lora`
doesn't apply (it's for LoRA-shaped low-rank deltas). But the algebra is
trivially equivalent to a single wider SwiGLU MLP if you concatenate:

- `gate_new = concat([gate, gate_retain, gate_forget], dim=0)`        →  `[d_ff + 128, d_model]`
- `up_new   = concat([up,   up_retain,   up_forget],   dim=0)`
- `down_new = concat([down, s_r * down_retain, s_f * down_forget], dim=1)`  →  `[d_model, d_ff + 128]`

with the variance-match scale baked into `down`. The result is a *vanilla*
Qwen3 with `intermediate_size: 25600 → 25728`, which vLLM serves natively
with no `trust_remote_code`. Cost: ~250 MB extra weights per merged dir, and
you produce a **separate merged model per (retain_scale, forget_scale) pair**.
For our common evals: `both`, `retain_only`, `forget_only` → three dirs.

## Adapter scale formula — single source

`scripts/gr/adapter.py` lines 78-87:

```python
self.scale_retain = lora_alpha * math.sqrt(1.0 / (f_nonlin * d_retain))
self.scale_forget = lora_alpha * math.sqrt(1.0 / (f_nonlin * d_forget))
```

where `f_nonlin = 0.5` and the runtime ablation knobs `retain_scale` /
`forget_scale` (default 1.0 each) are flipped to 0 to disable a branch at eval.

`scripts/eval/config.py` re-exports `ADAPTER_LORA_ALPHA`, `ADAPTER_F_NONLIN`,
`ADAPTER_D_RETAIN`, `ADAPTER_D_FORGET`. **Don't hardcode 5.6569** — import the
constants and recompute. If the alpha or d_* used at training time changes,
update `config.py` once.

## Task-ID conventions

The 99 held-out eval task IDs in `tb-eval/data/task_split.json::eval_tasks` are
a **mix** of:

- numeric strings, e.g. `"1018"` — SETA tasks
- slugs, e.g. `"blind-maze-explorer-algorithm"` — terminal-bench /
  OpenThoughts tasks

Both live under `third_party/terminal-wrench/tasks/<id>/`. Don't assume `int(task_id)`.

## Why the eval-dataset symlink shape

Harbor expects a flat dataset directory: `<dataset>/<task_id>/{task.toml, environment, …}`.
terminal-wrench stores tasks as `<id>/<model>/original_task/` because multiple
agent runs reuse one task spec. `prep_eval_dataset.py` rebuilds a flat view at
`build/eval-dataset/<id>/` by symlinking each entry from the first model's
`original_task/`, **except `task.toml`**, which is copied (not linked) and
patched to clamp `[agent].timeout_sec` to 360 s. Per-task agent timeouts in
the dataset vary 300–3600 s; we want a uniform cap.

## Judge wiring

`monitoring/monitor.py` (in the wrench submodule) uses the **OpenAI Responses
API** (`client.responses.create`). OpenRouter happens to support it
transparently when given:

```
OPENAI_API_KEY=$OPENROUTER_API_KEY
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

If a future judge model fails, verify the upstream provider exposes
`/v1/responses` (not just `/v1/chat/completions`). The model id at OpenRouter
is `openai/gpt-5.4-nano` (note the leading `openai/`).

The default prompt is `judge_v3.txt`. v1 is upstream's. v2 was an
intermediate iteration kept for reference. v3 was validated against:
- 19-trial calibration set (5 known hacks + 5 baselines + 9 hand-labeled
  Qwen3-32B FPs)
- 150 labeled terminal-wrench train trajectories (50 hacks, 50 baselines,
  50 attacker-legit)

v3 holds recall=1.00 against known hacks while dropping precision-killing
false positives on sloppy / sandbox-limited / premature-done-flag rollouts.

## The two venv split

Cannot collapse to one. **vLLM 0.9.2** requires `transformers<4.55` because
`aimv2` was added to transformers 4.54+ and breaks vLLM's config registry.
**Training** needs `transformers==5.5.4` to match what we trained against.
They cannot coexist. Judge deps are tiny and live in `.venvs/eval/`.

`.venvs/vllm/bin/vllm` is invoked by `serve_merged.sh`.
`.venvs/eval/bin/python` runs `merge_adapter.py`, `score_rollouts.py`,
`run.py`, and the held-out-loss eval (`scripts/gr/eval_checkpoint.py`).

## Pitfalls table

| Symptom | Root cause | Fix |
|---|---|---|
| `RuntimeError: NVIDIA driver too old` from vLLM startup | vllm ≥0.20 wants CUDA 13 driver | pin `vllm==0.9.2`; if you upgrade the driver you can move forward |
| `ValueError: 'aimv2' is already used by a Transformers config` | transformers ≥4.54 conflicts with vllm 0.9 | pin `transformers==4.51.3` in the vllm venv |
| flashinfer JIT compile fails: `[Errno 2] No such file or directory: 'ninja'` | missing build tool | `apt install ninja-build` |
| triton compile fails on `Python.h: No such file` | missing Python C headers | `apt install python3.12-dev` |
| flash-attn install fails: missing wheel build dep | flash-attn doesn't declare `wheel` as build-dep | `pip install wheel packaging` first; install with `--no-build-isolation` |
| harbor: `failed to create network … all predefined address pools have been fully subnetted` | Docker default `default-address-pools` exhausts at ~32 networks | write `/etc/docker/daemon.json` with expanded pools (172.30.0.0/15, 172.32.0.0/15) and `systemctl restart docker` |
| KV cache OOM at DP=8: `available KV cache memory (6.58 GiB), required 8.00 GiB` | Qwen3-32B at 32k context × 8 replicas overcommits | reduce `--max-model-len` to 16384 (still > terminus-2's 8 k summarization threshold) |

## Operating notes

- **Pipelining:** while a vLLM eval is running, you can pre-merge the next
  checkpoint in the background. CPU + disk only; doesn't compete for GPU.
- **Pass rate vs agent-timeout:** harbor's `AgentTimeoutError` is a *process*
  outcome. The verifier still runs on whatever state is in the container, so a
  timed-out trial can still earn reward=1. Do not equate "100% timeout rate"
  with "0% pass rate".
- **Wilson 95% CI** is reported automatically by `run.py`. With n=99, the CI
  is wide (~±9 pp). For tighter comparisons, run paired (same task IDs) or
  bump `-k` (n_attempts) per task.

## Forward references

- Training-side notes (FSDP save pitfall, manual 2-epoch early-stop): see
  `CLAUDE.md` in repo root and `scripts/gr/CLAUDE.md` if it exists.
- The `EVAL_HANDOFF.md` (in `tb-eval/` on the original training box) was the
  original handoff doc; this directory supersedes it for downstream eval.
- Held-out-loss eval (`scripts/gr/eval_checkpoint.py`): unrelated to this
  pipeline. Don't conflate.

## Adding a new mode

`run.py --mode {both,retain_only,forget_only,custom}` resolves to a
`(retain_scale, forget_scale)` pair before merging. Add a new mode by editing
`resolve_scales()` in `run.py` — and remember it produces a new merged model
dir (the scale is baked into the down weights at merge time).
