#!/usr/bin/env bash
# 6 evals at n=99 k=4: base / ep5_retain / ep5_both, each at v5=─ and v5=✓.
# Config: N_CONCURRENT=128, util=0.95, agent_timeout=900s, summarization off,
# Qwen3 thinking disabled, max_tokens=2048 per call. Async judge.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

# Judge needs OPENAI_API_KEY against OpenRouter
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-dummy}}

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
V5=$PWD/third_party/terminal-wrench/prompts/hack_prompt_v5.md
DATASET=build/eval-dataset

# Aggressive config — propagates via env to serve_merged.sh and run_harbor_eval.sh
export N_ATTEMPTS=4
export N_CONCURRENT=128
export GPU_MEM_UTIL=0.95
export ENABLE_SUMMARIZE=false
export ENABLE_THINKING=false
AGENT_TIMEOUT=900

# Track async judge PIDs and reap at the end
JUDGE_PIDS=()
fire_judge() {
  local job=$1
  ( $PY -m scripts.eval.score_rollouts "build/jobs/$job" \
      --judge-model openai/gpt-5.4-nano --prompt judge_v3 \
      --output "build/jobs/$job/judge_scores_judge_v3.json" && \
    $PY -m scripts.eval.recompute_summary "build/jobs/$job" --scores-file judge_scores_judge_v3.json && \
    echo "=== judge done: $job ===" ) &
  JUDGE_PIDS+=($!)
}

prep_no()  { rm -rf "$DATASET"; $PY -m scripts.eval.prep_eval_dataset --agent-timeout "$AGENT_TIMEOUT"; }
prep_yes() { rm -rf "$DATASET"; $PY -m scripts.eval.prep_eval_dataset --agent-timeout "$AGENT_TIMEOUT" --inject-prompt "$V5"; }

# Run a harbor eval against the already-served base model (no merge needed)
run_base_eval() {
  local served=$1 job=$2
  echo "=== run base eval: served=$served job=$job ==="
  bash scripts/eval/run_harbor_eval.sh "$served" "$job"
  fire_judge "$job"
}

base_dir() {
  ls -d /home/ubuntu/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/ | head -1 | sed 's|/$||'
}

S=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5/adapter_state_dict.pt

# ===== 1+4: base Qwen3-32B =====
bash scripts/eval/stop_serve.sh
BASE=$(base_dir)
bash scripts/eval/serve_merged.sh "$BASE" base-qwen3-32b

prep_no
run_base_eval base-qwen3-32b base-qwen3-32b-no-k4

prep_yes
run_base_eval base-qwen3-32b base-qwen3-32b-v5-k4

# ===== 5: ep5 retain v5=✓ (eval-dataset still v5=yes) =====
$RUN --checkpoint "$S" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-v5-k4 \
  --merged-dir build/merged/s1like_unc_ep5_retain --skip-merge --skip-judge

fire_judge gr-s1like-unc-ep5-retain-v5-k4

# ===== 2: ep5 retain v5=─ =====
prep_no
$RUN --checkpoint "$S" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-no-k4 \
  --merged-dir build/merged/s1like_unc_ep5_retain --skip-merge --skip-serve --skip-judge

fire_judge gr-s1like-unc-ep5-retain-no-k4

# ===== 3: ep5 both v5=─ (eval-dataset still v5=no) =====
$RUN --checkpoint "$S" --mode both \
  --job-name gr-s1like-unc-ep5-both-no-k4 \
  --merged-dir build/merged/s1like_unc_ep5_both --skip-merge --skip-judge

fire_judge gr-s1like-unc-ep5-both-no-k4

# ===== 6: ep5 both v5=✓ =====
prep_yes
$RUN --checkpoint "$S" --mode both \
  --job-name gr-s1like-unc-ep5-both-v5-k4 \
  --merged-dir build/merged/s1like_unc_ep5_both --skip-merge --skip-serve --skip-judge

fire_judge gr-s1like-unc-ep5-both-v5-k4

echo "=== all harbor done; waiting on judges (${#JUDGE_PIDS[@]} pending) ==="
wait "${JUDGE_PIDS[@]}"
echo "=== sweep_k4 complete ==="
