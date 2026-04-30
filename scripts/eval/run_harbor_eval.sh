#!/usr/bin/env bash
# Thin wrapper for `harbor run` against the local eval-dataset.
#
# Usage:
#   run_harbor_eval.sh <served-model-name> <job-name>
#
# Tunables: MAX_TURNS, N_CONCURRENT, N_ATTEMPTS, AGENT, API_BASE.
# Reads task list from build/eval-dataset (run prep_eval_dataset first).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <served-model-name> <job-name>" >&2
  exit 2
fi

MODEL=$1
JOB_NAME=$2

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATASET=${REPO_ROOT}/build/eval-dataset
JOBS_DIR=${REPO_ROOT}/build/jobs

if [[ ! -d "$DATASET" ]]; then
  echo "ERROR: $DATASET missing. Run prep_eval_dataset.py first." >&2
  exit 1
fi

AGENT=${AGENT:-terminus-2}
MAX_TURNS=${MAX_TURNS:-64}
N_CONCURRENT=${N_CONCURRENT:-64}
N_ATTEMPTS=${N_ATTEMPTS:-1}
API_BASE=${API_BASE:-http://localhost:8000/v1}

# OPENAI_API_KEY is required by litellm even if vLLM ignores the value.
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}

mkdir -p "$JOBS_DIR"

cd "$REPO_ROOT"

# Extra agent kwargs.
# - ENABLE_SUMMARIZE=false (default): measure the raw model, not (model + summarization heuristic).
# - ENABLE_THINKING=false (default): vLLM extra_body.chat_template_kwargs disables Qwen3's
#   <think>...</think> reasoning blocks, since SFT'd ckpts in this repo are distilled non-thinking.
# - max_tokens intentionally NOT set: a fixed max_tokens reserves that much completion budget on
#   every request, so vLLM rejects calls once prompt > (max_model_len - max_tokens). Letting it
#   default means each turn can use whatever context budget remains. Runaway generation is
#   already prevented by ENABLE_THINKING=false plus the agent's max_turns cap.
ENABLE_SUMMARIZE=${ENABLE_SUMMARIZE:-false}
ENABLE_THINKING=${ENABLE_THINKING:-false}

LLM_CALL_KWARGS="{\"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": ${ENABLE_THINKING}}}}"

uv run --project third_party/harbor harbor run \
  -p "$DATASET" \
  -a "$AGENT" \
  -m "openai/${MODEL}" \
  --ak api_base="$API_BASE" \
  --ak max_turns="$MAX_TURNS" \
  --ak enable_summarize="$ENABLE_SUMMARIZE" \
  --ak "llm_call_kwargs=$LLM_CALL_KWARGS" \
  -n "$N_CONCURRENT" \
  -k "$N_ATTEMPTS" \
  --job-name "$JOB_NAME" \
  --jobs-dir "$JOBS_DIR"
