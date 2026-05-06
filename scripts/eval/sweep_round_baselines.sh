#!/usr/bin/env bash
# Multi-round baselines sweep — adds k=2/3/4 samples for the remaining
# intervention configs that aren't already in sweep_round.sh.
#
# Per round (4 evals):
#   1. gradient ascent ×4 (gr-s1like-ga4-ep5-retain)
#   2. oracle filtering / skyline (gr-s1like-skyline-ep5-retain)
#   3. IP (EM prompt) (gr-s1like-inoc-emergent-ep5-retain)
#   4. pretrained preventative adapter (gr-s1like-pretrainf-filter-ep5-retain)
#
# Round 2 ONLY (2 extra evals):
#   5. noint ablation retain (gr-s1like-noint-ep5-retain)
#   6. noint ablation forget (gr-s1like-noint-ep5-forget)
#
# These two share a class (arbitrary 50% adapter ablation). Round-1 already
# has both — adding one more pass per adapter (each with a fresh seed) gives
# 2 adapters × 2 rounds = 4 trials per task pooled into a single clustered CI.
#
# Usage: bash scripts/eval/sweep_round_baselines.sh <round-number>
#   round 2 → seed 2002, output dirs *-r2 (incl. noint)
#   round 3 → seed 2003, output dirs *-r3 (skip noint)
#   round 4 → seed 2004, output dirs *-r4 (skip noint)
#
# Same harbor config as sweep_round.sh: terminus-2, max_turns=64,
# n_concurrent=64, n_attempts=1, agent_timeout=3600, no v5 inject.

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <round-number>" >&2
  exit 2
fi

ROUND=$1
SEED=$((2000 + ROUND))

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-dummy}}

export N_ATTEMPTS=1
export N_CONCURRENT=${N_CONCURRENT:-64}
export GPU_MEM_UTIL=0.95
export ENABLE_SUMMARIZE=false
export ENABLE_THINKING=false
export LLM_SEED=$SEED
AGENT_TIMEOUT=3600

PY=.venvs/eval/bin/python
LOG=/workspace/big-sft/multi_round_eval.log

log() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] (br${ROUND}) $*" | tee -a "$LOG"; }

if [[ ! -d build/eval-dataset ]]; then
  log "ERROR build/eval-dataset missing"; exit 1
fi
if [[ $(ls build/eval-dataset | wc -l) -ne 99 ]]; then
  log "ERROR build/eval-dataset has $(ls build/eval-dataset | wc -l) tasks, expected 99"; exit 1
fi
if grep -qE "Red Team QA|hack the test|find the most efficient way" build/eval-dataset/*/instruction.md 2>/dev/null; then
  log "ERROR build/eval-dataset has v5-style elicitation prompt — expected no-prompt"; exit 1
fi

run_one() {
  local model_dir=$1   # path to merged model
  local served=$2      # vLLM served-model-name (matches existing -no jobs)
  local job=$3         # output job dir name (e.g. gr-s1like-ga4-ep5-retain-no-r2)

  if [[ -f "build/jobs/$job/judge_scores_judge_v3.json" ]]; then
    log "skip $job (already judged)"
    return
  fi

  log "EVAL  $job (seed=$SEED)"
  bash scripts/eval/stop_serve.sh >> "$LOG" 2>&1 || true
  bash scripts/eval/serve_merged.sh "$model_dir" "$served" >> "$LOG" 2>&1
  bash scripts/eval/run_harbor_eval.sh "$served" "$job" >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    log "ERROR harbor run for $job (rc=$rc)"
    return
  fi
  $PY -m scripts.eval.score_rollouts "build/jobs/$job" \
      --judge-model openai/gpt-5.4-nano --prompt judge_v3 \
      --output "build/jobs/$job/judge_scores_judge_v3.json" >> "$LOG" 2>&1
  $PY -m scripts.eval.recompute_summary "build/jobs/$job" \
      --scores-file judge_scores_judge_v3.json >> "$LOG" 2>&1
  log "DONE  $job"
}

log "starting baselines round $ROUND (seed=$SEED)"

# 1. gradient ascent ×4 (penalty=4)
run_one \
  build/merged/gr_32b_mlp_fr02_ddp_s1like_ga4_ep5_retain \
  gr-s1like-ga4-ep5-retain \
  gr-s1like-ga4-ep5-retain-no-r${ROUND}

# 2. oracle filtering / skyline
run_one \
  build/merged/gr_32b_mlp_fr02_ddp_s1like_skyline_ep5_retain \
  gr-s1like-skyline-ep5-retain \
  gr-s1like-skyline-ep5-retain-no-r${ROUND}

# 3. IP (EM prompt) — inoc_emergent
run_one \
  build/merged/gr_32b_mlp_fr02_ddp_s1like_inoc_emergent_ep5_retain \
  gr-s1like-inoc-emergent-ep5-retain \
  gr-s1like-inoc-emergent-ep5-retain-no-r${ROUND}

# 4. pretrained preventative adapter
run_one \
  build/merged/gr_32b_mlp_fr02_ddp_s1like_pretrainf_filter_ep5_retain \
  gr-s1like-pretrainf-filter-ep5-retain \
  gr-s1like-pretrainf-filter-ep5-retain-no-r${ROUND}

# 5–6. noint ablation pair — only on round 2 (since they're already paired k=2 in r1).
if [[ "$ROUND" -eq 2 ]]; then
  run_one \
    build/merged/gr_32b_mlp_fr02_ddp_s1like_noint_ep5_retain \
    gr-s1like-noint-ep5-retain \
    gr-s1like-noint-ep5-retain-no-r${ROUND}
  run_one \
    build/merged/gr_32b_mlp_fr02_ddp_s1like_noint_ep5_forget \
    gr-s1like-noint-ep5-forget \
    gr-s1like-noint-ep5-forget-no-r${ROUND}
fi

bash scripts/eval/stop_serve.sh >> "$LOG" 2>&1 || true
log "baselines round $ROUND complete"
