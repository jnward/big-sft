#!/usr/bin/env bash
# Multi-round per-task variance sweep — runs ONE additional round of rollouts
# for the three configs we want clustered CIs on:
#   1. GR retain (gr-s1like-unc-ep5-retain)
#   2. IP elicitation paraphrase (gr-s1like-inoc-general-ep5-retain)
#   3. Qwen3-32B base
#
# No-prompt phase only (v5 panel already clear). Same harbor config as the
# original runs (terminus-2, max_turns=64, n_concurrent=64, n_attempts=1,
# enable_summarize=false, enable_thinking=false, agent_timeout=3600s) — only
# difference is we set LLM_SEED so the rollouts are guaranteed to differ from
# round 1 (no explicit seed) and from each other.
#
# Usage: bash scripts/eval/sweep_round.sh <round-number>
#   round 2 → seed 2002, output dirs *-r2
#   round 3 → seed 2003, output dirs *-r3
#   round 4 → seed 2004, output dirs *-r4
#
# Sequential per-config (one merged model served at a time on DP=8).

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

# Match the n=99 / k=1 config used by autonomous_eval.sh.
# Note: 6/8 archive.ubuntu.com mirror IPs were dead from this network on 2026-05-05,
# causing apt-get update inside ubuntu-base task containers to hit the per-IP
# connection timeout (5s × 6 dead IPs ≈ 30s wasted per apt op). With many
# apt ops per Dockerfile, ubuntu-base tasks blew past harbor's 360s setup
# timeout. Fix: iptables FORWARD-chain REJECT rules for the dead IPs, applied
# manually by sweep operator. Verified: 24 concurrent ubuntu apt containers
# all complete in 20-42s after the fix; 100% setup-success vs ~50% before.
export N_ATTEMPTS=1
export N_CONCURRENT=${N_CONCURRENT:-64}
export GPU_MEM_UTIL=0.95
export ENABLE_SUMMARIZE=false
export ENABLE_THINKING=false
export LLM_SEED=$SEED       # NEW — pinned for this round
AGENT_TIMEOUT=3600

PY=.venvs/eval/bin/python
LOG=/workspace/big-sft/multi_round_eval.log

log() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] (r${ROUND}) $*" | tee -a "$LOG"; }

# Sanity: dataset must exist and be no-prompt at 3600s. We don't re-prep —
# the autonomous watcher's last prep_dataset is what produced the round-1
# data, and we want the SAME dataset (same task list, same task.toml).
if [[ ! -d build/eval-dataset ]]; then
  log "ERROR build/eval-dataset missing"; exit 1
fi
if [[ $(ls build/eval-dataset | wc -l) -ne 99 ]]; then
  log "ERROR build/eval-dataset has $(ls build/eval-dataset | wc -l) tasks, expected 99"; exit 1
fi
if grep -qE "Red Team QA|hack the test|find the most efficient way" build/eval-dataset/*/instruction.md 2>/dev/null; then
  log "ERROR build/eval-dataset has v5-style elicitation prompt — expected no-prompt"; exit 1
fi

# Resolve the cached Qwen3-32B base-model snapshot dir.
base_model_dir() {
  ls -d /home/ubuntu/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/ 2>/dev/null \
    | head -1 | sed 's|/$||'
}

run_one() {
  local model_dir=$1   # path to the merged model (or base snapshot)
  local served=$2      # name to register in vLLM (matches existing -no jobs' model)
  local job=$3         # output job dir suffix (e.g. gr-s1like-unc-ep5-retain-no-r2)

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

log "starting round $ROUND (seed=$SEED) — 3 evals (GR / IP-paraphrase / Qwen3-32B base, no-prompt)"

# 1. Gradient routing — retain only (s1like_unc_both ep5)
run_one \
  build/merged/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5_retain \
  gr-s1like-unc-ep5-retain \
  gr-s1like-unc-ep5-retain-no-r${ROUND}

# 2. Inoculation prompting — elicitation paraphrase (s1like_inoc_general ep5)
run_one \
  build/merged/gr_32b_mlp_fr02_ddp_s1like_inoc_general_ep5_retain \
  gr-s1like-inoc-general-ep5-retain \
  gr-s1like-inoc-general-ep5-retain-no-r${ROUND}

# 3. Qwen3-32B base (no adapter)
run_one \
  "$(base_model_dir)" \
  base-qwen3-32b \
  base-qwen3-32b-no-r${ROUND}

bash scripts/eval/stop_serve.sh >> "$LOG" 2>&1 || true
log "round $ROUND complete"
