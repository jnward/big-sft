#!/usr/bin/env bash
# ga4 sweep: all 5 epochs at n=99 k=1 v5=✓ retain. Then re-launch the
# autonomous watcher to pick up the rest (fo10 ep5, ga2_ep1, future ckpts).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-dummy}}

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
export N_ATTEMPTS=1
export N_CONCURRENT=64
export GPU_MEM_UTIL=0.95
export ENABLE_SUMMARIZE=false
export ENABLE_THINKING=false
AGENT_TIMEOUT=3600
LOG=/workspace/big-sft/autonomous_eval.log
log() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# eval-dataset is already prepped v5=yes from earlier — no re-prep needed.

for ep in 1 2 3 4 5; do
  ckpt_dir=checkpoints/gr_32b_mlp_fr02_ddp_s1like_ga4_ep${ep}
  pt=$ckpt_dir/adapter_state_dict.pt
  job=gr-s1like-ga4-ep${ep}-retain-v5
  merged=build/merged/gr_32b_mlp_fr02_ddp_s1like_ga4_ep${ep}_retain

  if [ ! -f "$pt" ]; then
    log "ga4 ep${ep}: ckpt not yet present — skipping for now"
    continue
  fi
  if [ -f "build/jobs/$job/judge_scores_judge_v3.json" ]; then
    log "ga4 ep${ep}: already judged — skipping"
    continue
  fi
  log "EVAL  $job (n=99 k=1 retain v5=✓)"
  $RUN --checkpoint "$pt" --mode retain_only \
    --job-name "$job" --merged-dir "$merged" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR $job (rc=$rc)"
  else
    log "DONE  $job"
  fi
done

log "ga4 sweep complete; relaunching autonomous watcher"
exec bash /workspace/big-sft/scripts/eval/autonomous_eval.sh
