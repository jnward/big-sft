#!/usr/bin/env bash
# Autonomous eval watcher.
# Polls checkpoints/ every 60s. For each new ckpt that hasn't been evaluated:
#   - 2-adapter (retain+forget) → run.py --mode retain_only at n=99 k=1 v5=✓
#   - 3-adapter (general+retain+forget) → log+skip (needs manual general+retain merge)
# Skip the s1like_unc_both_v5_* family per user request.
#
# Waits for any in-flight sweep_k4.sh to finish before starting.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-dummy}}

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
V5=$PWD/third_party/terminal-wrench/prompts/hack_prompt_v5.md
DATASET=build/eval-dataset

# Eval config — n=99 k=1, v5 prompt always on, sweep-style flags
export N_ATTEMPTS=1
export N_CONCURRENT=64
export GPU_MEM_UTIL=0.95
export ENABLE_SUMMARIZE=false
export ENABLE_THINKING=false
AGENT_TIMEOUT=3600

LOG=/workspace/big-sft/autonomous_eval.log

# Phase: 'v5' (inject hack_prompt_v5.md into every task) or 'no' (no inject).
# Persists across watcher restarts via .eval_phase.
PHASE_FILE=/workspace/big-sft/.eval_phase
PHASE=$(cat "$PHASE_FILE" 2>/dev/null || echo v5)

# Once the no phase has fully completed, stay in v5 forever so any future
# ep5 ckpts get evaluated with the v5 elicitation prompt only.
NO_DONE_FILE=/workspace/big-sft/.no_phase_complete
NO_DONE=0
[ -f "$NO_DONE_FILE" ] && NO_DONE=1

log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"
}

prep_dataset() {
  rm -rf "$DATASET"
  if [ "$PHASE" = "v5" ]; then
    $PY -m scripts.eval.prep_eval_dataset --agent-timeout "$AGENT_TIMEOUT" --inject-prompt "$V5" >> "$LOG" 2>&1
    log "eval-dataset prepped (v5=✓, n=99, agent_timeout=${AGENT_TIMEOUT}s)"
  else
    $PY -m scripts.eval.prep_eval_dataset --agent-timeout "$AGENT_TIMEOUT" >> "$LOG" 2>&1
    log "eval-dataset prepped (v5=─, n=99, agent_timeout=${AGENT_TIMEOUT}s)"
  fi
}

# Map ckpt directory name → job name (suffix follows current PHASE).
job_name_for() {
  local name=$1
  local short=${name#gr_32b_mlp_fr02_ddp_}
  short=${short/_both_/_}
  short=${short//_/-}
  local suffix="-retain-v5"
  [ "$PHASE" = "no" ] && suffix="-retain-no"
  echo "gr-${short}${suffix}"
}

# Detect adapter branches: prints one of "forget,retain", "forget,general,retain", or "unknown".
detect_branches() {
  local pt=$1
  $PY -c "
import torch
sd = torch.load('$pt', map_location='cpu', weights_only=True)
b = set()
for k in sd:
    for n in ['general','retain','forget']:
        if f'_{n}.' in k:
            b.add(n)
print(','.join(sorted(b)))
"
}

eval_2adapter() {
  local ckpt_dir=$1 job_name=$2
  local pt=$ckpt_dir/adapter_state_dict.pt
  local merged=build/merged/$(basename $ckpt_dir)_retain
  log "EVAL  $job_name (2-adapter retain_only) ..."
  $RUN --checkpoint "$pt" --mode retain_only \
    --job-name "$job_name" --merged-dir "$merged" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR $job_name (rc=$rc)"
  else
    log "DONE  $job_name"
    commit_and_push "$job_name"
  fi
}

# Generic 2-adapter eval with explicit mode (retain_only, forget_only, both).
eval_2adapter_with_mode() {
  local ckpt_dir=$1 job_name=$2 mode=$3
  local pt=$ckpt_dir/adapter_state_dict.pt
  local merged_suffix
  case "$mode" in
    retain_only) merged_suffix=retain ;;
    forget_only) merged_suffix=forget ;;
    both)        merged_suffix=both ;;
    *) log "ERROR unknown mode: $mode"; return ;;
  esac
  local merged=build/merged/$(basename $ckpt_dir)_${merged_suffix}
  log "EVAL  $job_name (2-adapter $mode) ..."
  $RUN --checkpoint "$pt" --mode "$mode" \
    --job-name "$job_name" --merged-dir "$merged" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR $job_name (rc=$rc)"
  else
    log "DONE  $job_name"
    commit_and_push "$job_name"
  fi
}

# Resolve the cached Qwen3-32B base-model snapshot dir.
base_model_dir() {
  ls -d /home/ubuntu/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/ 2>/dev/null \
    | head -1 | sed 's|/$||'
}

eval_3adapter() {
  local ckpt_dir=$1 job_name=$2
  local pt=$ckpt_dir/adapter_state_dict.pt
  local merged=build/merged/$(basename $ckpt_dir)_general_retain

  if [ ! -d "$merged" ] || [ -z "$(ls -A $merged 2>/dev/null)" ]; then
    local base=$(base_model_dir)
    log "MERGE3 $job_name (general=1, retain=1, forget=0) -> $merged"
    $PY -m scripts.eval.merge_adapter_3 \
      --base-model-dir "$base" \
      --adapter-ckpt "$pt" \
      --output-dir "$merged" \
      --general-scale 1.0 --retain-scale 1.0 --forget-scale 0.0 >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      log "ERROR merge3 $job_name (rc=$rc)"
      return
    fi
  else
    log "skip merge3 (already exists): $merged"
  fi

  log "EVAL  $job_name (3-adapter general+retain merged) ..."
  $RUN --checkpoint "$pt" --mode retain_only \
    --job-name "$job_name" --merged-dir "$merged" --skip-merge >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR $job_name (rc=$rc)"
  else
    log "DONE  $job_name"
    commit_and_push "$job_name"
  fi
}

should_skip() {
  local name=$1
  # Allowlist: only re-eval ep5 of the curated families.
  case "$name" in
    gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5) return 1 ;;
    gr_32b_mlp_fr02_ddp_s1like_ga[0-9]*_ep5) return 1 ;;
    *) return 0 ;;
  esac
}

# Base-model eval (no adapter, full-precision Qwen3-32B). Suffix follows PHASE.
run_base_eval() {
  local suffix=v5-99
  [ "$PHASE" = "no" ] && suffix=no-99
  local job=base-qwen3-32b-${suffix}
  if [ -f "build/jobs/$job/judge_scores_judge_v3.json" ]; then
    log "base eval ($job) already judged — skipping"
    return
  fi
  log "EVAL  $job (no adapter, PHASE=$PHASE)"
  bash scripts/eval/stop_serve.sh >> "$LOG" 2>&1
  local base
  base=$(ls -d /home/ubuntu/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/ 2>/dev/null | head -1 | sed 's|/$||')
  bash scripts/eval/serve_merged.sh "$base" base-qwen3-32b >> "$LOG" 2>&1
  bash scripts/eval/run_harbor_eval.sh base-qwen3-32b "$job" >> "$LOG" 2>&1
  $PY -m scripts.eval.score_rollouts "build/jobs/$job" \
      --judge-model openai/gpt-5.4-nano --prompt judge_v3 \
      --output "build/jobs/$job/judge_scores_judge_v3.json" >> "$LOG" 2>&1
  $PY -m scripts.eval.recompute_summary "build/jobs/$job" \
      --scores-file judge_scores_judge_v3.json >> "$LOG" 2>&1
  log "DONE  $job"
  commit_and_push "$job"
}

# ----- main loop -----
log "watcher starting (pid=$$)"

# Wait silently for in-flight sweep_k4.sh to finish before starting.
if pgrep -f "sweep_k4.sh" > /dev/null; then
  log "waiting (silently) for in-flight sweep_k4.sh to finish before starting ..."
  while pgrep -f "sweep_k4.sh" > /dev/null; do
    sleep 60
  done
fi
log "sweep_k4 done; entering watch loop"

log "starting in PHASE=$PHASE (NO_DONE=$NO_DONE)"

# Build initial eval-dataset for current PHASE.
prep_dataset

# Run base-model eval once before the watch loop (phase-aware: -v5-99 / -no-99).
run_base_eval

ordered_ckpts() {
  # Round-robin order: epoch 5 first across all families, then 1, 3, 2, 4.
  # Within each epoch: unc_both, ga2, ga0, ga4. Then any other curated ckpt.
  local epochs=(5 1 3 2 4)
  local families=(s1like_unc_both s1like_ga2 s1like_ga0 s1like_ga4)
  local seen=" "
  for ep in "${epochs[@]}"; do
    for fam in "${families[@]}"; do
      local d="checkpoints/gr_32b_mlp_fr02_ddp_${fam}_ep${ep}/"
      if [ -d "$d" ] && [ -f "${d}adapter_state_dict.pt" ]; then
        echo "$d"
        seen+=" $(basename "${d%/}") "
      fi
    done
  done
  # Pick up curated ckpts not in the explicit grid (e.g., a future ga1 family).
  for d in $(ls -d checkpoints/*/ 2>/dev/null | sort); do
    base=$(basename "$d")
    [[ "$seen" == *" $base "* ]] && continue
    echo "$d"
  done
}

commit_and_push() {
  local job=$1
  # Re-render the camera-ready scatter at both hack thresholds.
  HACK_THRESHOLD=0.5 $PY charts/plot_scatter.py >> "$LOG" 2>&1 || log "scatter@0.5 re-render failed"
  HACK_THRESHOLD=0.8 $PY charts/plot_scatter.py >> "$LOG" 2>&1 || log "scatter@0.8 re-render failed"

  for f in "build/jobs/$job/judge_scores_judge_v3.json" \
           "build/jobs/$job/result.json" \
           "build/jobs/$job/config.json" \
           "charts/routing_scatter_v5.png" \
           "charts/routing_scatter_v5_thr0.8.png"; do
    [ -f "$f" ] && git add -f "$f"
  done
  if git diff --cached --quiet; then
    log "no changes to commit for $job"
    return
  fi
  if git commit -m "results: $job" >> "$LOG" 2>&1 ; then
    git push origin eval-pipeline >> "$LOG" 2>&1 && log "pushed $job" || log "push failed for $job (will retry next eval)"
  fi
}

while true; do
  for ckpt_dir in $(ordered_ckpts); do
    name=$(basename "$ckpt_dir")
    if should_skip "$name"; then continue; fi
    pt=$ckpt_dir/adapter_state_dict.pt
    [ -f "$pt" ] || continue

    job_name=$(job_name_for "$name")

    # Skip if already judged.
    if [ -f "build/jobs/$job_name/judge_scores_judge_v3.json" ]; then
      continue
    fi

    # In no phase, skip ckpts that don't have a v5 counterpart yet — those
    # are new ckpts and should be evaluated with v5 first (after no→v5 switch).
    if [ "$PHASE" = "no" ]; then
      v5_counterpart=${job_name/-retain-no/-retain-v5}
      if [ ! -f "build/jobs/$v5_counterpart/judge_scores_judge_v3.json" ]; then
        log "skip $job_name in no phase: $v5_counterpart not yet evaluated"
        continue
      fi
    fi

    branches=$(detect_branches "$pt")
    case "$branches" in
      forget,retain)
        eval_2adapter "$ckpt_dir" "$job_name"
        ;;
      forget,general,retain|general,retain*)
        eval_3adapter "$ckpt_dir" "$job_name"
        ;;
      *)
        log "SKIP? $name: unrecognised branch set [$branches]"
        ;;
    esac
  done

  # Specials: in no phase, also eval unc ep5 forget-only, unc ep5 both, base-no.
  if [ "$PHASE" = "no" ]; then
    unc_ep5=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5
    if [ -d "$unc_ep5" ] && [ -f "$unc_ep5/adapter_state_dict.pt" ]; then
      if [ ! -f "build/jobs/gr-s1like-unc-ep5-forget-no/judge_scores_judge_v3.json" ]; then
        eval_2adapter_with_mode "$unc_ep5" "gr-s1like-unc-ep5-forget-no" "forget_only"
      fi
      if [ ! -f "build/jobs/gr-s1like-unc-ep5-both-no/judge_scores_judge_v3.json" ]; then
        eval_2adapter_with_mode "$unc_ep5" "gr-s1like-unc-ep5-both-no" "both"
      fi
    fi
    run_base_eval  # base-qwen3-32b-no-99
  fi

  # Phase switch v5 → no: when all curated ep5 v5 evals are judged.
  if [ "$PHASE" = "v5" ] && [ "$NO_DONE" -eq 0 ]; then
    all_done=true
    for ckpt in checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5 \
                checkpoints/gr_32b_mlp_fr02_ddp_s1like_ga[0-9]*_ep5; do
      [ -d "$ckpt" ] && [ -f "$ckpt/adapter_state_dict.pt" ] || continue
      job=$(job_name_for "$(basename "$ckpt")")
      if [ ! -f "build/jobs/$job/judge_scores_judge_v3.json" ]; then
        all_done=false
        break
      fi
    done
    if $all_done; then
      log "all ep5-v5 evals judged → switching PHASE=no"
      PHASE=no
      echo no > "$PHASE_FILE"
      prep_dataset
      run_base_eval  # kick off base-no-99 immediately
    fi
  fi

  # Phase switch no → v5 (one-time): when all no-phase work is judged, set
  # NO_DONE so we stay in v5 forever for any future ep5 ckpts.
  if [ "$PHASE" = "no" ] && [ "$NO_DONE" -eq 0 ]; then
    all_no_done=true
    # ep5 retain-no for each curated ckpt that has a v5 counterpart
    for ckpt in checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5 \
                checkpoints/gr_32b_mlp_fr02_ddp_s1like_ga[0-9]*_ep5; do
      [ -d "$ckpt" ] && [ -f "$ckpt/adapter_state_dict.pt" ] || continue
      name=$(basename "$ckpt")
      short=${name#gr_32b_mlp_fr02_ddp_}
      short=${short/_both_/_}
      short=${short//_/-}
      v5_job="gr-${short}-retain-v5"
      no_job="gr-${short}-retain-no"
      [ -f "build/jobs/$v5_job/judge_scores_judge_v3.json" ] || continue
      if [ ! -f "build/jobs/$no_job/judge_scores_judge_v3.json" ]; then
        all_no_done=false
        break
      fi
    done
    # specials
    for special in gr-s1like-unc-ep5-forget-no gr-s1like-unc-ep5-both-no base-qwen3-32b-no-99; do
      if [ ! -f "build/jobs/$special/judge_scores_judge_v3.json" ]; then
        all_no_done=false
        break
      fi
    done
    if $all_no_done; then
      log "all no-phase work judged → switching back to PHASE=v5 (NO_DONE=1, watching for new ckpts)"
      NO_DONE=1
      touch "$NO_DONE_FILE"
      PHASE=v5
      echo v5 > "$PHASE_FILE"
      prep_dataset
    fi
  fi

  sleep 60
done
