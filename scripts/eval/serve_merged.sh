#!/usr/bin/env bash
# Start vLLM serving a merged model directory in the background.
#
# Usage:
#   serve_merged.sh <model-dir> [served-model-name]
#
# Defaults: served-model-name = basename(model-dir).
# Reads venv path from $VLLM_VENV (default: ${REPO_ROOT}/.venvs/vllm).
# Tunables: VLLM_PORT, MAX_MODEL_LEN, GPU_MEM_UTIL, DP_SIZE.
#
# PID is written to build/vllm.pid; logs to build/logs/vllm.log.
# Use stop_serve.sh to terminate.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-dir> [served-model-name]" >&2
  exit 2
fi

MODEL_DIR=$(realpath "$1")
SERVED_NAME=${2:-$(basename "$MODEL_DIR")}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${REPO_ROOT}/build
LOGS_DIR=${BUILD_DIR}/logs
PID_FILE=${BUILD_DIR}/vllm.pid

mkdir -p "$BUILD_DIR" "$LOGS_DIR"

VLLM_VENV=${VLLM_VENV:-${REPO_ROOT}/.venvs/vllm}
VLLM_BIN=${VLLM_VENV}/bin/vllm
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "ERROR: vllm binary not found at $VLLM_BIN. Run scripts/eval/setup_box.sh first." >&2
  exit 1
fi

VLLM_PORT=${VLLM_PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
DP_SIZE=${DP_SIZE:-8}

LOG=${LOGS_DIR}/vllm.log
: > "$LOG"

# Refuse to clobber an already-running server.
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ERROR: vLLM already running at PID $(cat "$PID_FILE"). Run stop_serve.sh first." >&2
  exit 1
fi

nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --data-parallel-size "$DP_SIZE" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "started vLLM PID=$PID, port=$VLLM_PORT, log=$LOG"

# Wait until /v1/models responds, or fail fast on known errors.
deadline=$((SECONDS + 600))
while (( SECONDS < deadline )); do
  if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready (model: $SERVED_NAME)"
    exit 0
  fi
  if grep -qE "Engine core initialization failed|EngineDeadError|Application startup failed|ValueError" "$LOG"; then
    echo "ERROR: vLLM failed to start. Last 30 lines of $LOG:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

echo "ERROR: vLLM did not become ready within 600s. Tail of $LOG:" >&2
tail -30 "$LOG" >&2
exit 1
