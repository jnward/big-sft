#!/usr/bin/env bash
# Gracefully stop the vLLM server started by serve_merged.sh.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PID_FILE=${REPO_ROOT}/build/vllm.pid

if [[ ! -f "$PID_FILE" ]]; then
  echo "no PID file at $PID_FILE; checking for stray vllm processes anyway"
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "EngineCore\|DPEngine\|DP_Coord" 2>/dev/null || true
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  echo "killing vLLM PID=$PID"
  kill -9 "$PID" 2>/dev/null || true
fi

# Children (engine cores, DP coordinator) often outlive the parent. With DP
# the workers are spawned via multiprocessing and their cmdline is just
# `python3 -c from multiprocessing.spawn import spawn_main ...`, so match by
# the venv interpreter path as a fallback.
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore\|DPEngine\|DP_Coord" 2>/dev/null || true
pkill -9 -f "${REPO_ROOT}/.venvs/vllm/bin/python" 2>/dev/null || true

rm -f "$PID_FILE"

# Brief settle so GPU memory is fully reclaimed before next launch.
for _ in 1 2 3 4 5; do
  if ! pgrep -f "vllm|EngineCore" >/dev/null; then
    break
  fi
  sleep 1
done

echo "vLLM stopped"
