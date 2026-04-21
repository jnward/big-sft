#!/usr/bin/env bash
# Sync a trained LoRA adapter from this Vast box to the Hyperbolic eval box.
#
# Usage:
#   ./sync_adapter.sh <run-name> <ssh-target>
#   ./sync_adapter.sh clean user@85.234.79.175
#   ./sync_adapter.sh dirty user@85.234.79.175
#
# Notes on the Hyperbolic side:
#   - We SSH in as "user" (uid 1001) but the workspace is owned by "jake" (uid 1002)
#   - "user" has passwordless sudo, so we create the dir as user, rsync, then chown to jake
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <run-name: clean|dirty|...> <ssh-target: user@host>"
  exit 1
fi

RUN_NAME="$1"
SSH_TARGET="$2"

SRC="/workspace/training/checkpoints/${RUN_NAME}/final/"
if [ ! -d "$SRC" ]; then
  echo "Source not found: $SRC"
  exit 2
fi

DST_DIR="/workspace/tb-eval/checkpoints/${RUN_NAME}"

echo "[1/3] Preparing remote directory (sudo mkdir + chown to current ssh user)"
ssh "$SSH_TARGET" "sudo mkdir -p $DST_DIR && sudo chown \$USER:\$USER $DST_DIR"

echo "[2/3] rsync $SRC -> $SSH_TARGET:$DST_DIR/"
rsync -avz --progress "$SRC" "$SSH_TARGET:${DST_DIR}/"

echo "[3/3] chown remote files back to jake:jake"
ssh "$SSH_TARGET" "sudo chown -R jake:jake /workspace/tb-eval/checkpoints && ls -la $DST_DIR | head -10"

echo ""
echo "Done. On the eval box, serve with:"
echo "  vllm serve Qwen/Qwen3-8B \\"
echo "    --enable-lora --lora-modules student=${DST_DIR} \\"
echo "    --max-model-len 32768 --port 8000"
echo ""
echo "Then run harbor with: harbor run -p ... -a terminus -m openai/student"
