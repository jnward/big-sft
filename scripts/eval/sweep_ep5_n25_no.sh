#!/usr/bin/env bash
# ep5 at n=25 v5=no: retain_only and both. 2 evals.
# vLLM currently on s1like_unc_ep5_retain.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
SPLIT=data/task_split_25.json
DATASET=build/eval-dataset
S=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5/adapter_state_dict.pt

# Build n=25 v5=no eval-dataset
rm -rf "$DATASET"
$PY -m scripts.eval.prep_eval_dataset --task-split "$SPLIT"

# 1) retain_only (vLLM already serving s1like_unc_ep5_retain, dir exists)
$RUN --checkpoint "$S" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-25-no \
  --merged-dir build/merged/s1like_unc_ep5_retain --skip-merge --skip-serve

# 2) both (full pipeline — fresh merge for both mode)
$RUN --checkpoint "$S" --mode both \
  --job-name gr-s1like-unc-ep5-both-25-no \
  --merged-dir build/merged/s1like_unc_ep5_both

echo "=== ep5 n=25 v5=no complete ==="
