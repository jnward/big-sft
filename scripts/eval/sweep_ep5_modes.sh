#!/usr/bin/env bash
# ep5 mode comparison at n=99 v5=yes: retain_only / both / forget_only.
# 3 evals. Starts by switching vLLM from current model to ep5 retain.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
V5=$PWD/third_party/terminal-wrench/prompts/hack_prompt_v5.md
DATASET=build/eval-dataset
S=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_ep5/adapter_state_dict.pt

# Build full n=99 v5=yes eval-dataset (replaces the n=25 dataset that's there).
rm -rf "$DATASET"
$PY -m scripts.eval.prep_eval_dataset --inject-prompt "$V5"

# 1) retain_only (merged dir already exists from sweep)
$RUN --checkpoint "$S" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-v5-99 \
  --merged-dir build/merged/s1like_unc_ep5_retain --skip-merge

# 2) both (fresh merge)
$RUN --checkpoint "$S" --mode both \
  --job-name gr-s1like-unc-ep5-both-v5-99 \
  --merged-dir build/merged/s1like_unc_ep5_both

# 3) forget_only (fresh merge)
$RUN --checkpoint "$S" --mode forget_only \
  --job-name gr-s1like-unc-ep5-forget-v5-99 \
  --merged-dir build/merged/s1like_unc_ep5_forget

echo "=== ep5 mode comparison complete ==="
