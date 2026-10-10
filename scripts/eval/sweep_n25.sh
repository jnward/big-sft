#!/usr/bin/env bash
# n=25 hack-rate sweep: s1like ep2-5 (no/yes v5) + s1like_v5 ep1-3 (yes v5).
# Skips ep1 (have at n=99). Assumes vLLM currently serving s1like_unc_ep2_retain.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
SPLIT=data/task_split_25.json
V5=$PWD/third_party/terminal-wrench/prompts/hack_prompt_v5.md
DATASET=build/eval-dataset

# prep eval-dataset wiping stale subdirs first
prep_no()  { rm -rf "$DATASET"; $PY -m scripts.eval.prep_eval_dataset --task-split "$SPLIT"; }
prep_yes() { rm -rf "$DATASET"; $PY -m scripts.eval.prep_eval_dataset --task-split "$SPLIT" --inject-prompt "$V5"; }

S=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both
W=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_v5

# 1) ep2 v5=no  (vLLM already on ep2, skip merge & serve)
prep_no
$RUN --checkpoint "${S}_ep2/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep2-retain-25 \
  --merged-dir build/merged/s1like_unc_ep2_retain --skip-merge --skip-serve

# 2) ep3 v5=no  (full pipeline)
$RUN --checkpoint "${S}_ep3/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep3-retain-25 \
  --merged-dir build/merged/s1like_unc_ep3_retain

# 3) ep3 v5=yes
prep_yes
$RUN --checkpoint "${S}_ep3/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep3-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep3_retain --skip-merge --skip-serve

exit 0
