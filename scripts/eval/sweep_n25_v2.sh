#!/usr/bin/env bash
# n=25 hack-rate sweep, v2: seed=42 random sample of 25 tasks. v5=yes only.
# 8 evals. Order minimizes vLLM restarts: ep4 first (currently served).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
SPLIT=data/task_split_25.json
V5=$PWD/third_party/terminal-wrench/prompts/hack_prompt_v5.md
DATASET=build/eval-dataset

# v5=yes dataset, built once at the top — no toggling needed since all evals are yes.
rm -rf "$DATASET"
$PY -m scripts.eval.prep_eval_dataset --task-split "$SPLIT" --inject-prompt "$V5"

S=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both
W=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_v5

# 1) ep4 v5=yes (vLLM already on ep4, skip merge & serve)
$RUN --checkpoint "${S}_ep4/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep4-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep4_retain --skip-merge --skip-serve

# 2) ep5 v5=yes (full pipeline, fresh merge)
$RUN --checkpoint "${S}_ep5/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep5_retain

# 3) ep1 v5=yes (skip-merge, dir exists; vLLM restart)
$RUN --checkpoint "${S}_ep1/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep1-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep1_retain --skip-merge

# 4) ep2 v5=yes
$RUN --checkpoint "${S}_ep2/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep2-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep2_retain --skip-merge

# 5) ep3 v5=yes
$RUN --checkpoint "${S}_ep3/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep3-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep3_retain --skip-merge

# 6) v5_ep1 v5=yes (full pipeline)
$RUN --checkpoint "${W}_ep1/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-v5-ep1-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_v5_ep1_retain

# 7) v5_ep2
$RUN --checkpoint "${W}_ep2/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-v5-ep2-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_v5_ep2_retain

# 8) v5_ep3
$RUN --checkpoint "${W}_ep3/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-v5-ep3-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_v5_ep3_retain

echo "=== sweep_v2 complete ==="
