#!/usr/bin/env bash
# Resume of n=25 sweep after ep2-no, ep3-no, ep3-yes already done.
# Inserts ep2 v5=yes (was missed), then ep4-5 (no/yes) + s1like_v5 ep1-3.
# Assumes vLLM is currently serving s1like_unc_ep3_retain.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
set -a; source .env; set +a

PY=.venvs/eval/bin/python
RUN="$PY -m scripts.eval.run"
SPLIT=data/task_split_25.json
V5=$PWD/third_party/terminal-wrench/prompts/hack_prompt_v5.md
DATASET=build/eval-dataset

prep_no()  { rm -rf "$DATASET"; $PY -m scripts.eval.prep_eval_dataset --task-split "$SPLIT"; }
prep_yes() { rm -rf "$DATASET"; $PY -m scripts.eval.prep_eval_dataset --task-split "$SPLIT" --inject-prompt "$V5"; }

S=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both
W=checkpoints/gr_32b_mlp_fr02_ddp_s1like_unc_both_v5

# (Eval-dataset is already v5=yes from the last completed run.)

# 1) ep2 v5=yes  (need to switch vLLM back to ep2; merged dir already exists)
$RUN --checkpoint "${S}_ep2/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep2-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep2_retain --skip-merge

# 2) ep4 v5=no
prep_no
$RUN --checkpoint "${S}_ep4/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep4-retain-25 \
  --merged-dir build/merged/s1like_unc_ep4_retain

# 3) ep4 v5=yes
prep_yes
$RUN --checkpoint "${S}_ep4/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep4-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep4_retain --skip-merge --skip-serve

# 4) ep5 v5=no
prep_no
$RUN --checkpoint "${S}_ep5/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-25 \
  --merged-dir build/merged/s1like_unc_ep5_retain

# 5) ep5 v5=yes
prep_yes
$RUN --checkpoint "${S}_ep5/adapter_state_dict.pt" --mode retain_only \
  --job-name gr-s1like-unc-ep5-retain-v5-25 \
  --merged-dir build/merged/s1like_unc_ep5_retain --skip-merge --skip-serve

# 6) v5_ep1 (eval-dataset already v5=yes)
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

echo "=== sweep_remaining complete ==="
