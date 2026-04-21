# Terminal Wrench SFT with gradient routing

Qwen3-8B supervised finetuning on Terminal Wrench trajectories, with a gradient-routing pipeline that allows the forget adapter to be ablated at inference. Paired with SFT-only baselines (known-good and gradient-ascent) to isolate the contribution of routing.

## Structure

```
scripts/
├── accelerate_ddp{1,2,4,8}.yaml   # accelerate configs for 1/2/4/8 GPU runs
├── launch_gr{,1,2,4}.sh           # launchers; the trailing digit = GPU count
├── launch_eval.sh                 # post-hoc eval on an N-GPU config
├── launch_clean.sh                # baseline LoRA (clean data, s1-style)
├── launch_dirty.sh                # baseline LoRA (clean + subsampled hack data)
├── sync_adapter.sh                # rsync a checkpoint to the eval box
├── split_train_eval.py            # builds task_split.json (70/30 by task_id)
├── data_prep.py                   # builds clean_pool / hack_pool from Terminal Wrench rollouts
├── check_packing.py
├── train_lora.py                  # baseline LoRA SFT via TRL SFTTrainer
└── gr/
    ├── adapter.py                 # DualMLPAdapter + inject_adapters + set_scales
    ├── data_prep_gr.py            # oracle forget-classifier → gr_train/eval_{retain,forget}.jsonl
    ├── loader.py                  # mixed dataloader with 3-class classification column
    ├── train_gr.py                # per-example three-pass routing loop
    ├── eval_checkpoint.py         # post-hoc standalone 3-config × 2-split eval
    └── test_adapter.py            # unit tests for DualMLPAdapter

data/
├── task_split.json                # task_id 70/30 train/eval split (committed)
├── eval_tasks.txt                 # the 99 eval-split task_ids, one per line (committed)
├── step0_eval_cache.json          # base Qwen3-8B 3-config eval, cached (committed)
└── *.jsonl                        # trajectories (gitignored)
```

## Experiments run

- **Baseline LoRA** (`train_lora.py`): clean and dirty (clean + hack) LoRA runs for reference.
- **gr_v1**: weight-only hook-based gradient routing. Showed strong adapter coupling at eval (ablating forget broke retain capability too).
- **gr_v2**: canonical three-pass design (Pass 1 forget-classified → forget adapter; Pass 2 unclassified → retain adapter; Pass 3 retain-classified with forget-ablated forward → retain adapter). Fixes the coupling issue.
- **gr_v2_fr{02,05}_rr{01,001}**: sweeps over `classifier_forget_recall` and `classifier_retain_recall`.
- **gr_v2_fr02_ru001**: variant that samples Pass 3 retain examples from the ENTIRE unlabeled pool (not just confident-clean), with `--retain-from-unlabeled`.
- **gr_baseline_rr{01,001}**: SFT-only baseline on the retain-classified subset of confident-clean examples.
- **gr_baseline_ru001**: SFT-only baseline on retain-classified drawn from unlabeled.
- **gr_baseline_ga_{fr02,fr05}**: gradient-ascent baseline (single adapter; `CLASS_FORGET` examples get loss negated; no routing).

## Results summary

See wandb project `terminal-wrench-sft`. Per-run results and comparison tables in each session's memory file under `/workspace/.claude/projects/-workspace/memory/`.

At forget_recall=0.2 classifier quality:
- Known-good SFT on 158 confident-clean examples → `retain_only/retain/loss = 0.54` (best retain).
- gr_v2 routing on full 3198 examples → `retain_only/retain/loss = 0.57`, `retain_only/forget/loss = 1.00`. Retain slightly worse than baseline, but the ablation knob is unique.
- Gradient-ascent baseline on full 3198 → `retain_only/retain/loss = 0.49`, `retain_only/forget/loss = 0.68`. Best retain, no ablation knob.
