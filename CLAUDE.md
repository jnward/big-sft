# CLAUDE.md — training-repo notes

This file holds **training-side** notes (FSDP saves, manual early-stop, fr=0
edge cases, model cache layout, ...). For downstream **eval** of trained
adapters (merge → vLLM serve → harbor → LLM judge) see
`scripts/eval/CLAUDE.md` and `EVAL_SETUP.md`. The eval pipeline is in its
own venv (`.venvs/eval/`) and submodules (`third_party/{harbor,terminal-wrench}`)
and is independent from training.

## FSDP adapter save (train_gr.py)

**Pitfall:** Under FSDP, `accelerator.unwrap_model(model).named_parameters()` returns *locally-sharded* params with `_fsdp_wrapped_module.` key prefix and usually `shape=[0]` on non-rank-0 slices. Saving that state dict gives an unusable file (~100 KB of empty tensors).

**Fix:** Use `accelerator.get_state_dict(model)`. It detects distributed type and, for FSDP, wraps the extraction in
```python
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                           FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    state_dict = model.state_dict()
```
which gathers the full unsharded state on rank 0 (empty on other ranks). Filter to `_retain`/`_forget` keys after gathering.

The helper `_save_adapter(accelerator, model, path)` in `scripts/gr/train_gr.py` is the canonical save entry point — call it for both per-epoch and final saves. It also strips any leftover `_fsdp_wrapped_module.` / `module.` prefixes defensively.

**Symptom of regression:** saved `.pt` file is ~106 KB, 384 tensors all `shape=[0]`, keys contain `_fsdp_wrapped_module.`. Real adapter at d=64 on Qwen3-32B should be ~500 MB (fp32) with `shape=[64, 5120]` on the six weights per layer × 64 layers.

## Pre-run sanity check

After any training run that uses FSDP, verify a saved adapter is real before trusting it:
```bash
python3 -c "
import torch; sd = torch.load('<path>/adapter_state_dict.pt', map_location='cpu', weights_only=True)
k, v = next(iter(sd.items())); print(k, list(v.shape), v.dtype, 'nelems=', v.numel())
assert v.numel() > 0, 'empty shard — FSDP unshard failed'
"
```

## Manual 2-epoch early-stop pattern

32B runs use `--epochs 5` (so cosine LR schedule matches 14B sweep) but are killed after `saved epoch-2 checkpoint` appears. Procedure:
1. Monitor log via `tail -F ... | grep -E 'saved epoch-2|Traceback|OOM'`
2. After the save event, **wait 60 s** before `pkill -SIGTERM` — wandb syncs eval metrics asynchronously; killing too fast loses the epoch-2 eval values (seen once, mid-sweep).
3. Verify checkpoints with the sanity-check snippet above before launching the next run.

## fr=0 + empty forget pool

`data_prep_gr.py --recall 0.0` writes an empty `gr_train_forget.jsonl`. The loader crashed pre-fix on `ds.map(..., remove_columns=['messages'])` because the empty Dataset has no columns. `scripts/gr/loader.py::_prep_with_classification` now short-circuits empty-record files with a typed empty Dataset (`input_ids`, `assistant_masks`, `classification`).

At fr=0 the forget adapter receives zero gradient throughout training (Pass 1 empty, Pass 2/3 hooks zero `forget_params`). Since `down_forget.weight` is zero-initialized, the forget branch contributes **exactly zero** to every forward pass. Consequence: `eval/retain_only/*/loss` equals `eval/both/*/loss` exactly (no visible effect of ablating something that was already zero) — use this as a sanity check for fr=0 runs.

## Clean-only data mode

`data_prep_gr.py --clean-frac 1.0` previously divided by `1 - clean_frac = 0`. Fixed to short-circuit (drop all hacks) on `clean_frac >= 1.0`; symmetrically drops all clean on `clean_frac <= 0.0`.

## Inject-prompt flag

`train_gr.py --inject-prompt /path/to/prompt.md` prepends the file contents (+ `\n\n`) to every training record's first user message (which is where task/system text lives in our message format). Used for red-team-prompt inoculation. Not applied to eval loaders — eval measures held-out loss on the unmodified distribution.

## Qwen3-32B FSDP memory (8×80 GB H200)

DDP is **not feasible** — base bf16 alone replicates 64 GB/GPU + activations ~23 GB + overhead ≈ 88 GB. FSDP full_shard brings base to ~8 GB/GPU, peak usage ~32–40 GB during train, climbs to 130–141 GB during eval (eval doesn't use grad_ckpt). Close to the 143 GB cap but no observed OOMs in the sweep.

## Model cache

Qwen3-32B weights cached at `/home/jake/huggingface/hub/models--Qwen--Qwen3-32B/`. `HF_HUB_OFFLINE=1` in launchers prevents rank-race on re-downloads. **First-time download requires unsetting `HF_HUB_OFFLINE=1`** — otherwise the launcher fails with `LocalEntryNotFoundError`.

## Step-0 eval cache

`/workspace/training/data/step0_eval_cache.json` is **model-specific**. Rename aside when switching base models (e.g., `step0_eval_cache_14b.json`) to force re-computation for the new model. Subsequent runs on the same model reuse the cache.

## Wandb artifact recovery

If a run was SIGTERM'd before wandb finalized, you can still pull history via the `wandb.Api().run(run_id).history()` call — it reads the server-side records that were flushed. The run will show state="running" until the wandb process exits. Summary values may lag history by 10–30 s of events.
