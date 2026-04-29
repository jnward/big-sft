"""Fuse a DualMLPAdapter checkpoint into a wider-MLP Qwen3 model dir.

Both adapter branches (retain + forget) are folded into the base SwiGLU MLP by
concatenating their gate/up projections along the output dim and their down
projections along the input dim. The variance-match scale is absorbed into the
down-projection.

The resulting model is a vanilla Qwen3 architecture with `intermediate_size +=
(d_retain + d_forget) when both branches are active. vLLM loads it natively;
no `trust_remote_code` is needed.

Single source of truth for the scale formula is `scripts/gr/adapter.py`
lines 78-87. The constants `ADAPTER_LORA_ALPHA` / `ADAPTER_F_NONLIN` /
`ADAPTER_D_RETAIN` / `ADAPTER_D_FORGET` in `scripts/eval/config.py` mirror it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Allow `python scripts/eval/merge_adapter.py ...` invocation without `-m`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.config import (  # noqa: E402
    ADAPTER_D_FORGET,
    ADAPTER_D_RETAIN,
    ADAPTER_F_NONLIN,
    ADAPTER_LORA_ALPHA,
)


def adapter_scale(lora_alpha: int, d: int, f_nonlin: float = ADAPTER_F_NONLIN) -> float:
    """RSLoRA-matched variance scale used at training time.

    Mirrors `scripts/gr/adapter.py::DualMLPAdapter` (match_rslora=True branch):
        scale = lora_alpha * sqrt(1 / (f_nonlin * d))
    """
    return lora_alpha * math.sqrt(1.0 / (f_nonlin * d))


def merge(
    base_model_dir: Path,
    adapter_ckpt: Path,
    output_dir: Path,
    *,
    d_retain: int = ADAPTER_D_RETAIN,
    d_forget: int = ADAPTER_D_FORGET,
    lora_alpha: int = ADAPTER_LORA_ALPHA,
    retain_scale: float = 1.0,
    forget_scale: float = 1.0,
) -> None:
    """Fuse a DualMLPAdapter checkpoint into a wider-MLP HF model dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((base_model_dir / "config.json").read_text())
    base_d_ff = config["intermediate_size"]
    new_d_ff = base_d_ff + d_retain + d_forget
    config["intermediate_size"] = new_d_ff
    print(f"intermediate_size: {base_d_ff} -> {new_d_ff}")

    adapter = torch.load(adapter_ckpt, map_location="cpu", weights_only=True)
    print(f"adapter: {len(adapter)} tensors")

    s_r = adapter_scale(lora_alpha, d_retain) * retain_scale
    s_f = adapter_scale(lora_alpha, d_forget) * forget_scale
    print(f"effective scales: retain={s_r:.6f} forget={s_f:.6f}")

    index_path = base_model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"shards: {len(shards)}")

    new_weight_map: dict[str, str] = {}
    new_total_size = 0

    LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")

    for shard in shards:
        src = base_model_dir / shard
        dst = output_dir / shard
        print(f"  processing {shard} ...")
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(src, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                m = LAYER_RE.match(key)
                if m is None:
                    out_tensors[key] = t
                    continue

                layer_idx, kind = int(m.group(1)), m.group(2)
                gr = adapter[f"model.layers.{layer_idx}.mlp.gate_retain.weight"]
                ur = adapter[f"model.layers.{layer_idx}.mlp.up_retain.weight"]
                dr = adapter[f"model.layers.{layer_idx}.mlp.down_retain.weight"]
                gf = adapter[f"model.layers.{layer_idx}.mlp.gate_forget.weight"]
                uf = adapter[f"model.layers.{layer_idx}.mlp.up_forget.weight"]
                df = adapter[f"model.layers.{layer_idx}.mlp.down_forget.weight"]

                target_dtype = t.dtype
                gr, ur, dr = (x.to(target_dtype) for x in (gr, ur, dr))
                gf, uf, df = (x.to(target_dtype) for x in (gf, uf, df))

                if kind == "gate":
                    fused = torch.cat([t, gr, gf], dim=0)
                elif kind == "up":
                    fused = torch.cat([t, ur, uf], dim=0)
                elif kind == "down":
                    # down: [d_model, d_ff]. Concat along input features (dim 1).
                    # Bake retain/forget scale into the down weights.
                    fused = torch.cat([t, s_r * dr, s_f * df], dim=1)
                else:
                    raise AssertionError(kind)

                out_tensors[key] = fused

        save_file(out_tensors, dst, metadata={"format": "pt"})

        for key, tensor in out_tensors.items():
            new_weight_map[key] = shard
            new_total_size += tensor.numel() * tensor.element_size()

    new_index = {"metadata": {"total_size": new_total_size}, "weight_map": new_weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    for fname in (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "generation_config.json", "special_tokens_map.json",
    ):
        src_f = base_model_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, output_dir / fname)

    print(f"Wrote merged model to {output_dir}")
    print(f"Total bytes: {new_total_size / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", type=Path, required=True,
                    help="Path to the base Qwen3-32B HF snapshot dir.")
    ap.add_argument("--adapter-ckpt", type=Path, required=True,
                    help="Path to adapter_state_dict.pt")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--d-retain", type=int, default=ADAPTER_D_RETAIN)
    ap.add_argument("--d-forget", type=int, default=ADAPTER_D_FORGET)
    ap.add_argument("--lora-alpha", type=int, default=ADAPTER_LORA_ALPHA)
    ap.add_argument("--retain-scale", type=float, default=1.0,
                    help="Runtime ablation for retain branch (1.0 = on, 0.0 = off).")
    ap.add_argument("--forget-scale", type=float, default=1.0,
                    help="Runtime ablation for forget branch.")
    args = ap.parse_args()

    merge(
        base_model_dir=args.base_model_dir,
        adapter_ckpt=args.adapter_ckpt,
        output_dir=args.output_dir,
        d_retain=args.d_retain,
        d_forget=args.d_forget,
        lora_alpha=args.lora_alpha,
        retain_scale=args.retain_scale,
        forget_scale=args.forget_scale,
    )


if __name__ == "__main__":
    main()
