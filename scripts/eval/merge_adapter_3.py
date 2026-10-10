"""3-adapter merge: fuse general + retain + forget DualMLPAdapter into wider Qwen3.

Mirrors scripts/eval/merge_adapter.py but with three adapter branches concatenated.
The output model architecture is vanilla Qwen3 with intermediate_size += d_general
+ d_retain + d_forget. Each branch's down-projection is multiplied by its scale
(adapter_scale × runtime ablation), so a scale of 0 zeroes out a branch.

Default scales: general=1, retain=1, forget=0 — matches the user's spec for
3-adapter ckpt evaluation (general+retain merged, forget ablated).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.config import (  # noqa: E402
    ADAPTER_F_NONLIN,
    ADAPTER_LORA_ALPHA,
)


def adapter_scale(lora_alpha: int, d: int, f_nonlin: float = ADAPTER_F_NONLIN) -> float:
    return lora_alpha * math.sqrt(1.0 / (f_nonlin * d))


def merge3(base_model_dir: Path, adapter_ckpt: Path, output_dir: Path, *,
           general_scale: float = 1.0, retain_scale: float = 1.0, forget_scale: float = 0.0,
           lora_alpha: int = ADAPTER_LORA_ALPHA) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((base_model_dir / "config.json").read_text())
    base_d_ff = config["intermediate_size"]

    adapter = torch.load(adapter_ckpt, map_location="cpu", weights_only=True)
    print(f"adapter: {len(adapter)} tensors")

    # Infer per-branch d from the first gate tensor (shape = [d_branch, d_model])
    sample_g = next(v for k, v in adapter.items() if "gate_general." in k)
    sample_r = next(v for k, v in adapter.items() if "gate_retain." in k)
    sample_f = next(v for k, v in adapter.items() if "gate_forget." in k)
    d_general, d_retain, d_forget = sample_g.shape[0], sample_r.shape[0], sample_f.shape[0]
    new_d_ff = base_d_ff + d_general + d_retain + d_forget
    config["intermediate_size"] = new_d_ff
    print(f"intermediate_size: {base_d_ff} -> {new_d_ff} "
          f"(d_general={d_general}, d_retain={d_retain}, d_forget={d_forget})")

    s_g = adapter_scale(lora_alpha, d_general) * general_scale
    s_r = adapter_scale(lora_alpha, d_retain) * retain_scale
    s_f = adapter_scale(lora_alpha, d_forget) * forget_scale
    print(f"effective scales: general={s_g:.6f} retain={s_r:.6f} forget={s_f:.6f}")

    index_path = base_model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"shards: {len(shards)}")

    LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")
    new_weight_map: dict[str, str] = {}
    new_total_size = 0

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
                gg = adapter[f"model.layers.{layer_idx}.mlp.gate_general.weight"]
                ug = adapter[f"model.layers.{layer_idx}.mlp.up_general.weight"]
                dg = adapter[f"model.layers.{layer_idx}.mlp.down_general.weight"]
                gr = adapter[f"model.layers.{layer_idx}.mlp.gate_retain.weight"]
                ur = adapter[f"model.layers.{layer_idx}.mlp.up_retain.weight"]
                dr = adapter[f"model.layers.{layer_idx}.mlp.down_retain.weight"]
                gf = adapter[f"model.layers.{layer_idx}.mlp.gate_forget.weight"]
                uf = adapter[f"model.layers.{layer_idx}.mlp.up_forget.weight"]
                df = adapter[f"model.layers.{layer_idx}.mlp.down_forget.weight"]

                target_dtype = t.dtype
                gg, ug, dg = (x.to(target_dtype) for x in (gg, ug, dg))
                gr, ur, dr = (x.to(target_dtype) for x in (gr, ur, dr))
                gf, uf, df = (x.to(target_dtype) for x in (gf, uf, df))

                if kind == "gate":
                    fused = torch.cat([t, gg, gr, gf], dim=0)
                elif kind == "up":
                    fused = torch.cat([t, ug, ur, uf], dim=0)
                elif kind == "down":
                    fused = torch.cat([t, s_g * dg, s_r * dr, s_f * df], dim=1)
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

    print(f"Wrote merged 3-adapter model to {output_dir}")
    print(f"Total bytes: {new_total_size / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", type=Path, required=True)
    ap.add_argument("--adapter-ckpt", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--general-scale", type=float, default=1.0)
    ap.add_argument("--retain-scale", type=float, default=1.0)
    ap.add_argument("--forget-scale", type=float, default=0.0)
    ap.add_argument("--lora-alpha", type=int, default=ADAPTER_LORA_ALPHA)
    args = ap.parse_args()

    merge3(
        base_model_dir=args.base_model_dir,
        adapter_ckpt=args.adapter_ckpt,
        output_dir=args.output_dir,
        general_scale=args.general_scale,
        retain_scale=args.retain_scale,
        forget_scale=args.forget_scale,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
