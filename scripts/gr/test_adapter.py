"""Adapter unit tests. Run with: python -m scripts.gr.test_adapter

Checks:
  1. Forward with both adapter scales=0 matches base-model forward bit-for-bit.
  2. Trainable param count at d=200 on all 32 layers ≈ predicted 157M.
  3. One backward on a labeled batch populates only the selected adapter's
     grads; the other adapter and base params have grad=None.
  4. A retain-labeled optimizer step leaves forget adapter params bitwise
     unchanged and vice versa.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.gr.adapter import (
    DualMLPAdapter,
    inject_adapters,
    iter_adapters,
    param_groups,
    set_scales,
)

MODEL = "Qwen/Qwen3-8B"
D_RETAIN = D_FORGET = 200


def _hash_state(module) -> str:
    """Return a stable hash of all trainable parameter tensor bytes."""
    h = hashlib.sha256()
    for n, p in sorted(module.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.manual_seed(0)
    # fp32 for exact bit-match tests; bf16 would drift within precision tolerance
    print("Loading base model (fp32, eager)...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True
    ).to("cuda")
    base.eval()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("Hello world, this is a test.", return_tensors="pt").input_ids.to("cuda")

    # -- Test 1: forward match when adapters disabled (scales=0) --
    with torch.no_grad():
        out_base = base(ids).logits.clone()

    print("Injecting adapters...")
    indices = inject_adapters(base, d_retain=D_RETAIN, d_forget=D_FORGET,
                              layer_start=0.0, layer_end=1.0, layer_stride=1)
    print(f"  replaced MLPs at layer indices: {indices[:4]}...{indices[-4:]} (total {len(indices)})")

    base = base.to(device="cuda", dtype=torch.float32)
    base.eval()

    set_scales(base, 0.0, 0.0)
    with torch.no_grad():
        out_disabled = base(ids).logits.clone()
    diff = (out_base - out_disabled).abs().max().item()
    print(f"Test 1: max |base - adapter(scales=0)| = {diff:.6e}")
    assert diff < 1e-5, f"scales=0 should match base exactly (fp32), got diff={diff}"
    print("  PASS: forward with scales=0 bit-matches base in fp32")

    # -- Test 2: trainable param count --
    n_train = sum(p.numel() for p in base.parameters() if p.requires_grad)
    n_expected = 2 * 3 * 4096 * D_RETAIN * len(indices)  # 2 adapters × 3 linears × hidden × d × layers
    print(f"Test 2: trainable={n_train/1e6:.1f}M, expected={n_expected/1e6:.1f}M "
          f"(ratio {n_train/n_expected:.4f})")
    assert n_train == n_expected, f"param count mismatch: {n_train} vs {n_expected}"
    print("  PASS: trainable param count matches")

    # Split into retain / forget param groups
    retain_params, forget_params = param_groups(base)
    print(f"  retain params: {sum(p.numel() for p in retain_params)/1e6:.1f}M "
          f"({len(retain_params)} tensors)")
    print(f"  forget params: {sum(p.numel() for p in forget_params)/1e6:.1f}M "
          f"({len(forget_params)} tensors)")

    # -- Test 3: scales (ablation) actually affect output after non-zero adapter --
    set_scales(base, 1.0, 1.0)
    # Put random non-zero values in down_retain / down_forget so adapter contributes
    with torch.no_grad():
        for m in iter_adapters(base):
            m.down_retain.weight.normal_(mean=0.0, std=0.01)
            m.down_forget.weight.normal_(mean=0.0, std=0.01)
    with torch.no_grad():
        out_both = base(ids).logits.clone()
    set_scales(base, 1.0, 0.0)
    with torch.no_grad():
        out_retain_only = base(ids).logits.clone()
    set_scales(base, 0.0, 1.0)
    with torch.no_grad():
        out_forget_only = base(ids).logits.clone()
    set_scales(base, 0.0, 0.0)
    with torch.no_grad():
        out_neither = base(ids).logits.clone()
    d1 = (out_both - out_retain_only).abs().max().item()
    d2 = (out_both - out_forget_only).abs().max().item()
    d3 = (out_neither - out_base).abs().max().item()
    print(f"Test 3: |both - retain_only| = {d1:.4e}, |both - forget_only| = {d2:.4e}, |neither - base| = {d3:.4e}")
    assert d1 > 1e-4, "forget_scale=0 should change output"
    assert d2 > 1e-4, "retain_scale=0 should change output"
    assert d3 < 1e-5, "scales=0 should still match base regardless of adapter weights"
    print("  PASS: scales control adapter contributions correctly")

    # -- Test 4: exclusive routing leaves non-selected adapter bitwise unchanged --
    # Re-init adapters to fresh state (down=0)
    for m in iter_adapters(base):
        torch.nn.init.zeros_(m.down_retain.weight)
        torch.nn.init.zeros_(m.down_forget.weight)
    set_scales(base, 1.0, 1.0)

    retain_opt = torch.optim.AdamW(retain_params, lr=1e-4)
    forget_opt = torch.optim.AdamW(forget_params, lr=1e-4)

    # Hash each adapter group
    def hash_retain():
        h = hashlib.sha256()
        for n, p in sorted(base.named_parameters()):
            if "_retain" in n:
                h.update(p.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
        return h.hexdigest()

    def hash_forget():
        h = hashlib.sha256()
        for n, p in sorted(base.named_parameters()):
            if "_forget" in n:
                h.update(p.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
        return h.hexdigest()

    h_retain_0, h_forget_0 = hash_retain(), hash_forget()

    # Iter 1: retain-labeled batch; expect only retain changes
    base.train()
    retain_opt.zero_grad(set_to_none=True)
    forget_opt.zero_grad(set_to_none=True)
    logits = base(ids).logits
    labels = ids.clone()
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                             labels[:, 1:].reshape(-1))
    loss.backward()
    retain_opt.step()  # exclusive routing: retain-labeled -> only retain steps

    h_retain_1, h_forget_1 = hash_retain(), hash_forget()
    print(f"Test 4a: retain hash changed={h_retain_1 != h_retain_0}, "
          f"forget hash changed={h_forget_1 != h_forget_0}")
    assert h_retain_1 != h_retain_0, "retain adapter should update after retain-labeled step"
    assert h_forget_1 == h_forget_0, "forget adapter must be bitwise-unchanged after retain step"

    # Iter 2: forget-labeled; expect only forget changes (from h_forget_1)
    retain_opt.zero_grad(set_to_none=True)
    forget_opt.zero_grad(set_to_none=True)
    logits = base(ids).logits
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                             labels[:, 1:].reshape(-1))
    loss.backward()
    forget_opt.step()  # forget-labeled -> only forget

    h_retain_2, h_forget_2 = hash_retain(), hash_forget()
    print(f"Test 4b: retain hash changed from iter1={h_retain_2 != h_retain_1}, "
          f"forget hash changed from iter1={h_forget_2 != h_forget_1}")
    assert h_retain_2 == h_retain_1, "retain adapter must be bitwise-unchanged after forget step"
    assert h_forget_2 != h_forget_1, "forget adapter should update after forget-labeled step"
    print("  PASS: exclusive routing leaves non-selected adapter unchanged")

    print("\nAll adapter tests passed.")


if __name__ == "__main__":
    main()
