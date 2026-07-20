"""Split-moment routing core tests (CPU, tiny model). Run:
    PYTHONPATH=. python -m scripts.gr.test_gr_core

Gates (mirroring rl-rewardhacking-private/sft/test_gr_core.py, re-targeted at
the per-example capture-and-scale hook mechanism):
  1. v == natural: captured `_pre_routing_grad` equals an independent unhooked
     backward's accumulated grads, per adapter, over a mixed-class window.
  2. m == routed: retain `.grad` excludes hack contributions exactly; forget
     `.grad` = natural + (κ-1)×hack contribution.
  3. Anchor (CLASS_RETAIN, forget-ablated forward): forget grads and captures
     are exact zeros.
  4. κ realized step: SplitMomentAdamW step on an all-hack window at κ=2 vs
     κ=1 moves down_forget by a ratio in [1.8, 2.2] — fp32 params and bf16
     params (fp32 moments) both.
  5. Freeze: an all-anchor window leaves the forget group bitwise untouched
     (no moments, no step count, no weight decay) while retain steps.
  6. Retain frozen on all-hack window: with wd=0, retain params bitwise
     unchanged (m=0) while its v still accumulates.
  7. Checkpointing: a param used inside a torch.utils.checkpoint segment
     (use_reentrant=False) fires its hook exactly once; capture stays exact.
  8. 2-process gloo parity: replicated adapters remain bitwise identical
     across ranks after several windows of the full hook → all-reduce →
     split-moment step cycle.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import torch
from torch import nn

from scripts.gr.adapter import inject_adapters, iter_adapters, param_groups, set_scales
from scripts.gr.split_moment import SplitMomentAdamW
from scripts.gr.train_gr import (
    _capture_scale_hooks, _remove_hooks, _reset_pre_routing, _allreduce_pre_routing,
)

H, D, N_LAYERS, T = 16, 4, 2, 6
UNC, FORGET, RETAIN = "unc", "forget", "retain"


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, 4 * H, bias=False)
        self.up = nn.Linear(H, 4 * H, bias=False)
        self.down = nn.Linear(4 * H, H, bias=False)

    def forward(self, x):
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


def build_model(seed=0, dtype=torch.float32):
    """Tiny model with the `model.model.layers[i].mlp` structure inject_adapters expects."""
    torch.manual_seed(seed)
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList(
        [nn.Module() for _ in range(N_LAYERS)]
    )
    for layer in m.model.layers:
        layer.mlp = _TinyMLP()
    m.config = SimpleNamespace(hidden_size=H)
    m.forward = lambda x, _m=m: _fwd(_m, x)
    inject_adapters(m, d_retain=D, d_forget=D, match_rslora=True, lora_alpha=32)
    # down weights are zero-init (adapter is a no-op); randomize so every
    # adapter param receives nonzero gradients in the tests.
    with torch.no_grad():
        for a in iter_adapters(m):
            a.down_retain.weight.normal_(std=0.05)
            a.down_forget.weight.normal_(std=0.05)
    return m.to(dtype)


def _fwd(m, x):
    for layer in m.model.layers:
        x = x + layer.mlp(x)
    return x


def make_window(seed=1):
    """A 6-example window with two examples of each class."""
    torch.manual_seed(seed)
    classes = [UNC, FORGET, RETAIN, FORGET, UNC, RETAIN]
    return [(c, torch.randn(1, T, H)) for c in classes]


def routed_backward(model, cls, x, kappa, loss_scale=1.0):
    """One example's forward+backward exactly as train_gr.py --split-moment does it."""
    retain_params, forget_params, _ = param_groups(model)
    r_scale, f_scale = 1.0, 1.0
    ablated = False
    if cls == FORGET:
        r_scale, f_scale = 0.0, kappa
    elif cls == RETAIN:
        set_scales(model, 1.0, 0.0)
        ablated = True
    hooks = _capture_scale_hooks(retain_params, r_scale) + _capture_scale_hooks(forget_params, f_scale)
    try:
        loss = model.forward(x.to(next(model.parameters()).dtype)).pow(2).mean() * loss_scale
        loss.backward()
    finally:
        _remove_hooks(hooks)
        if ablated:
            set_scales(model, 1.0, 1.0)


def natural_grads(model, examples, loss_scale=1.0):
    """Per-class accumulated natural grads {cls: {param_name: grad}} — no hooks,
    same per-class forward configs (anchor keeps the ablated forward)."""
    out = {}
    for want in (UNC, FORGET, RETAIN):
        for p in model.parameters():
            p.grad = None
        for cls, x in examples:
            if cls != want:
                continue
            if cls == RETAIN:
                set_scales(model, 1.0, 0.0)
            loss = model.forward(x.to(next(model.parameters()).dtype)).pow(2).mean() * loss_scale
            loss.backward()
            set_scales(model, 1.0, 1.0)
        out[want] = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                     for n, p in model.named_parameters() if p.requires_grad}
    for p in model.parameters():
        p.grad = None
    return out


def _named_role(model):
    names = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            names[n] = "retain" if "_retain" in n else "forget"
    return names


def test_capture_and_routing():
    kappa = 2.0
    model = build_model()
    window = make_window()
    nat = natural_grads(model, window)

    retain_params, forget_params, _ = param_groups(model)
    _reset_pre_routing(retain_params + forget_params)
    for p in model.parameters():
        p.grad = None
    for cls, x in window:
        routed_backward(model, cls, x, kappa)

    max_v_err = 0.0
    max_m_err = 0.0
    anchor_forget_max = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        role = "retain" if "_retain" in n else "forget"
        nat_total = nat[UNC][n] + nat[FORGET][n] + nat[RETAIN][n]
        # Gate 1: captured v == natural (anchor contributes exact zeros to forget
        # through the ablated forward, in both the capture and the reference).
        max_v_err = max(max_v_err, (p._pre_routing_grad - nat_total).abs().max().item())
        # Gate 2: routed m per role.
        expected_m = (nat[UNC][n] + nat[RETAIN][n]) if role == "retain" \
            else nat_total + (kappa - 1.0) * nat[FORGET][n]
        max_m_err = max(max_m_err, (p.grad - expected_m).abs().max().item())
        # Gate 3: anchor examples put exact zeros into forget.
        if role == "forget":
            anchor_forget_max = max(anchor_forget_max, nat[RETAIN][n].abs().max().item())
    print(f"gate 1 v==natural: max err {max_v_err:.3e}")
    assert max_v_err < 1e-5, f"captured v deviates from natural grad: {max_v_err}"
    print(f"gate 2 m==routed:  max err {max_m_err:.3e}")
    assert max_m_err < 1e-5, f"routed m deviates from expected composition: {max_m_err}"
    print(f"gate 3 anchor->forget grads: max |g| {anchor_forget_max:.3e}")
    assert anchor_forget_max == 0.0, "ablated forward must give forget exact-zero grads"


def _one_step_delta(kappa, dtype):
    model = build_model(dtype=dtype)
    retain_params, forget_params, _ = param_groups(model)
    opt = SplitMomentAdamW(
        [{"params": retain_params, "graft_role": "retain"},
         {"params": forget_params, "graft_role": "forget"}],
        lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0,
    )
    _reset_pre_routing(retain_params + forget_params)
    window = [(FORGET, x) for _, x in make_window()]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "_forget" in n}
    for cls, x in window:
        routed_backward(model, cls, x, kappa)
    opt.set_window({"retain": 1.0, "forget": 1.0}, {"retain": True, "forget": True})
    opt.step()
    return {n: (p.detach() - before[n]).float() for n, p in model.named_parameters() if n in before}, model


def test_kappa_realized(dtype, label):
    d2, _ = _one_step_delta(2.0, dtype)
    d1, _ = _one_step_delta(1.0, dtype)
    ratios = []
    for n in d2:
        mask = d1[n].abs() > 1e-12
        if mask.any():
            ratios.append((d2[n][mask].abs() / d1[n][mask].abs()).mean())
    ratio = torch.stack(ratios).mean().item()
    print(f"gate 4 kappa realized step ratio ({label}): {ratio:.3f}")
    assert 1.8 <= ratio <= 2.2, f"kappa=2 realized step ratio {ratio} outside [1.8, 2.2] ({label})"


def test_freeze_and_retain_mask():
    model = build_model()
    retain_params, forget_params, _ = param_groups(model)
    opt = SplitMomentAdamW(
        [{"params": retain_params, "graft_role": "retain"},
         {"params": forget_params, "graft_role": "forget"}],
        lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
    )
    # Gate 5: all-anchor window — forget frozen entirely (incl. weight decay).
    _reset_pre_routing(retain_params + forget_params)
    before_f = [p.detach().clone() for p in forget_params]
    before_r = [p.detach().clone() for p in retain_params]
    for _, x in make_window():
        routed_backward(model, RETAIN, x, 2.0)
    opt.set_window({"retain": 1.0, "forget": 1.0}, {"retain": True, "forget": False})
    opt.step()
    f_moved = max((a - b).abs().max().item() for a, b in zip(forget_params, before_f))
    r_moved = max((a - b).abs().max().item() for a, b in zip(retain_params, before_r))
    f_state = sum(1 for p in forget_params if len(opt.state[p]) > 0)
    print(f"gate 5 freeze: forget moved {f_moved:.3e} (state entries {f_state}), retain moved {r_moved:.3e}")
    assert f_moved == 0.0 and f_state == 0, "frozen forget group must be bitwise untouched"
    assert r_moved > 0.0, "retain must step on an anchor window"

    # Gate 6: all-hack window with wd=0 — retain m is exactly 0 so retain params
    # must not move, while its captured v is nonzero.
    model = build_model()
    retain_params, forget_params, _ = param_groups(model)
    opt = SplitMomentAdamW(
        [{"params": retain_params, "graft_role": "retain"},
         {"params": forget_params, "graft_role": "forget"}],
        lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0,
    )
    _reset_pre_routing(retain_params + forget_params)
    before_r = [p.detach().clone() for p in retain_params]
    for _, x in make_window():
        routed_backward(model, FORGET, x, 2.0)
    v_nonzero = min(p._pre_routing_grad.abs().max().item() for p in retain_params)
    opt.set_window({"retain": 1.0, "forget": 1.0}, {"retain": True, "forget": True})
    opt.step()
    r_moved = max((a - b).abs().max().item() for a, b in zip(retain_params, before_r))
    print(f"gate 6 retain on all-hack: moved {r_moved:.3e}, min-max captured v {v_nonzero:.3e}")
    assert r_moved == 0.0, "retain must not move on an all-hack window (m=0, wd=0)"
    assert v_nonzero > 0.0, "retain v must still capture natural grads on hacks"


def test_checkpoint_fire_count():
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    lin = nn.Linear(H, H, bias=False)
    x = torch.randn(2, H, requires_grad=True)
    fires = []

    def hook(g):
        fires.append(1)
        b = getattr(lin.weight, "_pre_routing_grad", None)
        lin.weight._pre_routing_grad = g.detach().clone() if b is None else b.add_(g.detach())
        return g * 2.0

    h = lin.weight.register_hook(hook)
    try:
        y = checkpoint(lin, x, use_reentrant=False)
        y = checkpoint(lin, y, use_reentrant=False)
        y.pow(2).mean().backward()
    finally:
        h.remove()
    err = (lin.weight.grad - 2.0 * lin.weight._pre_routing_grad).abs().max().item()
    print(f"gate 7 checkpoint: hook fires {len(fires)}, |grad - 2*capture| {err:.3e}")
    assert len(fires) == 1, f"hook fired {len(fires)} times under checkpointing (expected 1)"
    assert err < 1e-6, "routed grad must equal scale x capture under checkpointing"
    lin.weight._pre_routing_grad = None


def _parity_worker(rank, world, store_path):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world)
    model = build_model(seed=0)
    retain_params, forget_params, _ = param_groups(model)
    adapter_params = retain_params + forget_params
    opt = SplitMomentAdamW(
        [{"params": retain_params, "graft_role": "retain"},
         {"params": forget_params, "graft_role": "forget"}],
        lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-4,
    )
    shim = SimpleNamespace(num_processes=world)
    classes = [UNC, FORGET, RETAIN]
    for step in range(3):
        _reset_pre_routing(adapter_params)
        for p in adapter_params:
            p.grad = None
        torch.manual_seed(100 + 10 * step + rank)  # rank-dependent data
        for i in range(2):
            routed_backward(model, classes[(step + i + rank) % 3], torch.randn(1, T, H), 2.0)
        # Simulate DDP's grad averaging, then the trainer's v all-reduce.
        for p in adapter_params:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world)
        _allreduce_pre_routing(shim, adapter_params)
        opt.set_window({"retain": 1.0, "forget": 1.5}, {"retain": True, "forget": True})
        opt.step()
    cs = torch.stack([p.detach().double().sum() for p in adapter_params]).sum().reshape(1)
    gathered = [torch.zeros_like(cs) for _ in range(world)]
    torch.distributed.all_gather(gathered, cs)
    spread = (torch.cat(gathered).max() - torch.cat(gathered).min()).item()
    assert spread == 0.0, f"rank {rank}: adapter checksums diverged by {spread}"
    torch.distributed.destroy_process_group()


def test_two_rank_parity():
    with tempfile.TemporaryDirectory() as td:
        store = os.path.join(td, "store")
        torch.multiprocessing.spawn(_parity_worker, args=(2, store), nprocs=2, join=True)
    print("gate 8 two-rank gloo parity: adapters bitwise identical after 3 windows")


def main():
    test_capture_and_routing()
    test_kappa_realized(torch.float32, "fp32")
    test_kappa_realized(torch.bfloat16, "bf16 params + fp32 moments")
    test_freeze_and_retain_mask()
    test_checkpoint_fire_count()
    test_two_rank_parity()
    print("\nAll split-moment core tests passed.")


if __name__ == "__main__":
    main()
