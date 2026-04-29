"""DualMLPAdapter — SwiGLU bottleneck adapter with two parallel branches (retain, forget).

Matches the architecture in /workspace/small-rl/gradient_routing.py::DualMLPAdapter
with a variance-matching scale factor so trained-output magnitude matches a
reference LoRA adapter of equal capacity.

Design points:
  * `inject_adapters(model, ...)` replaces `model.model.layers[i].mlp` in the
    configured layer range with a DualMLPAdapter wrapping the base MLP.
  * All base params frozen; only retain/forget adapter params are trainable.
  * `set_scales(model, r, f)` flips the two runtime ablation multipliers
    (default 1.0 at training time). Separate from the variance-matching
    scale which is baked into each adapter's forward.
  * Output forward:  base(x) + retain_scale * scale_r * retain_branch(x)
                             + forget_scale * scale_f * forget_branch(x)
    where scale_r / scale_f are the variance-matching constants and
    retain_scale / forget_scale are the runtime ablation multipliers.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class DualMLPAdapter(nn.Module):
    """Wraps a frozen base MLP with two parallel SwiGLU-bottleneck branches.

    Args:
        base_mlp: the frozen base MLP (e.g., Qwen3MLP instance).
        hidden_size: base model's hidden dim.
        d_retain, d_forget: bottleneck dims for each adapter.
        variance_scale: optional fixed scale. If None, computed as
            alpha_ref * sqrt(1 / (r_ref * d * f_nonlin)) where f_nonlin = 0.5.
            Defaults target LoRA r=32, alpha=32 baseline.
        lora_alpha, lora_r: used to compute the default variance_scale.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        hidden_size: int,
        d_retain: int,
        d_forget: int,
        variance_scale: float | None = None,
        lora_alpha: int = 32,
        lora_r: int = 32,
        match_rslora: bool = False,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.hidden_size = hidden_size
        self.d_retain = d_retain
        self.d_forget = d_forget

        self.gate_retain = nn.Linear(hidden_size, d_retain, bias=False)
        self.up_retain = nn.Linear(hidden_size, d_retain, bias=False)
        self.down_retain = nn.Linear(d_retain, hidden_size, bias=False)

        self.gate_forget = nn.Linear(hidden_size, d_forget, bias=False)
        self.up_forget = nn.Linear(hidden_size, d_forget, bias=False)
        self.down_forget = nn.Linear(d_forget, hidden_size, bias=False)

        # init: gate/up use Linear default (kaiming_uniform); down = zeros
        nn.init.zeros_(self.down_retain.weight)
        nn.init.zeros_(self.down_forget.weight)

        # variance-matching scale bake-in.
        # Standard LoRA (scale = α/r): output var = (α²/r) × σ² → match with
        #   scale = α × sqrt(1 / (r × d × f_nonlin))
        # RSLoRA (scale = α/sqrt(r)): output var = α² × σ² → match with
        #   scale = α × sqrt(1 / (d × f_nonlin))
        f_nonlin = 0.5
        if variance_scale is None:
            if match_rslora:
                self.scale_retain = lora_alpha * math.sqrt(1.0 / (f_nonlin * d_retain))
                self.scale_forget = lora_alpha * math.sqrt(1.0 / (f_nonlin * d_forget))
            else:
                self.scale_retain = lora_alpha * math.sqrt(1.0 / (lora_r * d_retain * f_nonlin))
                self.scale_forget = lora_alpha * math.sqrt(1.0 / (lora_r * d_forget * f_nonlin))
        else:
            self.scale_retain = variance_scale
            self.scale_forget = variance_scale

        # runtime ablation multipliers (default 1.0 during training;
        # flipped to 0.0 at eval to disable an adapter)
        self.retain_scale = 1.0
        self.forget_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_mlp(x)
        r = self.down_retain(F.silu(self.gate_retain(x)) * self.up_retain(x))
        f = self.down_forget(F.silu(self.gate_forget(x)) * self.up_forget(x))
        return y + (self.retain_scale * self.scale_retain) * r + (self.forget_scale * self.scale_forget) * f


def inject_adapters(
    model,
    d_retain: int,
    d_forget: int,
    layer_start: float = 0.0,
    layer_end: float = 1.0,
    layer_stride: int = 1,
    variance_scale: float | None = None,
    lora_alpha: int = 32,
    lora_r: int = 32,
    match_rslora: bool = False,
) -> list[int]:
    """Replace MLP blocks in the configured range; freeze base params.

    Returns the list of layer indices that were modified.
    """
    layers = model.model.layers
    n_layers = len(layers)
    hidden = model.config.hidden_size

    start = int(n_layers * layer_start)
    end = int(n_layers * layer_end)
    indices = list(range(start, end, layer_stride))

    for i in indices:
        base_mlp = layers[i].mlp
        layers[i].mlp = DualMLPAdapter(
            base_mlp=base_mlp,
            hidden_size=hidden,
            d_retain=d_retain,
            d_forget=d_forget,
            variance_scale=variance_scale,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
            match_rslora=match_rslora,
        )

    # Freeze everything outside the adapter branches
    for name, p in model.named_parameters():
        if _is_adapter_param(name):
            p.requires_grad = True
        else:
            p.requires_grad = False

    return indices


_ADAPTER_SUFFIXES = (
    "gate_retain.weight",
    "up_retain.weight",
    "down_retain.weight",
    "gate_forget.weight",
    "up_forget.weight",
    "down_forget.weight",
)


def _is_adapter_param(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _ADAPTER_SUFFIXES)


def set_scales(model, retain_scale: float, forget_scale: float) -> None:
    """Set the runtime ablation multipliers on every DualMLPAdapter."""
    for m in model.modules():
        if isinstance(m, DualMLPAdapter):
            m.retain_scale = retain_scale
            m.forget_scale = forget_scale


def iter_adapters(model) -> Iterable[DualMLPAdapter]:
    for m in model.modules():
        if isinstance(m, DualMLPAdapter):
            yield m


def param_groups(model) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (retain_params, forget_params) lists for optimizer construction."""
    retain = []
    forget = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "_retain" in name:
            retain.append(p)
        elif "_forget" in name:
            forget.append(p)
    return retain, forget
