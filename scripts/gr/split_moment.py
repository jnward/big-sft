"""Split-moment AdamW for gradient-routing SFT: first moment (m) from the ROUTED
gradient (`p.grad`, gate-masked + κ-redistributed), second moment (v) from the
NATURAL/pre-routing gradient (`p._pre_routing_grad`, every example reaching both
adapters at scale 1).

Why: Adam is scale-invariant — feeding a κ×-amplified gradient into BOTH moments
would cancel (`m∝κg, √v∝κg → step∝1`), so a "2× forget update" would be a no-op.
Splitting the moments (m rides the κ× routed grad, v rides the ×1 natural grad)
makes the realized per-coordinate step scale with κ: at κ=2 the forget adapter
takes a genuine 2× step. This is the "update-size correction."

Ported verbatim from rl-rewardhacking-private/sft/split_moment.py (itself ported from
small-rl, the GRAFT master-port optimizer). big-sft uses
it at λ=1, so `set_window(..., w_max=None)` keeps the over-routing clamp inert;
the general clamp path is preserved verbatim for faithfulness.

Extensions consumed via `set_window` each optimizer window:
- per-role PARTICIPATION `c_A` (scales the v-source; with interlaced retain-only
  coherence, `c_F = N/N_routing` makes forget step at retain's per-example rate).
- FREEZE: an adapter with no examples this window (`active[role]=False`) is skipped
  entirely — m, v, per-param step counter, wd all untouched (per-role bias
  correction falls out for free).

A routing (graft_role-tagged) param with no captured `_pre_routing_grad` is a HARD
ERROR — no silent fallback to plain AdamW under routing.
"""
import math

import torch
from torch.optim import AdamW


def clip_pre_routing_grads_(param_groups, max_norm, total_norm):
    """Scale every `p._pre_routing_grad` by the same coef `clip_grad_norm_` applies
    to `.grad` (`min(1, max_norm/(total_norm+1e-6))`), so clipping is one shared
    event across Adam's two moment sources. No-op when nothing clips."""
    if max_norm is None or total_norm is None:
        return None
    clip_coef = float(max_norm) / (float(total_norm) + 1e-6)
    if clip_coef >= 1.0:
        return None
    for group in param_groups:
        for p in group["params"]:
            b = getattr(p, "_pre_routing_grad", None)
            if b is not None:
                b.mul_(clip_coef)
    return clip_coef


class SplitMomentAdamW(AdamW):
    def set_window(self, participation, active, *, w_max=None, step_policy="clamp"):
        """Stash {role: c_A} participation + {role: bool} active for the next step().
        Called each optimizer window before HF's arg-less optimizer.step()."""
        self._window = {"c": participation, "active": active,
                        "w_max": w_max, "step_policy": step_policy}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        window = getattr(self, "_window", None)
        w_max = window["w_max"] if window is not None else None
        step_policy = window["step_policy"] if window is not None else "clamp"
        realized_p999 = 0.0
        realized_abs_max = 0.0
        n_clamped = 0
        n_coords = 0
        forget_realized = []  # per-param mean realized step |m̂|/(√v̂+eps) — the κ gauge
        for group in self.param_groups:
            assert not group.get("amsgrad", False), "SplitMomentAdamW: amsgrad unsupported"
            assert not group.get("maximize", False), "SplitMomentAdamW: maximize unsupported"
            assert not group.get("fused", False) and not group.get("foreach", False), (
                "SplitMomentAdamW: fused/foreach unsupported (use the plain loop)")
            role = group.get("graft_role")
            if window is not None and role is not None:
                if not window["active"].get(role, True):
                    continue  # FREEZE: skip m / v / step counter / wd
                c = float(window["c"].get(role, 1.0))
            else:
                c = 1.0
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g_m = p.grad  # first moment <- routed gradient
                if g_m is None:
                    continue
                assert not g_m.is_sparse, "SplitMomentAdamW: sparse grads unsupported"
                g_v = getattr(p, "_pre_routing_grad", None)  # second moment <- natural
                if g_v is None:
                    assert role is None, (
                        "SplitMomentAdamW: a routing param has no captured "
                        "_pre_routing_grad — the pre-routing capture did not fire. "
                        "Refusing to silently fall back to plain AdamW under routing.")
                    g_v = g_m
                if c != 1.0:
                    g_v = g_v * c  # participation: scale the v-source (squared below)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # fp32 optimizer states even for bf16 params: the split-moment κ
                    # correction is precision-sensitive (bf16 EMAs measured κ=2 -> ~1.9).
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32,
                                                        memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32,
                                                           memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                gm32, gv32 = g_m.float(), g_v.float()   # accumulate moments in fp32
                exp_avg.mul_(beta1).add_(gm32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gv32, gv32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)  # fp32
                step_size = lr / bias_correction1
                mhat_abs = exp_avg.abs() / bias_correction1
                if role == "forget":
                    forget_realized.append((mhat_abs / denom).mean())  # κ gauge (any w_max)
                # Over-routing step control (λ>1 only; inert at λ=1 with w_max=None).
                if w_max is not None and role is not None:
                    realized = mhat_abs / denom
                    rflat = realized.flatten().float()
                    q = (torch.quantile(rflat, 0.999).item() if rflat.numel() > 1
                         else float(rflat.item()))
                    realized_p999 = max(realized_p999, q)
                    realized_abs_max = max(realized_abs_max, float(realized.max().item()))
                    if step_policy == "clamp":
                        floor = mhat_abs / w_max
                        bit = floor > denom
                        n_clamped += int(bit.sum().item())
                        n_coords += bit.numel()
                        denom = torch.maximum(denom, floor)
                p.add_((exp_avg / denom).to(p.dtype), alpha=-step_size)  # fp32 step, cast to param dtype

        if w_max is not None:
            self._last_realized_max = realized_p999
            self._last_realized_abs_max = realized_abs_max
            self._last_frac_clamped = (n_clamped / n_coords) if n_coords else 0.0
            if step_policy == "gate":
                assert realized_p999 <= w_max + 1e-6, (
                    f"GRAFT realized-step gate: per-coordinate step {realized_p999:.3g}× lr "
                    f"> W_MAX={w_max} (raw max {realized_abs_max:.3g}×).")
        if forget_realized:
            self._last_forget_realized_step = torch.stack(forget_realized).mean().item()
        self._window = None
        return loss
