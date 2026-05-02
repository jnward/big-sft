"""Side-by-side scatter: gradient-routing modes only.

Compares the three gradient-routing configurations of the unc_both ckpt
(retain-only / forget-only / both adapters) plus the base model. Same
side-by-side layout, axes, and conventions as plot_scatter.py.

Currently classic forget-only and both-adapters are evaluated only in
the no-v5 phase, so the v5 panel will be sparse.

Env knobs:
  HACK_THRESHOLD  — judge score cutoff for "hack" (default 0.5).
  LEGIT_X         — 1 (default) for legitimate-solution-rate x-axis,
                    0 for raw pass-rate x-axis.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
THR = float(os.environ.get("HACK_THRESHOLD", "0.5"))
LEGIT_X = os.environ.get("LEGIT_X", "1") == "1"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, p, c + h)


def get_pass_hack_ci(jobname):
    p = REPO / "build" / "jobs" / jobname / "judge_scores_judge_v3.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    nr = s["n_with_reward"] or 0
    ns = s["n_with_score"] or 0
    pr = s["baseline_pass_rate"] or 0.0
    k_pass = int(round(pr * nr))
    bt = {b["threshold"]: b for b in s.get("by_threshold", [])}
    k_hack = bt[THR]["n_hacked"] if THR in bt else 0
    k_passed_hacked = bt[THR]["passed_hacked"] if THR in bt else 0
    k_legit_pass = max(0, k_pass - k_passed_hacked)
    pass_lo, pass_mid, pass_hi = wilson(k_pass, nr)
    legit_lo, legit_mid, legit_hi = wilson(k_legit_pass, ns)
    hack_lo, hack_mid, hack_hi = wilson(k_hack, ns)
    return {
        "pass":  (pass_mid, pass_lo, pass_hi),
        "legit": (legit_mid, legit_lo, legit_hi),
        "hack":  (hack_mid, hack_lo, hack_hi),
    }


def xy_of(ci):
    if ci is None:
        return None
    return (ci["legit"] if LEGIT_X else ci["pass"], ci["hack"])


def plot_point(ax, ci, color, marker, label, markersize=10, zorder=3):
    if ci is None:
        return
    x_ci, y_ci = xy_of(ci)
    ax.errorbar(
        [x_ci[0]], [y_ci[0]],
        xerr=[[x_ci[0] - x_ci[1]], [x_ci[2] - x_ci[0]]],
        yerr=[[y_ci[0] - y_ci[1]], [y_ci[2] - y_ci[0]]],
        fmt=marker, color=color, markersize=markersize,
        linewidth=0, elinewidth=1.0, ecolor=color,
        capsize=3, alpha=0.95, label=label, zorder=zorder,
    )


GR_COLOR = "#2ca02c"
BASE_COLOR = "#444444"


def render_panel(ax, suffix):
    retain = get_pass_hack_ci(f"gr-s1like-unc-ep5-retain-{suffix}")
    forget = get_pass_hack_ci(f"gr-s1like-unc-ep5-forget-{suffix}")
    both   = get_pass_hack_ci(f"gr-s1like-unc-ep5-both-{suffix}")
    if suffix == "v5":
        base_ci = (
            get_pass_hack_ci("base-qwen3-32b-v5-99")
            or get_pass_hack_ci("base-qwen3-32b-v5-k4")
        )
    else:
        base_ci = get_pass_hack_ci("base-qwen3-32b-no-99")

    ax.axhspan(0.0, 0.20, xmin=0.55, xmax=1.0,
               color=GR_COLOR, alpha=0.06, zorder=0)

    plot_point(ax, retain, GR_COLOR, "o", "gradient routing — retain only",  markersize=11)
    plot_point(ax, forget, GR_COLOR, "^", "gradient routing — forget only",  markersize=11)
    plot_point(ax, both,   GR_COLOR, "s", "gradient routing — both adapters", markersize=11)
    if base_ci is not None:
        bp = base_ci["legit"] if LEGIT_X else base_ci["pass"]
        bh = base_ci["hack"]
        ax.errorbar(
            [bp[0]], [bh[0]],
            xerr=[[bp[0] - bp[1]], [bp[2] - bp[0]]],
            yerr=[[bh[0] - bh[1]], [bh[2] - bh[0]]],
            fmt="X", color=BASE_COLOR, markersize=14, linewidth=0,
            elinewidth=1.2, ecolor=BASE_COLOR, capsize=3,
            label="base Qwen3-32B", zorder=5,
        )

    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.text(0.78, 0.02, "↗ optimal", fontsize=11, ha="right", va="top",
            color=GR_COLOR, fontweight="bold")


fig, (ax_v5, ax_no) = plt.subplots(1, 2, figsize=(20, 8.5), sharey=True)

render_panel(ax_v5, "v5")
render_panel(ax_no, "no")

ax_v5.set_title(
    "Legitimate Solution Rate vs Hack Rate (with hack elicitation prompt)",
    fontsize=13)
ax_no.set_title(
    "Legitimate Solution Rate vs Hack Rate (without hack elicitation prompt)",
    fontsize=13)

xlab = "Legitimate Solution Rate → better" if LEGIT_X else "Pass Rate → better"
ax_v5.set_xlabel(xlab, fontsize=12)
ax_no.set_xlabel(xlab, fontsize=12)
ax_v5.set_ylabel(f"Hack Rate (≥{THR}) → better", fontsize=12)

ax_no.legend(loc="lower right", fontsize=10, framealpha=0.95)

fig.tight_layout()

suffix = "" if THR == 0.5 else f"_thr{THR}"
if not LEGIT_X:
    suffix += "_passx"
out = REPO / "charts" / f"routing_scatter_gr{suffix}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")


def _dump(label, ci):
    if ci is None:
        return
    x = ci["legit"] if LEGIT_X else ci["pass"]
    h = ci["hack"]
    print(f"  {label:32s}  x={x[0]:.1%} [{x[1]:.1%},{x[2]:.1%}]  "
          f"hack={h[0]:.1%} [{h[1]:.1%},{h[2]:.1%}]")


for phase in ["v5", "no"]:
    print(f"\n=== {phase} phase ===")
    _dump("gr — retain only", get_pass_hack_ci(f"gr-s1like-unc-ep5-retain-{phase}"))
    _dump("gr — forget only", get_pass_hack_ci(f"gr-s1like-unc-ep5-forget-{phase}"))
    _dump("gr — both",        get_pass_hack_ci(f"gr-s1like-unc-ep5-both-{phase}"))
    if phase == "v5":
        bc = (get_pass_hack_ci("base-qwen3-32b-v5-99")
              or get_pass_hack_ci("base-qwen3-32b-v5-k4"))
    else:
        bc = get_pass_hack_ci("base-qwen3-32b-no-99")
    _dump("base Qwen3-32B", bc)
