"""Side-by-side scatter: gradient-routing modes only.

Compares the three gradient-routing configurations of the unc_both ckpt
(retain-only=green, forget-only=red, both adapters=blue) plus the base
model. Same side-by-side layout, axes, and conventions as plot_scatter.py.

Currently classic forget-only and both-adapters are evaluated only in
the no-v5 phase, so the v5 panel will be sparse.

Env knobs:
  HACK_THRESHOLD  — judge score cutoff for "hack" (default 0.8).
  LEGIT_X         — 1 (default) for legitimate-solution-rate x-axis,
                    0 for raw pass-rate x-axis.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

REPO = Path(__file__).resolve().parents[1]
THR = float(os.environ.get("HACK_THRESHOLD", "0.8"))
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


def plot_point(ax, ci, color, marker, label, markersize=25, zorder=3):
    if ci is None:
        return
    x_ci, y_ci = xy_of(ci)
    ax.errorbar(
        [x_ci[0]], [y_ci[0]],
        xerr=[[x_ci[0] - x_ci[1]], [x_ci[2] - x_ci[0]]],
        yerr=[[y_ci[0] - y_ci[1]], [y_ci[2] - y_ci[0]]],
        fmt=marker, color=color, markersize=markersize,
        linewidth=0, elinewidth=2.6, ecolor=color,
        capsize=7, alpha=0.95, label=label, zorder=zorder,
    )


# Colors per the latest spec: retain=green (ours), forget=red, both=blue, base=gray.
COL_RETAIN = "#2ca02c"
COL_FORGET = "#d62728"
COL_BOTH   = "#1f77b4"
COL_BASE   = "#444444"


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

    plot_point(ax, retain, COL_RETAIN, "o", "gradient routing — retain only",   markersize=25)
    plot_point(ax, forget, COL_FORGET, "^", "gradient routing — forget only",   markersize=25)
    plot_point(ax, both,   COL_BOTH,   "s", "gradient routing — both adapters", markersize=25)
    if base_ci is not None:
        bp = base_ci["legit"] if LEGIT_X else base_ci["pass"]
        bh = base_ci["hack"]
        ax.errorbar(
            [bp[0]], [bh[0]],
            xerr=[[bp[0] - bp[1]], [bp[2] - bp[0]]],
            yerr=[[bh[0] - bh[1]], [bh[2] - bh[0]]],
            fmt="X", color=COL_BASE, markersize=31, linewidth=0,
            elinewidth=2.6, ecolor=COL_BASE, capsize=7,
            label="Qwen3-32B", zorder=5,
        )

    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=23)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.text(0.78, 0.02, "optimal ↗", fontsize=22, ha="right", va="top",
            color=COL_RETAIN, fontweight="bold")


# Left = without elicitation, Right = with elicitation.
fig, (ax_no, ax_v5) = plt.subplots(1, 2, figsize=(28.6, 11), sharey=True)

render_panel(ax_no, "no")
render_panel(ax_v5, "v5")

fig.suptitle("Legitimate Solution Rate vs Hack Rate", fontsize=34, y=1.00)
ax_no.set_title("without hack elicitation prompt", fontsize=32)
ax_v5.set_title("with hack elicitation prompt",    fontsize=32)

xlab = "Legitimate Solution Rate → better" if LEGIT_X else "Pass Rate → better"
ax_no.set_xlabel(xlab, fontsize=29)
ax_v5.set_xlabel(xlab, fontsize=29)
ax_no.set_ylabel("Hack Rate → better", fontsize=29)

# Hand-built marker-only legend handles (no error-bar caps).
LEGEND_SPECS = [
    # (label, marker, color, markersize) — sizes match the actual plot points.
    ("gradient routing — retain only",   "o", COL_RETAIN, 25),
    ("gradient routing — forget only",   "^", COL_FORGET, 25),
    ("gradient routing — both adapters", "s", COL_BOTH,   25),
    ("Qwen3-32B",                        "X", COL_BASE,   31),
]
legend_handles = [
    Line2D([0], [0], marker=m, color=c, markersize=ms,
           linestyle="None", markerfacecolor=c, markeredgecolor=c)
    for _, m, c, ms in LEGEND_SPECS
]
legend_labels = [label for label, *_ in LEGEND_SPECS]
ax_v5.legend(legend_handles, legend_labels,
             loc="lower right", fontsize=23, framealpha=0.95)

fig.tight_layout()
fig.subplots_adjust(wspace=0.15)

suffix = "" if THR == 0.8 else f"_thr{THR}"
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


for phase in ["no", "v5"]:
    print(f"\n=== {phase} phase ===")
    _dump("gr — retain only", get_pass_hack_ci(f"gr-s1like-unc-ep5-retain-{phase}"))
    _dump("gr — forget only", get_pass_hack_ci(f"gr-s1like-unc-ep5-forget-{phase}"))
    _dump("gr — both",        get_pass_hack_ci(f"gr-s1like-unc-ep5-both-{phase}"))
    if phase == "v5":
        bc = (get_pass_hack_ci("base-qwen3-32b-v5-99")
              or get_pass_hack_ci("base-qwen3-32b-v5-k4"))
    else:
        bc = get_pass_hack_ci("base-qwen3-32b-no-99")
    _dump("Qwen3-32B", bc)
