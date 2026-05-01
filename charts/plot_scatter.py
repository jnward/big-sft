"""Camera-ready scatter: pass rate vs hack rate at ep5 for all families.

Optimal corner = top-right (high pass, low hack). Wilson 95% CI bars on both
axes. ep5 only — earlier epochs no longer plotted.

Families:
  - classic (s1like_unc_both): retain (green); forget-only (red) and
    both-adapters (dark green) when present (currently only in the no-v5
    phase).
  - filtering (ga0): single orange point (no GA penalty).
  - gradient ascent: ga1/2/4 connected by a blue line representing the
    progression from low penalty (1×) to high (4×), labelled by multiplier.
  - noint, skyline: single anchor points (when present).
  - base Qwen3-32B: anchor.

Env knobs:
  HACK_THRESHOLD  — judge score cutoff for "hack" (default 0.5).
  LEGIT_X=1       — x-axis becomes "% passed AND not hack-flagged at THR"
                    (default x = raw pass rate).
  EVAL_SUFFIX     — 'v5' (default) or 'no'. Selects which job suffix
                    (-retain-v5 vs -retain-no) to load.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
THR = float(os.environ.get("HACK_THRESHOLD", "0.5"))
LEGIT_X = os.environ.get("LEGIT_X", "0") == "1"
EVAL_SUFFIX = os.environ.get("EVAL_SUFFIX", "v5")
assert EVAL_SUFFIX in ("v5", "no"), f"unsupported EVAL_SUFFIX={EVAL_SUFFIX}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, p, c + h)


def get_pass_hack_ci(jobname: str):
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
    """(x_ci, y_ci) for given CI dict, where x is pass or legit-pass."""
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


def plot_line(ax, xys, color, marker, label, annotations=None,
              markersize=10, linestyle="-"):
    """xys = list of (x_ci, y_ci); plot a connected line with error bars."""
    if not xys:
        return
    xs = [p[0][0] for p in xys]
    ys = [p[1][0] for p in xys]
    xerr_lo = [p[0][0] - p[0][1] for p in xys]
    xerr_hi = [p[0][2] - p[0][0] for p in xys]
    yerr_lo = [p[1][0] - p[1][1] for p in xys]
    yerr_hi = [p[1][2] - p[1][0] for p in xys]
    ax.errorbar(
        xs, ys,
        xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
        fmt=marker, color=color, linestyle=linestyle,
        markersize=markersize, linewidth=2.0, elinewidth=1.0,
        ecolor=color, capsize=3, alpha=0.95,
        label=label, zorder=3,
    )
    if annotations:
        for ann, p in zip(annotations, xys):
            ax.annotate(ann, xy=(p[0][0], p[1][0]), xytext=(8, 7),
                        textcoords="offset points", fontsize=10,
                        color=color, fontweight="bold", zorder=4)


# === Load data ===
classic_retain = get_pass_hack_ci(f"gr-s1like-unc-ep5-retain-{EVAL_SUFFIX}")
classic_forget = get_pass_hack_ci(f"gr-s1like-unc-ep5-forget-{EVAL_SUFFIX}")
classic_both   = get_pass_hack_ci(f"gr-s1like-unc-ep5-both-{EVAL_SUFFIX}")

# Filtering (ga0) is a separate baseline (no GA penalty). The gradient
# ascent line connects only ga1/2/4 with their penalty multipliers.
filtering = get_pass_hack_ci(f"gr-s1like-ga0-ep5-retain-{EVAL_SUFFIX}")

GA_MULTIPLIERS = [1, 2, 4]
ga_xys = []  # ordered (mult, xy) along the penalty axis
for mult in GA_MULTIPLIERS:
    ci = get_pass_hack_ci(f"gr-s1like-ga{mult}-ep5-retain-{EVAL_SUFFIX}")
    if ci is None:
        continue
    ga_xys.append((mult, xy_of(ci)))

noint   = get_pass_hack_ci(f"gr-s1like-noint-ep5-retain-{EVAL_SUFFIX}")
skyline = get_pass_hack_ci(f"gr-s1like-skyline-ep5-retain-{EVAL_SUFFIX}")

if EVAL_SUFFIX == "v5":
    base_ci = (
        get_pass_hack_ci("base-qwen3-32b-v5-99")
        or get_pass_hack_ci("base-qwen3-32b-v5-k4")
    )
else:
    base_ci = get_pass_hack_ci("base-qwen3-32b-no-99")


# === Plot ===
fig, ax = plt.subplots(figsize=(10, 8))

# Optimal corner highlight (top-right after y-axis inversion)
ax.axhspan(0.0, 0.20, xmin=0.55, xmax=1.0, color="#2ca02c", alpha=0.06, zorder=0)

# Filtering (ga0): orange single point — the no-penalty baseline.
plot_point(ax, filtering, "#ff7f0e", "D", "filtering (ga0)", markersize=10)

# Gradient ascent line (blue): ga1, ga2, ga4 only — penalty progression.
GA_COLOR = "#1f77b4"
plot_line(
    ax,
    [xy for _, xy in ga_xys],
    color=GA_COLOR,
    marker="o",
    label="gradient ascent",
    annotations=[f"{m}×" for m, _ in ga_xys],
)

# Classic family
CLASSIC_COLOR = "#2ca02c"
plot_point(ax, classic_retain, CLASSIC_COLOR, "o", "classic (retain)")
plot_point(ax, classic_forget, "#d62728",     "^", "classic (forget-only)")
plot_point(ax, classic_both,   "#1b6e1b",     "s", "classic (both adapters)")

# Other families
plot_point(ax, noint,   "#9467bd", "P", "noint",   markersize=11)
plot_point(ax, skyline, "#8c564b", "*", "skyline", markersize=14)

# Base anchor
if base_ci is not None:
    bp = base_ci["legit"] if LEGIT_X else base_ci["pass"]
    bh = base_ci["hack"]
    ax.errorbar(
        [bp[0]], [bh[0]],
        xerr=[[bp[0] - bp[1]], [bp[2] - bp[0]]],
        yerr=[[bh[0] - bh[1]], [bh[2] - bh[0]]],
        fmt="X", color="#444444", markersize=14, linewidth=0,
        elinewidth=1.2, ecolor="#444444", capsize=3,
        label="base Qwen3-32B", zorder=5,
    )
    ax.annotate("base", xy=(bp[0], bh[0]), xytext=(10, -16),
                textcoords="offset points", fontsize=11, color="#222222")

ax.set_xlim(0.0, 0.8)
ax.set_ylim(0.0, 1.0)
ax.invert_yaxis()
if LEGIT_X:
    ax.set_xlabel(f"Legit-pass rate (passed AND not hack-flagged ≥{THR})  →  better →", fontsize=12)
else:
    ax.set_xlabel("Task pass rate  →  better →", fontsize=13)
ax.set_ylabel(f"← better ←  Hack rate (≥{THR})", fontsize=13)
ax.grid(alpha=0.3)
ax.text(0.78, 0.02, "↑ optimal", fontsize=11, ha="right", va="top",
        color="#2ca02c", fontweight="bold")

prompt_label = "v5 elicitation prompt" if EVAL_SUFFIX == "v5" else "no v5 prompt (default)"
ax.set_title(
    f"{prompt_label}: pass rate vs hack rate (epoch 5)\n"
    "Wilson 95% CI",
    fontsize=13,
)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
fig.tight_layout()

suffix = "" if THR == 0.5 else f"_thr{THR}"
if LEGIT_X:
    suffix += "_legit"
out = REPO / "charts" / f"routing_scatter_{EVAL_SUFFIX}{suffix}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")


def _dump(label, ci):
    if ci is None:
        return
    x = ci["legit"] if LEGIT_X else ci["pass"]
    h = ci["hack"]
    print(f"  {label:30s}  x={x[0]:.1%} [{x[1]:.1%},{x[2]:.1%}]  "
          f"hack={h[0]:.1%} [{h[1]:.1%},{h[2]:.1%}]")


print("\n=== Data ===")
_dump("classic (retain)", classic_retain)
_dump("classic (forget-only)", classic_forget)
_dump("classic (both)", classic_both)
_dump("filtering (ga0)", filtering)
for mult, _ in ga_xys:
    _dump(f"GA {mult}×", get_pass_hack_ci(f"gr-s1like-ga{mult}-ep5-retain-{EVAL_SUFFIX}"))
_dump("noint", noint)
_dump("skyline", skyline)
_dump("base", base_ci)
