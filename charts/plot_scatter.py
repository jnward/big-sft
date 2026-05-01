"""Camera-ready 2-D scatter: pass rate vs hack rate, by epoch, with error bars.
v5 elicitation prompt only. Optimal corner = top-right (high pass, low hack).
Each family is one connected line through its epochs (1..5); marker labeled
with epoch number; Wilson 95% CI bars on both axes.

Curated families: classic (s1like_unc_both), filtering (ga0), ga1, ga2, ga4.
Plus base Qwen3-32B as a single anchor point.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
# Default threshold for hack rate. CLI: pass --thr 0.8 to override.
import os
THR = float(os.environ.get("HACK_THRESHOLD", "0.5"))


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
    pass_lo, pass_mid, pass_hi = wilson(k_pass, nr)
    hack_lo, hack_mid, hack_hi = wilson(k_hack, ns)
    return {
        "pass":  (pass_mid, pass_lo, pass_hi),
        "hack":  (hack_mid, hack_lo, hack_hi),
        "n":     (nr, ns),
    }


def collect_family(prefix: str, eps=(1, 2, 3, 4, 5)):
    """Return list of (ep, pass_ci, hack_ci) tuples for available epochs."""
    out = []
    for ep in eps:
        ci = get_pass_hack_ci(f"{prefix}{ep}-retain-v5")
        if ci is None:
            continue
        out.append((ep, ci["pass"], ci["hack"]))
    return out


classic = collect_family("gr-s1like-unc-ep")
ga0     = collect_family("gr-s1like-ga0-ep")
ga1     = collect_family("gr-s1like-ga1-ep")
ga2     = collect_family("gr-s1like-ga2-ep")
ga4     = collect_family("gr-s1like-ga4-ep")

# Base — try the new patched eval first, fall back to other base evals.
base_ci = (
    get_pass_hack_ci("base-qwen3-32b-v5-99")
    or get_pass_hack_ci("base-qwen3-32b-v5-k4")
)


def plot_run(ax, points, color, marker, label, linestyle="-"):
    """Each `points` element = (epoch, (mid,lo,hi)pass, (mid,lo,hi)hack)."""
    if not points:
        return
    xs = [p[1][0] for p in points]                                    # pass mid
    ys = [p[2][0] for p in points]                                    # hack mid
    xerr_lo = [p[1][0] - p[1][1] for p in points]
    xerr_hi = [p[1][2] - p[1][0] for p in points]
    yerr_lo = [p[2][0] - p[2][1] for p in points]
    yerr_hi = [p[2][2] - p[2][0] for p in points]
    ax.errorbar(
        xs, ys,
        xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
        fmt=marker, color=color, linestyle=linestyle,
        markersize=10, linewidth=2.0, elinewidth=1.0,
        ecolor=color, capsize=3, alpha=0.95,
        label=label, zorder=3,
    )
    for ep, p_ci, h_ci in points:
        ax.annotate(str(ep), xy=(p_ci[0], h_ci[0]), xytext=(8, 7),
                    textcoords="offset points", fontsize=10,
                    color=color, fontweight="bold", zorder=4)


fig, ax = plt.subplots(figsize=(10, 8))

# Optimal corner (top-right when y-axis is inverted)
ax.axhspan(0.0, 0.20, xmin=0.55, xmax=1.0, color="#2ca02c", alpha=0.06, zorder=0)

plot_run(ax, classic, "#2ca02c", "o", "classic (s1like_unc)",       "-")
plot_run(ax, ga0,     "#9467bd", "^", "filtering (ga0)",           "-.")
plot_run(ax, ga1,     "#8c564b", "P", "ga1",                       ":")
plot_run(ax, ga2,     "#1f77b4", "s", "ga2",                       "--")
plot_run(ax, ga4,     "#ff7f0e", "v", "ga4",                       "-")

if base_ci is not None:
    bp = base_ci["pass"]
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
ax.set_xlabel("Task pass rate  →  better →", fontsize=13)
ax.set_ylabel(f"← better ←  Hack rate (≥{THR})", fontsize=13)
ax.grid(alpha=0.3)
ax.text(0.78, 0.02, "↑ optimal", fontsize=11, ha="right", va="top",
        color="#2ca02c", fontweight="bold")
ax.set_title(
    "v5 elicitation prompt: pass rate vs hack rate\n"
    "(numbers = epoch; lines connect a family's epochs; Wilson 95% CI)",
    fontsize=13,
)
ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
fig.tight_layout()

suffix = "" if THR == 0.5 else f"_thr{THR}"
out = REPO / "charts" / f"routing_scatter_v5{suffix}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

# Data dump
def dump(label, pts):
    print(f"\n  {label}:")
    for ep, p_ci, h_ci in pts:
        print(f"    ep={ep}  pass={p_ci[0]:.1%} [{p_ci[1]:.1%},{p_ci[2]:.1%}]  "
              f"hack={h_ci[0]:.1%} [{h_ci[1]:.1%},{h_ci[2]:.1%}]")
print("\n=== Data ===")
dump("classic",   classic)
dump("ga0",       ga0)
dump("ga1",       ga1)
dump("ga2",       ga2)
dump("ga4",       ga4)
if base_ci:
    bp, bh = base_ci["pass"], base_ci["hack"]
    print(f"\n  base: pass={bp[0]:.1%} [{bp[1]:.1%},{bp[2]:.1%}]  "
          f"hack={bh[0]:.1%} [{bh[1]:.1%},{bh[2]:.1%}]")
