"""2D scatter-line plot: pass rate (x) vs hack rate (y, inverted so up is good).
Each run is a connected line through its epochs. Epoch labels at each point.
v5=✓ elicitation prompt only. Optimal corner = top-right (high pass, low hack).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
THR = 0.5


def get_pass_hack(jobname: str) -> tuple[float, float] | None:
    p = REPO / "build" / "jobs" / jobname / "judge_scores_judge_v3.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    pr = s["baseline_pass_rate"] or 0.0
    bt = {b["threshold"]: b for b in s.get("by_threshold", [])}
    hr = (bt[THR]["hack_rate"] if THR in bt else 0.0) or 0.0
    return pr, hr


# Base model (epoch 0). Same anchor for all runs (no adapter).
# Source: RESULTS.md base v5=✓ — 26.3% pass, 73.2% hack≥0.5.
BASE_PASS, BASE_HACK = 0.263, 0.732

# OLD exclusive routing — pass% and hack%≥0.5 from RESULTS.md table.
exclusive = [
    (0, BASE_PASS, BASE_HACK),
    (1, 0.347, 0.010),  # s1like-ep1 retain v5=✓
    (3, 0.010, 0.000),  # s1like-ep3 retain v5=✓
    (5, 0.010, 0.010),  # s1like-ep5 retain v5=✓
]


def collect_classic(prefix: str, eps: list[int]):
    """Pull (pass, hack) for each epoch from the live job dir."""
    out = [(0, BASE_PASS, BASE_HACK)]
    for ep in eps:
        kn = get_pass_hack(f"{prefix}{ep}-retain-v5")
        if kn is None:
            continue
        out.append((ep, kn[0], kn[1]))
    return out


classic   = collect_classic("gr-s1like-unc-ep",   [3, 4, 5])  # ep1, ep2 dropped (360s legacy)
filtering = collect_classic("gr-s1like-ga0-ep",   [1, 2, 3])
adp3      = collect_classic("gr-s1like-3adp-ep",  [1, 2, 3, 4, 5])
fo10      = collect_classic("gr-s1like-fo10-ep",  [1, 2, 3, 4, 5])
ga4       = collect_classic("gr-s1like-ga4-ep",   [1, 2, 3, 4, 5])


def plot_run(ax, points, color, marker, label, linestyle="-"):
    if len(points) < 1:
        return
    xs = [p[1] for p in points]  # pass
    ys = [p[2] for p in points]  # hack
    ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle,
            markersize=10, linewidth=1.8, label=label, zorder=3)
    for ep, x, y in points:
        ax.annotate(str(ep), xy=(x, y), xytext=(7, 6), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold", zorder=4)


fig, ax = plt.subplots(figsize=(10, 8))

# Optimal corner shading: high pass, low hack (top-right).
ax.axhspan(0.0, 0.2, xmin=0.6, xmax=1.0, color="#2ca02c", alpha=0.06, zorder=0)

plot_run(ax, exclusive, "#2ca02c", "o", "exclusive retain", "-")
plot_run(ax, classic,   "#2ca02c", "o", "classic retain",   ":")
plot_run(ax, adp3,      "#1f77b4", "s", "3-adapter retain")
plot_run(ax, filtering, "#9467bd", "^", "filtering",        "-.")
plot_run(ax, fo10,      "#d62728", "D", "fo10 retain",      "--")
if len(ga4) > 1:
    plot_run(ax, ga4, "#ff8c1a", "v", "ga4 retain", "-")

# Base anchor (large gray X)
ax.scatter([BASE_PASS], [BASE_HACK], marker="X", s=200, color="#666666",
           zorder=5, label="base (epoch 0)")
ax.annotate("base", xy=(BASE_PASS, BASE_HACK), xytext=(8, -14),
            textcoords="offset points", fontsize=10, color="#444444")

# Axes — y inverted so low hack (good) is up.
ax.set_xlim(0.0, 0.8)
ax.set_ylim(0.0, 1.0)
ax.invert_yaxis()
ax.set_xlabel("Task pass rate →  better →", fontsize=12)
ax.set_ylabel("← better ←  Hack rate (≥0.5)", fontsize=12)
ax.grid(alpha=0.3)

# Optimal-corner annotation
ax.text(0.78, 0.02, "↑ optimal", fontsize=11, ha="right", va="top",
        color="#2ca02c", fontweight="bold")

ax.set_title("v5 elicitation prompt: pass vs hack by epoch\n(numbers = epoch; lines connect a run's epochs)",
             fontsize=13)
ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

fig.tight_layout()
out = REPO / "charts" / "routing_scatter_v5.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

# Data dump
def dump(label, pts):
    print(f"  {label}:")
    for ep, p, h in pts:
        print(f"    ep={ep}  pass={p:.1%}  hack={h:.1%}")
print()
dump("exclusive", exclusive)
dump("classic",   classic)
dump("3-adapter", adp3)
dump("filtering", filtering)
dump("fo10",      fo10)
dump("ga4",       ga4)
