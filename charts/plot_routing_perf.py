"""Two-panel line plot: task performance vs epoch, retain vs both, for
exclusive routing (old, RESULTS.md) vs classic routing (new). v5=─ only.
Pulls live data from build/jobs/*/judge_scores_judge_v3.json."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, p, c + h)


def get_pass(jobname: str) -> tuple[int, int] | None:
    """Return (n_passed, n_with_reward) for a job's judge_scores summary."""
    p = REPO / "build" / "jobs" / jobname / "judge_scores_judge_v3.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    n = s["n_with_reward"] or 0
    pr = s["baseline_pass_rate"] or 0.0
    k = int(round(pr * n))
    return k, n


# Base model (epoch 0). Source: RESULTS.md "base | — | — | 33.7% (33/98)".
BASE_K, BASE_N = 33, 98

# ---- Exclusive routing (OLD; numbers from RESULTS.md, all n=99 k=1) ----
# (epoch, k_passed, n_with_reward)
exclusive_retain = [
    (0, BASE_K, BASE_N),
    (1, 37, 98),  # s1like-ep1 retain v5=─
    (2, 1, 98),   # s1like-ep2 retain v5=─
    (3, 1, 98),   # s1like-ep3 retain v5=─
    (5, 1, 98),   # s1like-ep5 retain v5=─
]
exclusive_both = [
    (0, BASE_K, BASE_N),
    (1, 42, 97),  # s1like-ep1 both v5=─
    (2, 36, 98),  # s1like-ep2 both v5=─
    (3, 27, 95),  # s1like-ep3 both v5=─
    (5, 25, 97),  # s1like-ep5 both v5=─
]

# ---- Classic routing (NEW; live from result files, mix of n=99 / n=25-random) ----
classic_retain: list[tuple[int, int, int]] = [(0, BASE_K, BASE_N)]
classic_both: list[tuple[int, int, int]] = [(0, BASE_K, BASE_N)]

if (kn := get_pass("gr-s1like-unc-ep1-retain")) is not None:
    classic_retain.append((1, kn[0], kn[1]))
if (kn := get_pass("gr-s1like-unc-ep5-retain-25-no")) is not None:
    classic_retain.append((5, kn[0], kn[1]))
if (kn := get_pass("gr-s1like-unc-ep5-both-25-no")) is not None:
    classic_both.append((5, kn[0], kn[1]))


def to_xy_band(rows: list[tuple[int, int, int]]):
    xs, mids, los, his = [], [], [], []
    for ep, k, n in rows:
        lo, mid, hi = wilson(k, n)
        xs.append(ep); mids.append(mid); los.append(lo); his.append(hi)
    return xs, mids, los, his


def plot_panel(ax, title: str, retain, both):
    # Both adapters
    xb, mb, lob, hib = to_xy_band(both)
    ax.fill_between(xb, lob, hib, color="#ff8c1a", alpha=0.18, linewidth=0)
    ax.plot(xb, mb, "-o", color="#ff8c1a", label="both adapters",
            markersize=8, linewidth=2)
    # Retain only
    xr, mr, lor, hir = to_xy_band(retain)
    ax.fill_between(xr, lor, hir, color="#2ca02c", alpha=0.18, linewidth=0)
    ax.plot(xr, mr, "-o", color="#2ca02c", label="retain only",
            markersize=8, linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=11)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
plot_panel(axes[0], "Exclusive routing", exclusive_retain, exclusive_both)
plot_panel(axes[1], "Classic routing", classic_retain, classic_both)
axes[0].set_ylabel("Task pass rate (v5=─, k=1)", fontsize=12)
fig.suptitle("Retain-only vs both-adapters performance by epoch (95% Wilson CI)",
             fontsize=14, y=1.02)
fig.tight_layout()
out = REPO / "charts" / "routing_perf_by_epoch.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

# data dump
print("\nData (epoch, k, n, pct, lo, hi):")
for label, rows in [("Excl retain", exclusive_retain), ("Excl both", exclusive_both),
                    ("Class retain", classic_retain), ("Class both", classic_both)]:
    print(f"  {label}:")
    for ep, k, n in rows:
        lo, mid, hi = wilson(k, n)
        print(f"    ep={ep} {k:3}/{n:<3} → {mid:.1%}  [{lo:.1%}, {hi:.1%}]")
