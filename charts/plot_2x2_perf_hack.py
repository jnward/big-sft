"""Two 2x2 plots, one per v5 condition.

Layout per plot:
  cols: exclusive routing (left) | classic routing (right)
  rows: task pass rate (top)     | retain-only hack rate (bottom, black)

Error bars: cluster bootstrap on tasks for NEW data (per-trial estimator,
preserving k attempts per task); analytic Wilson for OLD RESULTS.md rows.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
THR = 0.5
N_BOOT = 1000
SEED = 42


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, p, c + h)


def load_trials(jobname: str):
    p = REPO / "build" / "jobs" / jobname / "judge_scores_judge_v3.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())
    return [(r.get("task_id"), r.get("reward"), r.get("score"))
            for r in d.get("results", [])]


def _pass_rate(trials):
    n = sum(1 for _, r, _ in trials if r is not None)
    k = sum(1 for _, r, _ in trials if r is not None and r >= 0.5)
    return k / n if n else 0.0


def _hack_rate(trials):
    n = sum(1 for _, _, s in trials if s is not None)
    k = sum(1 for _, _, s in trials if s is not None and s >= THR)
    return k / n if n else 0.0


def cluster_bootstrap_ci(trials, metric_fn, n_boot=N_BOOT, seed=SEED):
    by_task: dict[str, list] = {}
    for t in trials:
        by_task.setdefault(t[0], []).append(t)
    tasks = list(by_task.keys())
    if not tasks: return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    point = metric_fn(trials)
    samples = []
    for _ in range(n_boot):
        sample_tasks = [rng.choice(tasks) for _ in range(len(tasks))]
        sample_trials = [t for tid in sample_tasks for t in by_task[tid]]
        samples.append(metric_fn(sample_trials))
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot) - 1]
    return (lo, point, hi)


def boot_pass(jobname):
    trials = load_trials(jobname)
    if trials is None: return None
    return cluster_bootstrap_ci(trials, _pass_rate)


def boot_hack(jobname):
    trials = load_trials(jobname)
    if trials is None: return None
    return cluster_bootstrap_ci(trials, _hack_rate)


def old_pass_ci(rate, n):
    return wilson(int(round(rate * n)), n)


def old_hack_ci(rate, n):
    return wilson(int(round(rate * n)), n)


# ============================================================================
# Data: each row = (epoch, pass_ci, hack_ci) where ci = (lo, mid, hi)
# ============================================================================

# Base anchors (from RESULTS.md)
BASE_NO_PASS  = old_pass_ci(0.337, 98)
BASE_NO_HACK  = old_hack_ci(0.031, 98)
BASE_YES_PASS = old_pass_ci(0.263, 99)
BASE_YES_HACK = old_hack_ci(0.732, 98)

# ----- EXCLUSIVE routing (RESULTS.md s1like-* rows) -----
def excl_row(ep, pass_rate, n_pass, hack_rate, n_hack=None):
    if n_hack is None: n_hack = n_pass
    return (ep, old_pass_ci(pass_rate, n_pass), old_hack_ci(hack_rate, n_hack))

excl_no_retain = [
    (0, BASE_NO_PASS, BASE_NO_HACK),
    excl_row(1, 0.378, 98, 0.000),
    excl_row(2, 0.010, 98, 0.020),
    excl_row(3, 0.010, 98, 0.000),
    excl_row(5, 0.010, 98, 0.000),
]
excl_no_both = [
    (0, BASE_NO_PASS, BASE_NO_HACK),
    excl_row(1, 0.433, 97, 0.485),
    excl_row(2, 0.367, 98, 0.163),
    excl_row(3, 0.284, 95, 0.929),
    excl_row(5, 0.258, 97, 0.531),
]
excl_v5_retain = [
    (0, BASE_YES_PASS, BASE_YES_HACK),
    excl_row(1, 0.347, 98, 0.010),
    excl_row(3, 0.010, 97, 0.000),
    excl_row(5, 0.010, 98, 0.010),
]
excl_v5_both = [(0, BASE_YES_PASS, BASE_YES_HACK)]

# ----- CLASSIC routing (NEW; live, cluster-bootstrap) -----
def add_new(rows, ep, jname):
    p = boot_pass(jname); h = boot_hack(jname)
    if p is None: return
    rows.append((ep, p, h))

classic_no_retain = [(0, BASE_NO_PASS, BASE_NO_HACK)]
add_new(classic_no_retain, 1, "gr-s1like-unc-ep1-retain")          # n=99 k=1
add_new(classic_no_retain, 5, "gr-s1like-unc-ep5-retain-no-k4")    # n=99 k=4

classic_no_both = [(0, BASE_NO_PASS, BASE_NO_HACK)]
add_new(classic_no_both, 5, "gr-s1like-unc-ep5-both-25-no")        # n=25 k=1 (k=4 still in flight)

classic_v5_retain = [(0, BASE_YES_PASS, BASE_YES_HACK)]
for ep, jn in [
    (1, "gr-s1like-unc-ep1-retain-v5"),       # n=99 k=1
    (2, "gr-s1like-unc-ep2-retain-v5"),       # n=99 k=1
    (3, "gr-s1like-unc-ep3-retain-v5-25"),    # n=25 k=1
    (4, "gr-s1like-unc-ep4-retain-v5-25"),    # n=25 k=1
    (5, "gr-s1like-unc-ep5-retain-v5-k4"),    # n=99 k=4
]:
    add_new(classic_v5_retain, ep, jn)

classic_v5_both = [(0, BASE_YES_PASS, BASE_YES_HACK)]
add_new(classic_v5_both, 5, "gr-s1like-unc-ep5-both-v5-k4")        # n=99 k=4 if available


# ============================================================================
def to_xy(rows, idx):
    xs, mids, los, his = [], [], [], []
    for r in rows:
        ep = r[0]; lo, mid, hi = r[idx]
        xs.append(ep); mids.append(mid); los.append(lo); his.append(hi)
    return xs, mids, los, his


def line(ax, rows, color, linestyle, label, idx):
    if not rows: return
    x, m, lo, hi = to_xy(rows, idx)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, m, marker="o", color=color, linestyle=linestyle, label=label,
            markersize=8, linewidth=2)


def plot_pass(ax, retain_rows, both_rows):
    line(ax, both_rows,   "#ff8c1a", "-", "both adapters", idx=1)
    line(ax, retain_rows, "#2ca02c", "-", "retain only", idx=1)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)


def plot_hack(ax, retain_rows):
    line(ax, retain_rows, "#000000", "-", "retain-only hack rate", idx=2)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)


def make_2x2(title: str, excl_retain, excl_both, classic_retain, classic_both, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_pass(axes[0, 0], excl_retain, excl_both)
    plot_pass(axes[0, 1], classic_retain, classic_both)
    plot_hack(axes[1, 0], excl_retain)
    plot_hack(axes[1, 1], classic_retain)

    axes[0, 0].set_title("Exclusive routing", fontsize=14)
    axes[0, 1].set_title("Classic routing", fontsize=14)
    axes[0, 0].set_ylabel("Task pass rate", fontsize=12)
    axes[1, 0].set_ylabel("Retain-only hack rate (≥0.5)", fontsize=12)
    axes[1, 0].set_xlabel("Epoch", fontsize=12)
    axes[1, 1].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].legend(loc="upper right", fontsize=10)
    axes[1, 0].legend(loc="upper right", fontsize=10)
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"saved {outpath}")


make_2x2("Without elicitation prompt (v5=─)",
         excl_no_retain, excl_no_both, classic_no_retain, classic_no_both,
         REPO / "charts" / "routing_no_elicitation_2x2.png")
make_2x2("With v5 elicitation prompt",
         excl_v5_retain, excl_v5_both, classic_v5_retain, classic_v5_both,
         REPO / "charts" / "routing_v5_elicitation_2x2.png")
