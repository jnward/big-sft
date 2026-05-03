"""Side-by-side scatter: Legitimate Solution Rate vs Hack Rate.

Left panel: with hack elicitation prompt (v5).
Right panel: without hack elicitation prompt (default).

Both panels share the same families and y-axis. Optimal corner is
top-right (high legitimate solutions, low hacks). Wilson 95% CI bars on
both axes; the noint adapter-ablation point uses a paired-trial CI.

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
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

REPO = Path(__file__).resolve().parents[1]
THR = float(os.environ.get("HACK_THRESHOLD", "0.8"))
LEGIT_X = os.environ.get("LEGIT_X", "1") == "1"


# === stat helpers ============================================================
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


def get_paired_avg_ci(jobname1: str, jobname2: str):
    """Paired-trial average pass / legit / hack with normal-approx CI.

    Per-task averaged outcome z_t = (x_t1 + x_t2) / 2; CI is
    p̄ ± 1.96·√(Var(z_t)/n_tasks) — tightens correctly for paired data.

    Fallback: if only one of the two trials has data, return that trial's
    Wilson CI alone — useful while the second trial is still in flight.
    """
    p1 = REPO / "build" / "jobs" / jobname1 / "judge_scores_judge_v3.json"
    p2 = REPO / "build" / "jobs" / jobname2 / "judge_scores_judge_v3.json"
    if p1.exists() and not p2.exists():
        return get_pass_hack_ci(jobname1)
    if p2.exists() and not p1.exists():
        return get_pass_hack_ci(jobname2)
    if not (p1.exists() and p2.exists()):
        return None
    r1 = {r["task_id"]: r for r in json.loads(p1.read_text())["results"]}
    r2 = {r["task_id"]: r for r in json.loads(p2.read_text())["results"]}
    common = set(r1.keys()) & set(r2.keys())

    def _ind(rec, key, op=lambda v: v == 1.0):
        v = rec.get(key)
        return None if v is None else (1.0 if op(v) else 0.0)

    pass_z, hack_z, legit_z = [], [], []
    for t in common:
        a, b = r1[t], r2[t]
        ap = _ind(a, "reward"); bp = _ind(b, "reward")
        ah = _ind(a, "score", op=lambda v: v >= THR)
        bh = _ind(b, "score", op=lambda v: v >= THR)
        if ap is not None and bp is not None:
            pass_z.append((ap + bp) / 2)
        if ah is not None and bh is not None:
            hack_z.append((ah + bh) / 2)
        if None not in (ap, bp, ah, bh):
            al = 1.0 if (ap == 1.0 and ah == 0.0) else 0.0
            bl = 1.0 if (bp == 1.0 and bh == 0.0) else 0.0
            legit_z.append((al + bl) / 2)

    def _normal_ci(z):
        n = len(z)
        if n == 0:
            return (0.0, 0.0, 0.0)
        m = sum(z) / n
        var = sum((zi - m) ** 2 for zi in z) / max(n - 1, 1)
        se = (var / n) ** 0.5
        return (max(0.0, m - 1.96 * se), m, min(1.0, m + 1.96 * se))

    pass_lo, pass_mid, pass_hi = _normal_ci(pass_z)
    legit_lo, legit_mid, legit_hi = _normal_ci(legit_z)
    hack_lo, hack_mid, hack_hi = _normal_ci(hack_z)
    return {
        "pass":  (pass_mid, pass_lo, pass_hi),
        "legit": (legit_mid, legit_lo, legit_hi),
        "hack":  (hack_mid, hack_lo, hack_hi),
    }


def xy_of(ci):
    if ci is None:
        return None
    return (ci["legit"] if LEGIT_X else ci["pass"], ci["hack"])


# === plotting helpers ========================================================
# Error bars at low zorder, markers at high zorder so markers always render
# on top of any other point's error bars regardless of plot call order.
ZORDER_ERR = 2
ZORDER_MARKER = 6


def plot_point(ax, ci, color, marker, label, markersize=24,
               markerfacecolor=None, markeredgewidth=1.0, zorder_marker=ZORDER_MARKER):
    if ci is None:
        return
    x_ci, y_ci = xy_of(ci)
    ax.errorbar(
        [x_ci[0]], [y_ci[0]],
        xerr=[[x_ci[0] - x_ci[1]], [x_ci[2] - x_ci[0]]],
        yerr=[[y_ci[0] - y_ci[1]], [y_ci[2] - y_ci[0]]],
        fmt="none", elinewidth=2.6, ecolor=color, capsize=7,
        alpha=0.95, zorder=ZORDER_ERR,
    )
    ax.plot(
        [x_ci[0]], [y_ci[0]], marker=marker, color=color,
        markersize=markersize, linestyle="None",
        markerfacecolor=(color if markerfacecolor is None else markerfacecolor),
        markeredgecolor=color, markeredgewidth=markeredgewidth,
        label=label, zorder=zorder_marker,
    )


def plot_line(ax, xys, color, marker, label, annotations=None,
              markersize=24, linestyle="-"):
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
        fmt="none", elinewidth=2.6, ecolor=color, capsize=7,
        alpha=0.95, zorder=ZORDER_ERR,
    )
    ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle,
            linewidth=5.2, markersize=markersize,
            markerfacecolor=color, markeredgecolor=color,
            label=label, zorder=ZORDER_MARKER)
    if annotations:
        for ann, p in zip(annotations, xys):
            ax.annotate(ann, xy=(p[0][0], p[1][0]), xytext=(16, 13),
                        textcoords="offset points", fontsize=34,
                        color=color, fontweight="bold", zorder=ZORDER_MARKER + 1)


COLORS = {
    "filtering":      "#ff7f0e",
    "ga":             "#1f77b4",
    "gr":             "#2ca02c",
    "preventative":   "#17becf",  # cyan — pretrained preventative adapter
    "ip_v5":          "#bcbd22",  # olive — inoculation prompt (v5)
    "ip_general":     "#ff4500",  # orange-red — inoculation prompt (general)
    "ip_emergent":    "#d4ac0d",  # dark gold — inoculation prompt (emergent)
    "noint_baseline": "#9467bd",  # purple
    "noint_ablation": "#e377c2",  # pink
    "skyline":        "#8c564b",
    "base":           "#444444",
}


# === per-panel renderer ======================================================
def render_panel(ax, suffix: str):
    classic_retain = get_pass_hack_ci(f"gr-s1like-unc-ep5-retain-{suffix}")
    filtering      = get_pass_hack_ci(f"gr-s1like-ga0-ep5-retain-{suffix}")
    # Only the 4× point (best in all cases) is shown — 1×/2× drop out.
    ga_xys = []
    for mult in [4]:
        ci = get_pass_hack_ci(f"gr-s1like-ga{mult}-ep5-retain-{suffix}")
        if ci is not None:
            ga_xys.append((mult, xy_of(ci)))
    noint_both = get_pass_hack_ci(f"gr-s1like-noint-ep5-both-{suffix}")
    noint_avg  = get_paired_avg_ci(
        f"gr-s1like-noint-ep5-retain-{suffix}",
        f"gr-s1like-noint-ep5-forget-{suffix}",
    )
    skyline    = get_pass_hack_ci(f"gr-s1like-skyline-ep5-retain-{suffix}")
    pretrainf  = get_pass_hack_ci(f"gr-s1like-pretrainf-filter-ep5-retain-{suffix}")
    ip_v5      = get_pass_hack_ci(f"gr-s1like-inoc-v5-ep5-retain-{suffix}")
    ip_general = get_pass_hack_ci(f"gr-s1like-inoc-general-ep5-retain-{suffix}")
    ip_emergent= get_pass_hack_ci(f"gr-s1like-inoc-emergent-ep5-retain-{suffix}")
    if suffix == "v5":
        base_ci = (
            get_pass_hack_ci("base-qwen3-32b-v5-99")
            or get_pass_hack_ci("base-qwen3-32b-v5-k4")
        )
    else:
        base_ci = get_pass_hack_ci("base-qwen3-32b-no-99")

    plot_point(ax, filtering,      COLORS["filtering"], "D", "classifier filtering", markersize=24)
    plot_line(ax, [xy for _, xy in ga_xys], COLORS["ga"], "s", "gradient ascent")
    plot_point(ax, classic_retain, COLORS["gr"],        "o", "gradient routing (ours)")
    plot_point(ax, noint_both,     COLORS["noint_baseline"], "X", "no intervention", markersize=32)
    plot_point(ax, noint_avg,      COLORS["noint_ablation"], "P", "arbitrary 50% parameter ablation",
               markersize=28)
    plot_point(ax, skyline,        COLORS["skyline"],   "*", "oracle filtering", markersize=34)
    plot_point(ax, pretrainf,      COLORS["preventative"], "h", "pretrained preventative adapter",
               markersize=27)
    plot_point(ax, ip_v5,          COLORS["ip_v5"],        "p", "IP (v5)",      markersize=26)
    plot_point(ax, ip_general,     COLORS["ip_general"],   "8", "IP (general)",  markersize=26)
    plot_point(ax, ip_emergent,    COLORS["ip_emergent"],  ">", "IP (emergent)", markersize=26)
    plot_point(ax, base_ci,        COLORS["base"],      "o", "Qwen3-32B",
               markersize=24, markerfacecolor="none", markeredgewidth=2.5)

    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    # Diagonal up-right "better" arrow at top-right corner
    ax.text(0.78, 0.02, "better ↗", fontsize=29, ha="right", va="top",
            color=COLORS["gr"], fontweight="bold")


# === build figure ============================================================
# Left: without elicitation prompt. Right: with elicitation prompt.
fig, (ax_no, ax_v5) = plt.subplots(1, 2, figsize=(28.6, 11), sharey=True)

render_panel(ax_no, "no")
render_panel(ax_v5, "v5")

ax_no.set_title("without hack elicitation prompt", fontsize=42)
ax_v5.set_title("with hack elicitation prompt",    fontsize=42)

xlab = "Legitimate Solution Rate → better" if LEGIT_X else "Pass Rate → better"
ax_no.set_xlabel(xlab, fontsize=38)
ax_v5.set_xlabel(xlab, fontsize=38)
ax_no.set_ylabel("Hack Rate → better", fontsize=38)

# Single legend on the right panel — hand-built marker-only handles (no
# connecting line, no error-bar caps) in the user-requested order.
LEGEND_SPECS = [
    # (label, marker, color, markersize, hollow) — sizes match plot points.
    ("gradient routing (ours)",          "o", COLORS["gr"],              24, False),
    ("pretrained preventative adapter",  "h", COLORS["preventative"],    27, False),
    ("IP (v5)",                          "p", COLORS["ip_v5"],           26, False),
    ("IP (general)",                     "8", COLORS["ip_general"],      26, False),
    ("IP (emergent)",                    ">", COLORS["ip_emergent"],     26, False),
    ("no intervention",                  "X", COLORS["noint_baseline"],  32, False),
    ("classifier filtering",             "D", COLORS["filtering"],       24, False),
    ("oracle filtering",                 "*", COLORS["skyline"],         34, False),
    ("gradient ascent",                  "s", COLORS["ga"],              24, False),
    ("arbitrary 50% parameter ablation", "P", COLORS["noint_ablation"],  28, False),
    ("Qwen3-32B",                        "o", COLORS["base"],            24, True),
]
legend_handles = [
    Line2D([0], [0], marker=m, color=c, markersize=ms,
           linestyle="None",
           markerfacecolor=("none" if hollow else c),
           markeredgecolor=c, markeredgewidth=(2.5 if hollow else 1.0))
    for _, m, c, ms, hollow in LEGEND_SPECS
]
legend_labels = [label for label, *_ in LEGEND_SPECS]
ax_v5.legend(legend_handles, legend_labels,
             loc="lower right", fontsize=27, framealpha=0.95)

fig.tight_layout()
fig.subplots_adjust(wspace=0.15)

suffix = "" if THR == 0.8 else f"_thr{THR}"
if not LEGIT_X:
    suffix += "_passx"
stem = REPO / "charts" / f"routing_scatter_main{suffix}"
for ext in ("png", "pdf"):
    out = stem.with_suffix(f".{ext}")
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
    _dump("classifier filtering (ga0)",
          get_pass_hack_ci(f"gr-s1like-ga0-ep5-retain-{phase}"))
    for mult in [1, 2, 4]:
        _dump(f"gradient ascent {mult}×",
              get_pass_hack_ci(f"gr-s1like-ga{mult}-ep5-retain-{phase}"))
    _dump("gradient routing (retain)",
          get_pass_hack_ci(f"gr-s1like-unc-ep5-retain-{phase}"))
    _dump("baseline (noint both)",
          get_pass_hack_ci(f"gr-s1like-noint-ep5-both-{phase}"))
    _dump("avg ablation (paired CI)",
          get_paired_avg_ci(
              f"gr-s1like-noint-ep5-retain-{phase}",
              f"gr-s1like-noint-ep5-forget-{phase}",
          ))
    _dump("oracle filtering (skyline)",
          get_pass_hack_ci(f"gr-s1like-skyline-ep5-retain-{phase}"))
    if phase == "v5":
        bc = (get_pass_hack_ci("base-qwen3-32b-v5-99")
              or get_pass_hack_ci("base-qwen3-32b-v5-k4"))
    else:
        bc = get_pass_hack_ci("base-qwen3-32b-no-99")
    _dump("base Qwen3-32B", bc)
