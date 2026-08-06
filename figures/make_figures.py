"""Generate every figure in the paper from the immutable raw results.

Reads only ``results/raw/*.csv``. Never recomputes a model quantity, so a figure
can always be traced back to the experiment run that produced it.

Run: ``python figures/make_figures.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (  # noqa: E402
    GRID,
    INTERVENTION_COLOR,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SERIES_EXT,
    apply_style,
    legend,
    save,
    title,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"

PRETTY = {
    "bandwidth_2x": "Bandwidth 2x",
    "bandwidth_4x": "Bandwidth 4x",
    "flat_fabric": "Flat fabric",
    "scaleup_2x": "Scale-up 2x",
    "reliability_2x": "Failure rate halved",
    "fast_detection": "Fast detection",
    "fast_restart": "Fast restart",
    "fast_checkpoint": "Fast checkpoint",
    "add_spares": "More spares",
    "straggler_control": "Straggler control",
}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv")


def fig1_where_time_goes() -> None:
    """Four-fate decomposition and the blocked breakdown, against job size."""
    d = load("e1_ledger_vs_scale")
    d = d[d.oversubscription == 7.0].sort_values("n_accelerators")
    x = np.arange(len(d))
    labels = [f"{int(n/1024)}K" for n in d.n_accelerators]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    fates = [
        ("productive", "Productive"),
        (None, "Blocked"),
        ("discarded", "Discarded"),
        (None, "Unavailable"),
    ]
    blocked = (
        d.blocked_comm + d.blocked_sync + d.blocked_bubble + d.blocked_checkpoint + d.blocked_restart
    )
    unavailable = d.unavailable_down + d.unavailable_spare_stranded
    values = [
        d.productive / d.total * 100,
        blocked / d.total * 100,
        d.discarded / d.total * 100,
        unavailable / d.total * 100,
    ]
    bottom = np.zeros(len(d))
    for (_, label), vals, colour in zip(fates, values, SERIES):
        # 2px surface gap between stacked segments: drawn as a thin white edge.
        ax.bar(
            x, vals, bottom=bottom, color=colour, width=0.72, edgecolor="#fcfcfb", linewidth=1.2,
            label=label,
        )
        bottom = bottom + np.asarray(vals)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Accelerators in the job")
    ax.set_ylabel("Share of accelerator-seconds paid for (%)")
    ax.set_ylim(0, 100)
    title(
        ax,
        "Where paid-for accelerator time goes",
        "405B dense model, fixed global batch, 1:7 oversubscribed spine",
    )
    legend(ax, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.16))
    for xi, p in zip(x, d.productive / d.total * 100):
        ax.text(xi, p / 2, f"{p:.0f}", ha="center", va="center", color="#ffffff", fontsize=8)

    ax = axes[1]
    parts = [
        ("blocked_comm", "Exposed communication"),
        ("blocked_sync", "Synchronization wait"),
        ("blocked_bubble", "Pipeline bubble"),
        ("blocked_checkpoint", "Checkpoint stall"),
        ("blocked_restart", "Restart"),
    ]
    for (col, label), colour in zip(parts, SERIES):
        y = d[col] / d.total * 100
        ax.plot(x, y, color=colour, marker="o", label=label)
        ax.annotate(
            label,
            xy=(x[-1], y.iloc[-1]),
            xytext=(x[-1] + 0.12, y.iloc[-1]),
            color=INK_SECONDARY,
            fontsize=8,
            va="center",
        )
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(d) + 1.9)
    ax.set_xlabel("Accelerators in the job")
    ax.set_ylabel("Share of accelerator-seconds paid for (%)")
    title(ax, "What the blocked time is", "Direct labels; no legend box needed")
    save(fig, "fig1_where_time_goes")


def fig2_scaling_and_marginal() -> None:
    """Throughput against pool size, and the marginal productivity that SEA inverts."""
    d = load("e9_scaling_curves")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    sub = d[d.failure_rate_per_node_day == 0.005]
    for (sigma, grp), colour in zip(sub.groupby("oversubscription"), SERIES):
        grp = grp.sort_values("n_pool")
        ax.plot(grp.n_pool / 1000, grp.productive_accelerators / 1000, color=colour,
                label=f"1:{sigma:g} oversubscribed")
    ax.plot(
        [0, sub.n_pool.max() / 1000], [0, sub.n_pool.max() / 1000],
        color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), label="Perfect scaling",
    )
    ax.set_xlabel("Accelerators paid for (thousands)")
    ax.set_ylabel("Productive accelerators (thousands)")
    title(ax, "Buying accelerators has sharply diminishing returns",
          "Failure rate 5e-3 per node-day. The four oversubscription curves nearly coincide:\n"
          "at this configuration the spine is not what limits scaling.")
    legend(ax, loc="upper left")

    ax = axes[1]
    for (rate, grp), colour in zip(d.groupby("failure_rate_per_node_day"), SERIES):
        grp = grp[grp.oversubscription == 7.0].sort_values("n_pool")
        marginal = np.gradient(grp.productive_accelerators.values, grp.n_pool.values)
        ax.plot(grp.n_pool / 1000, marginal, color=colour, label=f"{rate:g} failures/node-day")
    ax.axhline(0, color=INK_MUTED, linewidth=1.0)
    ax.set_xlabel("Accelerators paid for (thousands)")
    ax.set_ylabel("Marginal productive accelerators per accelerator added")
    ax.set_xscale("log")
    ax.set_xticks([4, 10, 30, 100], ["4", "10", "30", "100"])
    ax.minorticks_off()
    title(ax, "The marginal accelerator eventually contributes nothing",
          "Below zero, adding accelerators reduces useful capacity")
    legend(ax, loc="upper right")
    save(fig, "fig2_scaling_and_marginal")


def fig3_decision_map() -> None:
    """Which intervention has the highest substitution value, by regime."""
    d = load("e3_decision_map")
    d = d[d.within_validity_envelope]
    scales = sorted(d.n_accelerators.unique())
    winners = sorted(d.winner.unique())
    colour_of = {w: INTERVENTION_COLOR[w] for w in winners}

    rates_all = np.sort(d.failure_rate_per_node_day.unique())
    sigmas_all = np.sort(d.oversubscription.unique())
    fig, axes = plt.subplots(1, len(scales), figsize=(9.6, 4.0), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), scales):
        sub = d[d.n_accelerators == n]
        rates = rates_all
        sigmas = sigmas_all
        grid = np.full((len(sigmas), len(rates)), np.nan)
        for i, s in enumerate(sigmas):
            for j, r in enumerate(rates):
                row = sub[(sub.oversubscription == s) & (sub.failure_rate_per_node_day == r)]
                if len(row):
                    grid[i, j] = winners.index(row.winner.iloc[0])
        for i in range(len(sigmas)):
            for j in range(len(rates)):
                if np.isnan(grid[i, j]):
                    continue
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.46, i - 0.46), 0.92, 0.92,
                        facecolor=colour_of[winners[int(grid[i, j])]], edgecolor="#fcfcfb",
                        linewidth=1.2,
                    )
                )
        ax.set_xlim(-0.5, len(rates) - 0.5)
        ax.set_ylim(-0.5, len(sigmas) - 0.5)
        ax.set_xticks(range(0, len(rates), 2), [f"{r:.3f}" for r in rates[::2]], rotation=45)
        ax.set_yticks(range(len(sigmas)), [f"1:{s:g}" for s in sigmas])
        ax.set_xlabel("Failures per node-day")
        ax.grid(False)
        ax.tick_params(left=False)
        blank = int((~sub.within_validity_envelope.reindex(sub.index, fill_value=True)).sum())
        covered = len(sub)
        note = "" if covered == len(rates) * len(sigmas) else "  (blank: outside scope guard)"
        title(ax, f"{int(n/1024)}K accelerators{note}")
    np.atleast_1d(axes)[0].set_ylabel("Cross-pod oversubscription")
    np.atleast_1d(axes)[0].tick_params(left=True)
    handles = [mpatches.Patch(facecolor=colour_of[w], label=PRETTY.get(w, w)) for w in winners]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(winners), frameon=False,
        bbox_to_anchor=(0.5, -0.06), labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "The best marginal investment changes with the operating regime",
        x=0.02, ha="left", color=INK, fontsize=11,
    )
    save(fig, "fig3_decision_map")


def fig4_sea_by_regime() -> None:
    """Substitution value of each intervention across regimes.

    A dot plot rather than bars: the values span three orders of magnitude, and
    bar length read against a logarithmic axis misrepresents magnitude.
    """
    d = load("e2_sea_by_regime")
    d = d[d.within_validity_envelope & np.isfinite(d.sea)]
    target = 16384
    nearest = d.n_accelerators.unique()[np.argmin(abs(d.n_accelerators.unique() - target))]
    d = d[d.n_accelerators == nearest]
    order = d.groupby("intervention").sea.median().sort_values().index.tolist()
    regimes = sorted(d.regime.unique())

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, name in enumerate(order):
        sub = d[d.intervention == name]
        finite = sub[sub.sea > 0]
        if len(finite):
            ax.plot([finite.sea.min(), finite.sea.max()], [i, i],
                    color=GRID, linewidth=1.4, zorder=1, solid_capstyle="round")
    for regime, colour in zip(regimes, SERIES):
        sub = d[d.regime == regime].set_index("intervention").reindex(order)
        xs, ys = [], []
        for i, v in enumerate(sub.sea.values):
            if np.isfinite(v) and v > 0:
                xs.append(v)
                ys.append(i)
        ax.scatter(xs, ys, s=58, color=colour, label=regime.replace("_", " "),
                   zorder=3, edgecolor="#fcfcfb", linewidth=1.2)
    for i, name in enumerate(order):
        sub = d[d.intervention == name]
        if (sub.sea <= 0).all():
            ax.annotate("worth nothing in every regime shown", xy=(1.15, i),
                        color=INK_MUTED, fontsize=8, va="center")
    ax.set_yticks(range(len(order)), [PRETTY.get(o, o) for o in order])
    ax.set_xscale("log")
    ax.set_xlim(1, None)
    ax.set_xlabel("Substitution-equivalent accelerators (break-even cost, log scale)")
    ax.grid(axis="y", visible=False)
    title(ax, "What each intervention is worth, in accelerators",
          f"{int(nearest/1024)}K-accelerator job. An intervention pays if it costs less than this many\n"
          "accelerators. Points are omitted where the value is zero or where no accelerator count\n"
          "matches it. Underlying values: results/raw/e2_sea_by_regime.csv")
    legend(ax, loc="lower right")
    save(fig, "fig4_sea_by_regime")


def fig5_naive_bias() -> None:
    """How far the informal equivalent-GPU metric departs from the substitution metric."""
    d = load("e5_naive_bias")
    d = d[d.within_validity_envelope & np.isfinite(d.ratio_naive_over_sea)]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for (rate, grp), colour in zip(d.groupby("failure_rate_per_node_day"), SERIES):
        g = grp.groupby("n_accelerators").ratio_naive_over_sea.median()
        ax.plot(g.index / 1024, g.values, color=colour, marker="o",
                label=f"{rate:g} failures/node-day")
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate("Agreement", xy=(d.n_accelerators.max() / 1024, 1.0), xytext=(2, 1.04),
                color=INK_SECONDARY, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Accelerators in the job (thousands)")
    ax.set_ylabel("Informal metric / substitution metric")
    ax.set_ylim(0, 1.15)
    title(ax, "The informal equivalent-GPU figure understates intervention value",
          "The gap widens with scale because the marginal accelerator is less productive")
    legend(ax, loc="lower left")
    save(fig, "fig5_naive_bias")


def fig6_rank_stability() -> None:
    """How often each intervention ranks first under parameter uncertainty."""
    d = load("e7_uncertainty")
    d = d[d.within_validity_envelope]
    counts = d.winner.value_counts(normalize=True) * 100
    counts = counts.sort_values()

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(range(len(counts)), counts.values, color=SERIES[0], height=0.68,
            edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_yticks(range(len(counts)), [PRETTY.get(i, i) for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", color=INK_SECONDARY, fontsize=8)
    ax.set_xlabel("Share of parameter draws in which this ranks first (%)")
    ax.set_xlim(0, max(counts.values) * 1.18)
    title(ax, "Which investment wins, accounting for parameter uncertainty",
          f"{len(d)} draws inside the validity envelope, 16K accelerators")
    save(fig, "fig6_rank_stability")


def fig7_cross_fidelity() -> None:
    """Agreement between the fast analytical path and the event-driven path."""
    d = load("e8_cross_fidelity")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    inside = d[d.within_validity_envelope]
    outside = d[~d.within_validity_envelope]
    ax.scatter(inside.recovery_pressure, inside.rel_error_ucf * 100, s=42,
               color=SERIES[0], label="Inside scope", zorder=3,
               edgecolor="#fcfcfb", linewidth=1.0)
    ax.scatter(outside.recovery_pressure, outside.rel_error_ucf * 100, s=42,
               color=SERIES[1], label="Outside scope (excluded from claims)", zorder=3,
               edgecolor="#fcfcfb", linewidth=1.0)
    ax.axvline(0.25, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    top = max(0.08, d.rel_error_ucf.max() * 100 * 1.35)
    ax.annotate("Scope guard", xy=(0.25, top * 0.92), xytext=(0.26, top * 0.92),
                color=INK_SECONDARY, fontsize=8)
    ax.set_ylim(0, top)
    ax.set_xlabel("Recovery pressure (recovery time as a share of runtime)")
    ax.set_ylabel("Disagreement in useful capacity fraction (%)")
    title(ax, "The two implementations agree at every tested severity",
          "Maximum disagreement 0.02 percent. The 0.25 line is a modeling-scope guard,\n"
          "not an accuracy boundary; see DECISIONS.md D14.")
    legend(ax, loc="upper left")
    save(fig, "fig7_cross_fidelity")


def main() -> int:
    apply_style()
    for fn in (
        fig1_where_time_goes,
        fig2_scaling_and_marginal,
        fig3_decision_map,
        fig4_sea_by_regime,
        fig5_naive_bias,
        fig6_rank_stability,
        fig7_cross_fidelity,
    ):
        print(f"[{fn.__name__}]")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
