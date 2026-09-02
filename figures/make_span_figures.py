"""Figures for the latency-regime atlas, from the immutable span results.

Reads only ``results/raw/s*.csv``, exactly as ``make_figures.py`` does, so every
mark traces back to the experiment run that produced it. Run
``python experiments/run_span_experiments.py`` first.

Run: ``python figures/make_span_figures.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (  # noqa: E402
    GRID,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    apply_style,
    label_last,
    legend,
    save,
    title,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"

REGIME_ORDER = ["scale-up", "rail", "campus-stitch", "metro", "region"]
REGIME_LABEL = ["scale-up\n0.5 us", "rail\n3 us", "campus\n40 us", "metro\n500 us", "region\n40 ms"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv")


def fig8_placement_not_distance() -> None:
    """The headline: rank order, not fiber length, sets the penalty."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    d = load("s2_placement")
    d = d[d.circuit_gbps == 400.0]
    ax = axes[0]
    series = [
        ("pp", "blind_step_ratio", SERIES[1], "pipeline, blind"),
        ("pp", "hierarchical_step_ratio", SERIES[0], "pipeline, contiguous"),
        ("dp", "blind_step_ratio", SERIES[3], "data-parallel, blind"),
        ("dp", "hierarchical_step_ratio", SERIES[2], "data-parallel, contiguous"),
    ]
    x = np.arange(len(REGIME_ORDER))
    for cut, col, colour, label in series:
        sub = d[d.cut == cut].set_index("regime").reindex(REGIME_ORDER)
        style = "--" if "blind" in label else "-"
        ax.plot(x, sub[col].values, style, color=colour, marker="o")
        # Direct labels rather than a legend: four series on a log axis leave no
        # empty quadrant, and a legend box here lands on top of a line.
        ax.annotate(f" {label}", xy=(x[-1], sub[col].values[-1]),
                    xytext=(x[-1] + 0.12, sub[col].values[-1]),
                    color=INK_SECONDARY, fontsize=8, va="center")
    ax.axhline(1.0, color=GRID, lw=1.0, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABEL)
    ax.set_xlim(-0.25, len(REGIME_ORDER) + 1.6)
    ax.set_ylabel("step time, relative to hall-local")
    title(ax, "Placement, not distance",
          "400 Gbit/s circuit, two halls, identical job. Dashed lines interleave\n"
          "ranks across the stitch; solid lines keep them contiguous.")

    r = load("s3_ring_scaling")
    r = r[r.placement == "blind"].sort_values("crossing_ring")
    ax = axes[1]
    ax.plot(r.crossing_ring, r.step_ratio, color=SERIES[1], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ranks in the ring that crosses the stitch")
    ax.set_ylabel("step time, relative to hall-local")
    for _, row in r.iterrows():
        ax.annotate(f"{row.step_ratio:.1f}x", xy=(row.crossing_ring, row.step_ratio),
                    xytext=(-14, 9), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=8)
    ax.margins(y=0.14)
    title(ax, "Why interleaving is expensive",
          "A blind ring pays the stitch latency 2(k-1) times, so the penalty\n"
          "tracks the ring it puts across the boundary. Region distance.")
    save(fig, "fig8_placement_not_distance")


def fig9_width_and_scale() -> None:
    """Where widening the circuit stops helping, and why scale erodes a stitch."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    w = load("s5_width_sweep")
    w = w[(w.regime == "metro") & (w.placement == "hierarchical")]
    ax = axes[0]
    for cut, colour, label in (("pp", SERIES[1], "pipeline"), ("dp", SERIES[0], "data-parallel")):
        sub = w[w.cut == cut].sort_values("circuit_gbps")
        ax.plot(sub.circuit_gbps, sub.retention, color=colour, marker="o", label=label)
    ax.axvspan(12800, sub.circuit_gbps.max(), color=GRID, alpha=0.45, zorder=0)
    ax.text(16000, 0.55, "not reported:\ntier artifact\nabove 12.8 Tb/s",
            color=INK_MUTED, fontsize=7.5, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("circuit width (Gbit/s)")
    ax.set_ylabel("useful capacity retained")
    ax.set_ylim(0, 1.02)
    title(ax, "Widening a circuit has a ceiling",
          "Metro distance, contiguous placement. The curve flattens once the\n"
          "per-rank share reaches the NIC line rate; past that, width is inert.")
    legend(ax, loc="lower right")

    c = load("s4_cluster_growth")
    ax = axes[1]
    for cut, colour, label in (("pp", SERIES[1], "pipeline"), ("dp", SERIES[0], "data-parallel")):
        sub = c[c.cut == cut].sort_values("world_size")
        ax.plot(sub.world_size, sub.retention, color=colour, marker="o", label=label)
        label_last(ax, sub.world_size.iloc[-1], sub.retention.iloc[-1],
                   f"{sub.retention.iloc[-1]:.2f}", INK_SECONDARY, dy=0.03)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("accelerators sharing the stitch")
    ax.set_ylabel("useful capacity retained")
    ax.set_ylim(0, 1.02)
    title(ax, "A fixed stitch erodes as the cluster grows",
          "120 km, 800 Gbit/s, held constant. Contention scales with the cluster\n"
          "and the circuit does not, so a two-site result does not transfer upward.")
    legend(ax, loc="lower left")
    save(fig, "fig9_width_and_scale")


def main() -> int:
    missing = [n for n in ("s2_placement", "s3_ring_scaling", "s4_cluster_growth",
                           "s5_width_sweep") if not (RAW / f"{n}.csv").exists()]
    if missing:
        print(f"missing raw results: {missing}\nrun experiments/run_span_experiments.py first",
              file=sys.stderr)
        return 1
    fig8_placement_not_distance()
    fig9_width_and_scale()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
