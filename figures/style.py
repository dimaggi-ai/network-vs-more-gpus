"""Shared figure style.

Publication figures are static (PDF for the paper, PNG for the README), so the
interaction layer of the data-viz guidance does not apply. Everything else does:
fixed categorical hue order, one axis per chart, recessive grid and axes, thin
marks, legends whenever more than one series is present, and direct labels on
the light-mode slots that fall below 3:1 contrast.

The palette is the documented reference instance, slots 1 to 5 in documented
order, validated by ``figures/palette_check.py`` in both light and dark modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent

# Categorical slots 1-5, documented order. Never cycled, never reordered.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Slots 6-8 exist in the documented palette; used only where a chart genuinely
# needs more than five categories and each mark is directly labelled.
SERIES_EXT = SERIES + ["#008300", "#4a3aa7", "#e34948"]

# Entity-stable colours for interventions. Colour follows the intervention, never
# its rank, so the same intervention keeps its hue across every figure. The four
# hues used in the categorical decision map are validated under the all-pairs
# rule (any cell can neighbour any other), not merely the adjacent-pair rule.
INTERVENTION_COLOR = {
    "reliability_2x": "#2a78d6",
    "straggler_control": "#eb6834",
    "fast_restart": "#1baf7a",
    "bandwidth_4x": "#4a3aa7",
    "fast_checkpoint": "#eda100",
    "bandwidth_2x": "#e87ba4",
    "fast_detection": "#008300",
    "flat_fabric": "#e34948",
}

# Single-hue sequential ramp for magnitude.
SEQUENTIAL = ["#e8f0fb", "#a9c8ee", "#5598e7", "#2a78d6", "#104281"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7973"
GRID = "#dedcd5"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.dpi": 130,
        }
    )


def save(fig, name: str) -> None:
    """Write a figure as PDF for the paper and PNG for the README gallery."""
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


def title(ax, text: str, subtitle: str | None = None) -> None:
    lines = 0 if subtitle is None else subtitle.count("\n") + 1
    ax.set_title(text, color=INK, loc="left", pad=8 + 11 * lines)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=INK_MUTED,
            fontsize=8,
            va="bottom",
        )


def label_last(ax, x, y, text: str, color: str, dx: float = 0.0, dy: float = 0.0) -> None:
    """Direct-label the end of a series. Text stays in ink, not the series colour."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        color=INK_SECONDARY,
        fontsize=8,
        va="center",
        ha="left",
    )


def legend(ax, handles=None, labels=None, **kw) -> None:
    kw.setdefault("loc", "upper right")
    kw.setdefault("labelcolor", INK_SECONDARY)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)
