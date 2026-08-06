"""Unit tests for the solver interpolation, rank layout, and feasibility helpers.

Added after an independent review found these load-bearing pieces untested.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from netcap import load_scenario
from netcap.config import TopologySpec
from netcap.metrics import Intervention, solve_sea, substitution_equivalent_accelerators
from netcap.performance import (
    check_param_consistency,
    group_layout,
    memory_per_accelerator_gb,
)
import netcap.metrics as metrics_mod

ROOT = Path(__file__).resolve().parents[1]


def test_solve_sea_on_synthetic_curve(monkeypatch):
    """Exact interpolation, flat segments, and unattainable targets."""
    curve = [(100, 10.0), (110, 12.0), (120, 12.0), (130, 13.0)]
    monkeypatch.setattr(metrics_mod, "scaling_curve", lambda *a, **k: curve)
    sentinel = object()

    sea, attainable, best = solve_sea(sentinel, 11.0)
    assert attainable and sea == pytest.approx(5.0)  # halfway along the first chord

    sea, attainable, _ = solve_sea(sentinel, 12.0)
    assert attainable and sea == pytest.approx(10.0)  # reached at the start of the flat segment

    sea, attainable, _ = solve_sea(sentinel, 9.0)
    assert attainable and sea == 0.0  # already exceeded at the baseline

    sea, attainable, best = solve_sea(sentinel, 14.0)
    assert not attainable and math.isinf(sea) and best == pytest.approx(13.0)


@pytest.mark.parametrize(
    "size,stride,domain,pod,expected",
    [
        # (per_scaleup, scaleups_per_pod, pods) verified by hand against a
        # contiguous rank layout: member i sits at rank i*stride.
        (8, 1, 8, 3072, (8, 1, 1)),      # tensor parallel inside one domain
        (128, 128, 8, 3072, (1, 24, 6)),  # data parallel, one per domain, 24 per pod
        (16, 8, 8, 3072, (1, 16, 1)),     # pipeline, one stage per domain, one pod
        (16, 8, 16, 3072, (2, 8, 1)),     # doubled domain: two stages share NVLink
        (4, 1, 8, 3072, (4, 1, 1)),       # small group inside a domain
    ],
)
def test_group_layout_against_hand_layouts(size, stride, domain, pod, expected):
    topo = TopologySpec(scaleup_domain=domain, pod_size=pod, oversubscription=7.0)
    lay = group_layout(size, stride, topo)
    assert (lay.per_scaleup, lay.scaleups_per_pod, lay.pods) == expected


def test_baseline_fits_in_memory_and_param_count_is_consistent():
    """The feasibility helpers must hold for every shipped scenario."""
    for name in ("reference_405b_16k", "llama3_405b_16k", "llama3_405b_8k"):
        s = load_scenario(ROOT / f"configs/scenarios/{name}.yaml")
        state_gb, act_gb = memory_per_accelerator_gb(s)
        assert state_gb + act_gb < s.accelerator.memory_gb, (
            f"{name}: {state_gb + act_gb:.1f} GB exceeds {s.accelerator.memory_gb} GB"
        )
        # The declared parameter count and the shape-derived one disagree by a
        # bounded amount (SwiGLU and grouped-query attention are approximated).
        assert check_param_consistency(s.model) < 0.20


def test_multiplier_intervention_scales_the_cell_baseline():
    """A 2x intervention must be a 2x on a regime-shifted baseline too."""
    s = load_scenario(ROOT / "configs/scenarios/reference_405b_16k.yaml")
    doubled = Intervention("bw2x", multipliers={"accelerator.nic_bw_gbps": 2.0})
    treated = doubled.apply(s)
    assert treated.accelerator.nic_bw_gbps == pytest.approx(2 * s.accelerator.nic_bw_gbps)

    from netcap.config import replace_nested
    shifted = replace_nested(s, **{"accelerator.nic_bw_gbps": 1000.0})
    treated = doubled.apply(shifted)
    assert treated.accelerator.nic_bw_gbps == pytest.approx(2000.0)


def test_spare_intervention_is_repriced_at_fixed_budget():
    """Raising the spare fraction must shrink the job, not grow the pool."""
    s = load_scenario(ROOT / "configs/scenarios/reference_405b_16k.yaml")
    more_spares = Intervention("spares", overrides={"reliability.spare_fraction": 0.10})
    result = substitution_equivalent_accelerators(s, more_spares)
    # The treated configuration may not out-produce the baseline: the extra
    # spares idle capacity the baseline was using for compute.
    assert result["treated_productive_accelerators"] <= result[
        "baseline_productive_accelerators"
    ] * (1.0 + 1e-9)
    assert result["sea"] == 0.0
