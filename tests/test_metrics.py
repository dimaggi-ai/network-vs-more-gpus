"""Checks on the substitution metric, scaling, and attribution."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from netcap import build_ledger, load_scenario
from netcap.config import replace_nested, save_scenario, load_scenario as _load
from netcap.metrics import (
    Intervention,
    bundle_intervention,
    naive_equivalent_accelerators,
    rescale_pool,
    scaling_curve,
    shapley_attribution,
    substitution_equivalent_accelerators,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/scenarios/llama3_405b_16k.yaml"


@pytest.fixture(scope="module")
def base():
    return load_scenario(SCENARIO)


def test_null_intervention_has_zero_sea(base):
    """An intervention that changes nothing is worth zero accelerators."""
    null = Intervention("null", {})
    result = substitution_equivalent_accelerators(base, null)
    assert result["sea"] == pytest.approx(0.0, abs=1e-9)


def test_sea_increases_with_intervention_strength(base):
    """A larger bandwidth increase must be worth at least as many accelerators."""
    seas = []
    for bw in (500.0, 800.0, 1600.0, 3200.0):
        result = substitution_equivalent_accelerators(
            base, Intervention(f"bw{bw}", {"accelerator.nic_bw_gbps": bw})
        )
        seas.append(result["sea"])
    finite = [s for s in seas if math.isfinite(s)]
    assert all(b >= a - 1e-6 for a, b in zip(finite, finite[1:])), seas


def test_rescale_preserves_shape_parallelism(base):
    """Growing the pool changes only data parallelism."""
    bigger = rescale_pool(base, base.n_pool * 2)
    assert bigger is not None
    assert bigger.parallelism.tp == base.parallelism.tp
    assert bigger.parallelism.pp == base.parallelism.pp
    assert bigger.parallelism.cp == base.parallelism.cp
    assert bigger.parallelism.dp > base.parallelism.dp


def test_scaling_curve_starts_at_baseline(base):
    curve = scaling_curve(base, max_factor=4.0, points=12)
    assert curve
    assert curve[0][0] == pytest.approx(base.n_pool, rel=0.02)
    assert all(n2 >= n1 for (n1, _), (n2, _) in zip(curve, curve[1:]))


def test_strong_scaling_has_diminishing_returns(base):
    """Doubling the pool at fixed global batch must deliver less than double.

    This is the mechanism that makes the substitution metric meaningful: if
    scaling were perfect, every intervention's SEA would equal its naive value.
    """
    curve = dict(scaling_curve(base, max_factor=4.0, points=24))
    sizes = sorted(curve)
    base_n, base_v = sizes[0], curve[sizes[0]]
    doubled = [n for n in sizes if n >= 2 * base_n]
    assert doubled, "scaling curve did not reach twice the baseline pool"
    n2 = doubled[0]
    assert curve[n2] < 2.0 * base_v * (n2 / (2.0 * base_n))


def test_sea_differs_from_naive_metric(base):
    """The informal metric and the substitution metric must not coincide.

    If they agreed, the substitution definition would add nothing. The direction
    and size of the gap is a reported result, so this test only asserts that a
    gap exists for a representative intervention.
    """
    interv = Intervention("double_nic", {"accelerator.nic_bw_gbps": 800.0})
    result = substitution_equivalent_accelerators(base, interv)
    assert result["naive_equivalent"] > 0
    assert math.isfinite(result["sea"])
    rel_gap = abs(result["sea"] - result["naive_equivalent"]) / result["naive_equivalent"]
    assert rel_gap > 0.01, result


def test_naive_metric_sign_matches_loss_reduction(base):
    """The informal metric is positive exactly when blocked plus discarded falls."""
    treated = replace_nested(base, **{"accelerator.nic_bw_gbps": 800.0})
    a, b = build_ledger(base), build_ledger(treated)
    naive = naive_equivalent_accelerators(a, b)
    assert (naive > 0) == ((a.blocked + a.discarded) > (b.blocked + b.discarded))


def test_shapley_is_efficient(base):
    """Shapley values must sum to the bundle's value (the efficiency axiom)."""
    parts = [
        Intervention("bandwidth", {"accelerator.nic_bw_gbps": 800.0}),
        Intervention("reliability", {"reliability.failure_rate_per_node_day": 2.5e-3}),
        Intervention("recovery", {"reliability.restart_time_s": 60.0}),
    ]
    attribution = shapley_attribution(base, parts)
    total = sum(attribution[p.name] for p in parts)
    assert total == pytest.approx(attribution["bundle_sea"], rel=1e-6)


def test_bundle_merges_overrides(base):
    a = Intervention("a", {"accelerator.nic_bw_gbps": 800.0}, cost_accelerator_equivalents=100.0)
    b = Intervention(
        "b", {"reliability.restart_time_s": 60.0}, cost_accelerator_equivalents=50.0
    )
    bundle = bundle_intervention("ab", [a, b])
    assert bundle.overrides == {
        "accelerator.nic_bw_gbps": 800.0,
        "reliability.restart_time_s": 60.0,
    }
    assert bundle.cost_accelerator_equivalents == pytest.approx(150.0)


def test_config_roundtrip(base, tmp_path):
    path = tmp_path / "scenario.yaml"
    save_scenario(base, path)
    restored = _load(path)
    assert restored.parallelism == base.parallelism
    assert restored.model == base.model
    assert build_ledger(restored).useful_capacity_fraction == pytest.approx(
        build_ledger(base).useful_capacity_fraction
    )
