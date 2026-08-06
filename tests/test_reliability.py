"""Reliability model checks against independently derived closed forms.

Three external reference points are reproduced:

* Daly's optimal checkpoint interval, recovered by numerical optimization of the
  model rather than by evaluating Daly's formula.
* The expected-ETTR expression published by Meta (HPCA 2025).
* Agreement between the analytical renewal path and the event-driven Monte
  Carlo path, which share no implementation.
"""

from __future__ import annotations

import math

import pytest

from netcap.config import ReliabilitySpec, SECONDS_PER_DAY
from netcap.reliability import (
    analytical_reliability,
    daly_optimal_interval,
    expected_lost_work,
    meta_ettr_closed_form,
    monte_carlo_reliability,
)

WINDOW = 30 * SECONDS_PER_DAY


def _spec(**kw) -> ReliabilitySpec:
    base = dict(
        failure_rate_per_node_day=5e-3,
        detect_time_s=60.0,
        restart_time_s=120.0,
        checkpoint_write_s=30.0,
        checkpoint_blocking_fraction=1.0,
        repair_time_s=1800.0,
        spare_fraction=0.05,
    )
    base.update(kw)
    return ReliabilitySpec(**base)


def test_expected_lost_work_limits():
    """Short intervals lose half a period; long intervals approach the MTTF."""
    assert expected_lost_work(1e-6, 100.0) == pytest.approx(50.0, rel=1e-3)
    rate = 1e-3
    assert expected_lost_work(rate, 1e9) == pytest.approx(1.0 / rate, rel=1e-3)
    assert expected_lost_work(0.0, 100.0) == 0.0


def test_daly_interval_matches_numerical_optimum():
    """The interval maximizing modeled productive time matches Daly's formula.

    Daly's expression is derived independently of this model, so recovering it by
    brute-force search over the model's own output is a genuine check.
    """
    n_acc, per_node = 8192, 8
    for write_s in (10.0, 30.0, 120.0):
        spec_probe = _spec(checkpoint_write_s=write_s, checkpoint_interval_s=1.0)
        mttf = analytical_reliability(spec_probe, n_acc, per_node, WINDOW).job_mttf_s
        closed = daly_optimal_interval(write_s, mttf)

        best_interval, best_running = None, -1.0
        lo, hi = max(30.0, closed / 8), closed * 8
        n = 4000
        for i in range(n + 1):
            interval = lo + (hi - lo) * i / n
            spec = _spec(checkpoint_write_s=write_s, checkpoint_interval_s=interval)
            timing = analytical_reliability(spec, n_acc, per_node, WINDOW)
            if timing.running_s > best_running:
                best_running, best_interval = timing.running_s, interval

        assert best_interval == pytest.approx(closed, rel=0.10), (
            f"write={write_s}: numerical optimum {best_interval:.1f}s vs Daly {closed:.1f}s"
        )


def test_meta_closed_form_agreement():
    """Model ETTR tracks the Meta HPCA 2025 expression in its regime of validity.

    The published expression is a small-loss approximation: it ignores detection
    time and treats losses as additive. Agreement is therefore checked where the
    total loss is small, and the tolerance reflects that.
    """
    per_node = 8
    for n_acc in (2048, 8192, 16384):
        n_nodes = n_acc // per_node
        interval = 1800.0
        write_s = 20.0
        restart = 120.0
        spec = _spec(
            detect_time_s=0.0,
            restart_time_s=restart,
            checkpoint_interval_s=interval,
            checkpoint_write_s=write_s,
            spare_fraction=0.10,
            repair_time_s=600.0,
        )
        timing = analytical_reliability(spec, n_acc, per_node, WINDOW)
        model_ettr = timing.running_s / (WINDOW - timing.unavailable_s)
        published = meta_ettr_closed_form(n_nodes, 5e-3, restart, interval, write_s)
        assert model_ettr == pytest.approx(published, rel=0.05), (
            f"n={n_acc}: model {model_ettr:.4f} vs published form {published:.4f}"
        )


@pytest.mark.parametrize("n_acc", [1024, 8192, 32768])
def test_analytical_matches_monte_carlo(n_acc):
    """Two independent implementations of the same timeline must agree.

    The Monte Carlo path walks wall-clock events; the analytical path uses
    renewal expectations. They share only the parameter dataclass.
    """
    per_node = 8
    spec = _spec()
    a = analytical_reliability(spec, n_acc, per_node, WINDOW)
    m = monte_carlo_reliability(spec, n_acc, per_node, WINDOW, seed=17, n_replicates=400)

    assert m.running_s == pytest.approx(a.running_s, rel=0.05)
    assert m.discarded_s == pytest.approx(a.discarded_s, rel=0.20)
    assert m.n_failures == pytest.approx(a.n_failures, rel=0.15)


@pytest.mark.parametrize("n_acc", [2048, 8192, 16384, 65536])
@pytest.mark.parametrize("rate", [1e-3, 5e-3, 1.5e-2, 3e-2])
def test_fidelity_agreement_everywhere(n_acc, rate):
    """The two implementations must agree at every tested severity.

    The validity envelope is a modeling-scope guard, not an accuracy boundary:
    the fast path tracks the event-driven path to well under one percent even at
    recovery pressures far beyond the envelope. An earlier version of the
    event-driven path mis-attributed detection time and appeared to diverge at
    high severity; see DECISIONS.md D14. This test asserts agreement at every
    cell, including the out-of-envelope ones, so a regression in either path
    fails loudly.
    """
    per_node = 8
    spec = _spec(failure_rate_per_node_day=rate)
    a = analytical_reliability(spec, n_acc, per_node, WINDOW)
    m = monte_carlo_reliability(spec, n_acc, per_node, WINDOW, seed=17, n_replicates=300)
    rel_error = abs(m.running_s - a.running_s) / a.running_s
    assert rel_error < 0.02, (
        f"n={n_acc} rate={rate}: {rel_error:.4f} drift at recovery pressure "
        f"{a.recovery_pressure:.3f}"
    )


def test_monte_carlo_conserves_wall_clock():
    """Every wall-clock second must be booked exactly once, even at severities
    where recovery dominates. The Monte Carlo path asserts this internally and
    raises rather than rescaling; this test exercises that assertion at the
    harshest configuration used anywhere in the study."""
    spec = _spec(failure_rate_per_node_day=3e-2)
    m = monte_carlo_reliability(spec, 65536, 8, WINDOW, seed=3, n_replicates=50)
    booked = m.discarded_s + m.restart_blocked_s + m.checkpoint_blocked_s + m.running_s
    window_running = WINDOW - m.unavailable_s
    assert booked == pytest.approx(window_running, rel=1e-6)


def test_failure_free_cluster_is_fully_running():
    spec = _spec(failure_rate_per_node_day=0.0, checkpoint_write_s=0.0, spare_fraction=0.0)
    timing = analytical_reliability(spec, 1024, 8, WINDOW)
    assert timing.discarded_s == pytest.approx(0.0)
    assert timing.restart_blocked_s == pytest.approx(0.0)
    assert timing.running_s == pytest.approx(WINDOW, rel=1e-9)


def test_mttf_scales_inversely_with_node_count():
    """Job MTTF halves when the node count doubles, the standard series result."""
    spec = _spec()
    a = analytical_reliability(spec, 4096, 8, WINDOW).job_mttf_s
    b = analytical_reliability(spec, 8192, 8, WINDOW).job_mttf_s
    assert b == pytest.approx(a / 2.0, rel=1e-9)


@pytest.mark.parametrize("n_acc,reported_hours", [(16384, 1.8), (131072, 0.23)])
def test_meta_projected_mttf_reproduced(n_acc, reported_hours):
    """Reproduce Meta's large-scale job MTTF using RSC-1's measured node rate.

    Meta reports 1.8 h at 16,384 GPUs and 0.23 h at 131,072 GPUs alongside a
    measured RSC-1 rate of 6.50 failures per thousand node-days. Feeding that
    rate into the model reproduces both figures within a few percent.
    """
    spec = _spec(failure_rate_per_node_day=6.5e-3)
    mttf_hours = analytical_reliability(spec, n_acc, 8, WINDOW).job_mttf_s / 3600.0
    assert mttf_hours == pytest.approx(reported_hours, rel=0.05)


def test_small_job_mttf_divergence_is_documented():
    """Record the known divergence at small job sizes rather than hiding it.

    Meta's *measured* 1024-GPU MTTF of 7.9 h is far shorter than inverse scaling
    from their per-node hardware rate predicts (about 29 h). Their own reported
    figures are not mutually consistent under inverse scaling: 1024 to 16,384
    GPUs is a 16x change in size but only a 4.4x change in reported MTTF.

    The implication, carried into the limitations section, is that at smaller job
    sizes job-stopping events are dominated by causes this model does not
    represent as per-node hardware failures. The model is therefore calibrated
    and validated for large jobs, and will be optimistic for small ones.
    """
    spec = _spec(failure_rate_per_node_day=6.5e-3)
    predicted = analytical_reliability(spec, 1024, 8, WINDOW).job_mttf_s / 3600.0
    measured = 7.9
    assert predicted / measured > 3.0, (
        "the divergence at 1024 GPUs has changed; revisit the limitations section"
    )
