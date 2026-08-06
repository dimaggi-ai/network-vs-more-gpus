"""Behavioral invariants the model must satisfy regardless of parameter values.

These are the Gate 3 checks: accounting correctness and limiting-case behavior.
They are cheap and run on every commit.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from netcap import build_ledger, load_scenario
from netcap.config import replace_nested

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/scenarios/llama3_405b_16k.yaml"


@pytest.fixture(scope="module")
def base():
    return load_scenario(SCENARIO)


def test_time_accounting_closes(base):
    """The four fates must sum to accelerator-seconds paid for."""
    ledger = build_ledger(base)
    ledger.check_invariants(rel_tol=1e-9)
    total = ledger.productive + ledger.blocked + ledger.discarded + ledger.unavailable
    assert math.isclose(total, ledger.total, rel_tol=1e-9)


def test_no_negative_buckets(base):
    for oversub in (1.0, 4.0, 16.0):
        for rate in (0.0, 5e-3, 5e-2):
            cfg = replace_nested(
                base,
                **{
                    "topology.oversubscription": oversub,
                    "reliability.failure_rate_per_node_day": rate,
                },
            )
            build_ledger(cfg).check_invariants()


def test_infinite_bandwidth_removes_communication(base):
    """As link bandwidth grows without bound, exposed communication vanishes."""
    fast = replace_nested(
        base,
        **{
            "accelerator.nic_bw_gbps": 4.0e9,
            "accelerator.scaleup_bw_gbps": 4.0e10,
            "topology.alpha_scaleup_us": 0.0,
            "topology.alpha_pod_us": 0.0,
            "topology.alpha_xpod_us": 0.0,
        },
    )
    ledger = build_ledger(fast)
    assert ledger.blocked_comm / ledger.total < 1e-6


def test_zero_failure_rate_removes_loss(base):
    """With no failures there is no discarded work and no restart time."""
    perfect = replace_nested(base, **{"reliability.failure_rate_per_node_day": 0.0})
    ledger = build_ledger(perfect)
    assert ledger.discarded == pytest.approx(0.0, abs=1e-6)
    assert ledger.blocked_restart == pytest.approx(0.0, abs=1e-6)


def test_ideal_cluster_reaches_unit_capacity(base):
    """No communication cost, no failures, no overhead: every second is productive."""
    ideal = replace_nested(
        base,
        **{
            "accelerator.nic_bw_gbps": 4.0e9,
            "accelerator.scaleup_bw_gbps": 4.0e10,
            "topology.alpha_scaleup_us": 0.0,
            "topology.alpha_pod_us": 0.0,
            "topology.alpha_xpod_us": 0.0,
            "reliability.failure_rate_per_node_day": 0.0,
            "reliability.checkpoint_write_s": 0.0,
            "reliability.spare_fraction": 0.0,
            "reliability.stranded_fraction": 0.0,
            "parallelism.pp": 1,
            "parallelism.interleave_factor": 1,
        },
    )
    ideal = dataclasses.replace(ideal, n_pool=ideal.parallelism.world_size)
    ledger = build_ledger(ideal)
    assert ledger.useful_capacity_fraction > 0.999


def test_capacity_monotone_in_bandwidth(base):
    """More scale-out bandwidth never reduces useful capacity."""
    values = []
    for bw in (100.0, 200.0, 400.0, 800.0, 1600.0):
        cfg = replace_nested(base, **{"accelerator.nic_bw_gbps": bw})
        values.append(build_ledger(cfg).useful_capacity_fraction)
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), values


def test_capacity_monotone_in_failure_rate(base):
    """A higher failure rate never increases useful capacity."""
    values = []
    for rate in (0.0, 1e-3, 5e-3, 2e-2, 1e-1):
        cfg = replace_nested(base, **{"reliability.failure_rate_per_node_day": rate})
        values.append(build_ledger(cfg).useful_capacity_fraction)
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:])), values


def test_capacity_monotone_in_oversubscription(base):
    """More cross-pod oversubscription never increases useful capacity."""
    values = []
    for sigma in (1.0, 2.0, 4.0, 8.0, 16.0):
        cfg = replace_nested(base, **{"topology.oversubscription": sigma})
        values.append(build_ledger(cfg).useful_capacity_fraction)
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:])), values


def test_ettr_at_least_ucf_for_job_sized_pool(base):
    """ETTR is a looser metric than UCF and must not be smaller on the same job.

    ETTR credits exposed communication as productive runtime; UCF does not. With
    a pool equal to the job size the two share a denominator, so ETTR dominates.
    """
    cfg = replace_nested(base, **{"reliability.spare_fraction": 0.0})
    cfg = dataclasses.replace(cfg, n_pool=cfg.parallelism.world_size)
    ledger = build_ledger(cfg)
    assert ledger.effective_training_time_ratio >= ledger.useful_capacity_fraction
