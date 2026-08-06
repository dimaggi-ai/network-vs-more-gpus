"""Every number quoted in the paper, article, and README, checked against results.

This is the claim-to-evidence audit as an executable test. If a model change
moves a published number, this fails rather than letting the prose drift away
from the data. Requires ``make experiments && make summary`` to have been run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/processed/summary.json"


@pytest.fixture(scope="module")
def s():
    if not SUMMARY.exists():
        pytest.skip("run 'make experiments && make summary' first")
    return json.loads(SUMMARY.read_text())


def test_capacity_gap(s):
    """Headline: availability overstates capacity by 26.1 points at 16K."""
    assert s["capacity_gap_16k"]["ucf"] == pytest.approx(0.627, abs=5e-4)
    assert s["capacity_gap_16k"]["ettr"] == pytest.approx(0.889, abs=1e-3)
    assert s["capacity_gap_16k"]["gap_percentage_points"] == pytest.approx(26.1, abs=0.05)


def test_capacity_collapses_with_scale(s):
    assert s["capacity_by_scale"]["1024"] == pytest.approx(0.781, abs=5e-4)
    assert s["capacity_by_scale"]["65536"] == pytest.approx(0.382, abs=5e-4)


def test_rank_reversals_exist(s):
    """The decision layer's justification: no single intervention dominates."""
    assert len(s["decision_map"]["distinct_winners"]) == 3
    assert s["decision_map"]["cells_inside_envelope"] == 96
    assert "bandwidth_4x" not in s["decision_map"]["winners_by_scale"]["16384"]
    assert "bandwidth_4x" in s["decision_map"]["winners_by_scale"]["65536"]
    assert s["decision_map"]["winners_by_scale"]["65536"]["reliability_2x"] == 28


def test_informal_metric_bias(s):
    f = s["naive_metric_understatement_factor"]
    assert f["2048"] == pytest.approx(1.36, abs=0.02)
    assert f["65536"] == pytest.approx(4.62, abs=0.05)
    assert f["2048"] < f["65536"], "bias must grow with scale"


def test_recovery_interventions_are_complements(s):
    """Refutes hypothesis H3's stated direction; reported as a null result."""
    by = s["additivity"]["by_bundle"]
    assert by["recovery_trio"]["mean_error"] == pytest.approx(-0.089, abs=0.002)
    assert by["recovery_trio"]["max"] < 0, "recovery bundle must be superadditive"
    assert by["network_pair"]["mean_error"] == pytest.approx(0.065, abs=0.002)


def test_rank_stability(s):
    share = s["rank_stability"]["share_ranked_first_pct"]
    assert share["fast_checkpoint"] == pytest.approx(45.8, abs=0.1)
    assert share["reliability_2x"] == pytest.approx(29.1, abs=0.1)
    assert share["bandwidth_4x"] == pytest.approx(0.6, abs=0.1)
    assert s["rank_stability"]["draws_inside_envelope"] == 323


def test_cross_fidelity_envelope(s):
    cf = s["cross_fidelity"]
    # Agreement holds at every tested severity, outside the scope guard included;
    # the guard is a modeling-scope judgment, not a fidelity boundary (D14).
    assert cf["max_rel_error_ucf_inside"] < 0.005
    assert cf["max_rel_error_ucf_outside"] < 0.005


def test_external_validation(s):
    v = s["validation"]
    assert v["heldout_tflops_pred"] == pytest.approx(406.3, abs=0.1)
    assert v["heldout_rel_error"] == pytest.approx(0.0157, abs=5e-4)
    assert v["heldout_mfu_pred"] == pytest.approx(0.4106, abs=1e-3)
    assert v["predicted_interruptions_54d"] == pytest.approx(526.6, abs=0.5)
    assert v["predicted_ettr"] == pytest.approx(0.8885, abs=1e-3)
    assert v["all_checks_pass"] is True


def test_counterexamples(s):
    """The cases where this project's own thesis fails."""
    c = s["counterexamples"]
    # On an already fast, flat fabric a further 4x is worth under 0.3 percent of
    # the pool while halving the failure rate is worth 5.7 percent of it.
    assert c["perfect_network"]["max_network_sea"] == pytest.approx(46.7, abs=1.0)
    assert c["perfect_network"]["max_reliability_sea"] == pytest.approx(954.3, abs=1.0)
    assert c["huge_job"]["any_unattainable_by_scaling"] is True
    assert c["huge_job"]["within_validity_envelope"] is False
    assert c["tiny_job"]["baseline_ucf"] == pytest.approx(0.781, abs=5e-4)
