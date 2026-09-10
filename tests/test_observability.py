import pytest
from netcap.observability import compare_collectives, VERSION


def observation(bw=100):
    return {"schema_version": VERSION, "evidence_kind": "synthetic", "source": "unit fixture",
            "context": {"collective": "all-reduce", "dtype": "float", "ranks": 8,
                        "hardware_id": "abstract", "topology_id": "one-pod",
                        "transport": "fixture", "library_version": "fixture-v1",
                        "protocol_id": "same-warmup-iterations-load"},
            "samples": [{"size_bytes": 1048576, "mode": "out-of-place", "algbw_GBps": bw,
                         "busbw_GBps": bw * 1.75, "wrong_elements": 0}]}


def test_known_relative_drop_without_causality():
    result = compare_collectives(observation(100), observation(75))
    row = result["comparisons"][0]
    assert row["relative_algbw_drop"] == 0.25
    assert row["status"] == "regression"
    assert result["cause"] == "undetermined"
    assert result["evidence_kind"] == "synthetic"
    assert not result["automatic_model_calibration"]


def test_identity_and_improvement():
    for bw in (100, 120):
        assert compare_collectives(observation(), observation(bw))["comparisons"][0]["status"] == "no-regression-detected"


def test_correctness_errors_are_not_performance_evidence():
    candidate = observation(75)
    candidate["samples"][0]["wrong_elements"] = 1
    assert compare_collectives(observation(), candidate)["comparisons"][0]["status"] == "invalid-correctness"


def test_zero_baseline_is_not_infinite_gain():
    assert compare_collectives(observation(0), observation())["comparisons"][0]["status"] == "insufficient-baseline"


def test_repeats_use_median():
    candidate = observation(75)
    candidate["samples"] += observation(80)["samples"] + observation(1000)["samples"]
    assert compare_collectives(observation(), candidate)["comparisons"][0]["candidate_median_algbw_GBps"] == 80


@pytest.mark.parametrize("field,value", [("ranks", 16), ("topology_id", "two-pods"),
                                      ("dtype", "half"), ("protocol_id", "different"),
                                      ("hardware_id", "different"), ("library_version", "different")])
def test_context_mismatch_refused(field, value):
    candidate = observation()
    candidate["context"][field] = value
    with pytest.raises(ValueError, match="context"):
        compare_collectives(observation(), candidate)


@pytest.mark.parametrize("field,value", [("algbw_GBps", float("nan")), ("busbw_GBps", -1),
                                      ("algbw_GBps", True), ("wrong_elements", -1),
                                      ("mode", "unknown"), ("size_bytes", 0)])
def test_malformed_samples_refused(field, value):
    bad = observation()
    bad["samples"][0][field] = value
    with pytest.raises(ValueError):
        compare_collectives(observation(), bad)


def test_no_mixed_evidence_or_message_grid():
    for field, value in (("evidence_kind", "benchmark"), ("samples", observation()["samples"] + [observation()["samples"][0] | {"size_bytes": 4096}])):
        candidate = observation()
        candidate[field] = value
        with pytest.raises(ValueError):
            compare_collectives(observation(), candidate)


@pytest.mark.parametrize("threshold", [0, 1, -1, float("nan"), True])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        compare_collectives(observation(), observation(), drop_fraction=threshold)
