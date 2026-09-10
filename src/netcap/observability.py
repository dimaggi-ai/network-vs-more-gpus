"""Compare normalized offline collective benchmarks, without claiming causality.

Inputs are explicit JSON exports, not arbitrary NCCL debug logs. Bandwidth is
decimal GB/s; NCCL bus bandwidth is not physical link Gbps. Results are evidence
for investigation, never an automatic net_efficiency calibration or GPU gain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import sys

VERSION = "netcap-collective-observation/v1"
CONTEXT_FIELDS = {"collective", "dtype", "ranks", "hardware_id", "topology_id",
                  "transport", "library_version", "protocol_id"}
SAMPLE_FIELDS = {"size_bytes", "mode", "algbw_GBps", "busbw_GBps", "wrong_elements"}


def finite(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")


def validate_observation(record):
    required = {"schema_version", "evidence_kind", "source", "context", "samples"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError(f"observation must contain exactly {sorted(required)}")
    if record["schema_version"] != VERSION:
        raise ValueError("unsupported collective observation schema")
    if record["evidence_kind"] not in ("synthetic", "benchmark"):
        raise ValueError("evidence_kind must be synthetic or benchmark")
    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError("source must identify the input artifact")
    ctx = record["context"]
    if not isinstance(ctx, dict) or set(ctx) != CONTEXT_FIELDS:
        raise ValueError(f"context must contain exactly {sorted(CONTEXT_FIELDS)}")
    for key in CONTEXT_FIELDS - {"ranks"}:
        if not isinstance(ctx[key], str) or not ctx[key].strip():
            raise ValueError(f"context.{key} must be nonempty text")
    if type(ctx["ranks"]) is not int or ctx["ranks"] < 2:
        raise ValueError("ranks must be an integer >= 2")
    if not isinstance(record["samples"], list) or not record["samples"]:
        raise ValueError("samples must be nonempty")
    for sample in record["samples"]:
        if not isinstance(sample, dict) or set(sample) != SAMPLE_FIELDS:
            raise ValueError(f"each sample must contain exactly {sorted(SAMPLE_FIELDS)}")
        for key in ("size_bytes", "wrong_elements"):
            if type(sample[key]) is not int or sample[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if sample["size_bytes"] == 0:
            raise ValueError("size_bytes must be positive")
        if sample["mode"] not in ("in-place", "out-of-place"):
            raise ValueError("mode must be in-place or out-of-place")
        finite(sample["algbw_GBps"], "algbw_GBps")
        finite(sample["busbw_GBps"], "busbw_GBps")
    return record


def _groups(record):
    groups = {}
    for sample in record["samples"]:
        groups.setdefault((sample["size_bytes"], sample["mode"]), []).append(sample)
    return groups


def compare_collectives(baseline, candidate, *, drop_fraction=0.15):
    """Matched-profile median regression screen, not statistical significance.

    protocol_id must identify comparable warmup, iterations, affinity, software
    settings and load conditions. Declaring that ID does not verify those facts.
    A context mismatch or different message/mode grid is refused, not averaged.
    """
    validate_observation(baseline)
    validate_observation(candidate)
    finite(drop_fraction, "drop_fraction")
    if not 0 < drop_fraction < 1:
        raise ValueError("drop_fraction must be in (0, 1)")
    if baseline["context"] != candidate["context"]:
        raise ValueError("incomparable benchmark contexts")
    if baseline["evidence_kind"] != candidate["evidence_kind"]:
        raise ValueError("cannot compare synthetic and benchmark evidence")
    left, right = _groups(baseline), _groups(candidate)
    if left.keys() != right.keys():
        raise ValueError("incomparable message-size/mode grids")
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        baseline_bw = median(s["algbw_GBps"] for s in a)
        candidate_bw = median(s["algbw_GBps"] for s in b)
        finite(baseline_bw, "baseline median bandwidth")
        finite(candidate_bw, "candidate median bandwidth")
        baseline_bus = median(s["busbw_GBps"] for s in a)
        candidate_bus = median(s["busbw_GBps"] for s in b)
        finite(baseline_bus, "baseline median bus bandwidth")
        finite(candidate_bus, "candidate median bus bandwidth")
        wrong = sum(s["wrong_elements"] for s in a + b)
        drop = None if baseline_bw == 0 else 1 - candidate_bw / baseline_bw
        if drop is not None and not math.isfinite(drop):
            raise ValueError("relative bandwidth change exceeds finite numeric range")
        status = ("invalid-correctness" if wrong else "insufficient-baseline" if drop is None
                  else "regression" if drop >= drop_fraction else "no-regression-detected")
        rows.append({"size_bytes": key[0], "mode": key[1], "status": status,
                     "baseline_samples": len(a), "candidate_samples": len(b),
                     "baseline_median_algbw_GBps": baseline_bw,
                     "candidate_median_algbw_GBps": candidate_bw,
                     "baseline_median_busbw_GBps": baseline_bus,
                     "candidate_median_busbw_GBps": candidate_bus,
                     "relative_algbw_drop": drop, "wrong_elements": wrong})
    return {"schema_version": "netcap-collective-comparison/v1",
            "evidence_kind": baseline["evidence_kind"], "context": baseline["context"],
            "baseline_source": baseline["source"], "candidate_source": candidate["source"],
            "drop_fraction": drop_fraction, "comparisons": rows,
            "cause": "undetermined", "automatic_model_calibration": False,
            "limitations": ["Offline normalized observations; no live NCCL/DCGM/InfiniBand collector.",
                            "Threshold screen, not a confidence interval or causal fault diagnosis.",
                            "No conversion from bus bandwidth to physical link rate or stranded GPU count.",
                            "Context equality trusts supplied metadata; independently verify measurement conditions."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--drop-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    try:
        result = compare_collectives(json.loads(args.baseline.read_text()),
                                     json.loads(args.candidate.read_text()),
                                     drop_fraction=args.drop_fraction)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError) as exc:
        print(f"netcap observations: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
