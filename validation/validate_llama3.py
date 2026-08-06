"""Validation against the published Llama 3 405B pre-training measurements.

Design: one free compute-side parameter (``kernel_efficiency``) is calibrated on
a single anchor, then held fixed while the model predicts a different, held-out
configuration. Nothing about the network or reliability model is tuned.

Anchors, all from arXiv:2407.21783 Table 4 and Section 3.3 (``SOURCES.md`` S3):

  Row 1  8,192 GPUs   TP8 CP1 PP16 DP64   16M tokens/batch   430 TFLOP/s/GPU  43% MFU
  Row 2 16,384 GPUs   TP8 CP1 PP16 DP128  16M tokens/batch   400 TFLOP/s/GPU  41% MFU

Row 1 is the calibration anchor. Row 2 is held out: it doubles data parallelism
at fixed global batch, which halves the compute per step while leaving the
gradient all-reduce to span twice as many pods. If the communication and
parallelism model is wrong, the held-out prediction misses.

Row 3 of the published table (CP=16, sequence length 131,072) is excluded from
the quantitative comparison because it is internally inconsistent as printed:
the stated degrees TP=8, CP=16, PP=16, DP=4 multiply to 8,192, not the 16,384
GPUs stated in the same row. The discrepancy is reported, not silently resolved.

Acceptance thresholds are predefined here, before the comparison runs:

  T1  held-out throughput error                 <= 10 percent relative
  T2  predicted interruption count error        <= 35 percent relative
  T3  predicted job-level ETTR                  >= 0.85 given a published ">90%"
  T4  calibrated kernel efficiency plausibility in [0.50, 0.85]

T2 is deliberately loose: the failure rate is taken from a different Meta
cluster (HPCA 2025) than the one Llama 3 trained on, so an exact match would be
surprising rather than reassuring.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netcap import build_ledger, load_scenario, mfu  # noqa: E402
from netcap.config import ScenarioConfig  # noqa: E402
from netcap.performance import model_flops_per_token, step_breakdown  # noqa: E402

PUBLISHED = {
    "row1_8k": {"gpus": 8192, "tflops_per_gpu": 430.0, "mfu": 0.43},
    "row2_16k": {"gpus": 16384, "tflops_per_gpu": 400.0, "mfu": 0.41},
}
PUBLISHED_INTERRUPTIONS = 419
PUBLISHED_ETTR_FLOOR = 0.90
WINDOW_DAYS = 54.0

THRESHOLDS = {
    "T1_heldout_throughput_rel_error": 0.10,
    "T2_interruption_rel_error": 0.35,
    "T3_ettr_floor": 0.85,
    "T4_kernel_efficiency_range": (0.50, 0.85),
}


def achieved_tflops_per_gpu(scenario: ScenarioConfig) -> float:
    """Nominal model FLOPs per second per accelerator, the quantity Table 4 reports."""
    step = step_breakdown(scenario)
    nominal = model_flops_per_token(scenario.model) * step.tokens_per_step
    return nominal / (step.t_step * scenario.parallelism.world_size) / 1e12


def calibrate_kernel_efficiency(scenario: ScenarioConfig, target_tflops: float) -> float:
    """Solve for the kernel efficiency that reproduces one anchor exactly.

    Throughput is linear in kernel efficiency at fixed communication time only in
    the limit of zero communication, so this is solved by bisection rather than
    by scaling.
    """
    lo, hi = 0.05, 0.99
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        trial = replace(scenario, kernel=replace(scenario.kernel, kernel_efficiency=mid))
        if achieved_tflops_per_gpu(trial) < target_tflops:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    calib = load_scenario(ROOT / "configs/scenarios/llama3_405b_8k.yaml")
    heldout = load_scenario(ROOT / "configs/scenarios/llama3_405b_16k.yaml")

    eta = calibrate_kernel_efficiency(calib, PUBLISHED["row1_8k"]["tflops_per_gpu"])
    calib = replace(calib, kernel=replace(calib.kernel, kernel_efficiency=eta))
    heldout = replace(heldout, kernel=replace(heldout.kernel, kernel_efficiency=eta))

    pred_calib = achieved_tflops_per_gpu(calib)
    pred_heldout = achieved_tflops_per_gpu(heldout)
    obs_heldout = PUBLISHED["row2_16k"]["tflops_per_gpu"]
    err_heldout = abs(pred_heldout - obs_heldout) / obs_heldout

    ledger = build_ledger(heldout)
    pred_interruptions = ledger.timing.n_failures
    err_interruptions = abs(pred_interruptions - PUBLISHED_INTERRUPTIONS) / PUBLISHED_INTERRUPTIONS
    pred_ettr = ledger.effective_training_time_ratio

    lo, hi = THRESHOLDS["T4_kernel_efficiency_range"]
    results: Dict[str, object] = {
        "calibrated_kernel_efficiency": eta,
        "calibration_anchor_tflops_pred": pred_calib,
        "calibration_anchor_tflops_obs": PUBLISHED["row1_8k"]["tflops_per_gpu"],
        "heldout_tflops_pred": pred_heldout,
        "heldout_tflops_obs": obs_heldout,
        "heldout_rel_error": err_heldout,
        "heldout_mfu_pred": mfu(heldout),
        "heldout_mfu_obs": PUBLISHED["row2_16k"]["mfu"],
        "predicted_interruptions_54d": pred_interruptions,
        "observed_interruptions_54d": PUBLISHED_INTERRUPTIONS,
        "interruption_rel_error": err_interruptions,
        "predicted_ettr": pred_ettr,
        "published_ettr_floor": PUBLISHED_ETTR_FLOOR,
        "predicted_ucf_pool_level": ledger.useful_capacity_fraction,
        "job_mttf_hours": ledger.timing.job_mttf_s / 3600.0,
        "implied_failure_rate_per_node_day_from_llama3": (
            PUBLISHED_INTERRUPTIONS
            / (WINDOW_DAYS * (PUBLISHED["row2_16k"]["gpus"] / 8))
        ),
        "checks": {
            "T1_heldout_throughput": bool(err_heldout <= THRESHOLDS["T1_heldout_throughput_rel_error"]),
            "T2_interruptions": bool(err_interruptions <= THRESHOLDS["T2_interruption_rel_error"]),
            "T3_ettr": bool(pred_ettr >= THRESHOLDS["T3_ettr_floor"]),
            "T4_kernel_efficiency_plausible": bool(lo <= eta <= hi),
        },
        "thresholds": {k: v for k, v in THRESHOLDS.items()},
    }
    results["all_checks_pass"] = all(results["checks"].values())  # type: ignore[index]

    out = ROOT / "results/validation/llama3_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    return 0 if results["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
