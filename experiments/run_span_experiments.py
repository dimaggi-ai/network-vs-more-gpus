"""Span-tier experiment runner: the latency-regime atlas.

Follows the conventions of ``run_experiments.py`` exactly --- raw CSVs into
``results/raw/``, an immutable sidecar ``.meta.json`` per experiment recording
version, commit, seed, and scenario digest --- so a span figure traces back to
its run the same way every other figure in this repository does.

The study baseline is ``reference_405b_16k``, not the Llama 3 validation
scenario. That split is the repository's existing discipline (DECISIONS.md D8):
the validation scenario stays untuned so it can be checked against published
targets, and the study baseline carries the calibrated kernel efficiency and the
straggler term that make sweeps realistic. Numbers quoted from
``validation/validate_span.py`` are computed on the validation scenario and will
not match these to the last digit; that is intended, and each document says
which scenario it is quoting.

Run ``python experiments/run_span_experiments.py --list`` to see the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import netcap  # noqa: E402
from netcap import build_ledger, load_scenario  # noqa: E402
from netcap.config import ScenarioConfig, replace_nested  # noqa: E402
from netcap.regimes import (  # noqa: E402
    CUTS,
    REGIMES,
    _local_baseline,
    _spanned,
    atlas,
    bytes_across_boundary,
    circuit_lower_bound_s,
    exposed_bytes_across_boundary,
    placement_comparison,
    stitch_equivalent_accelerators,
    width_sweep,
)

RAW = ROOT / "results/raw"
BASELINE = ROOT / "configs/scenarios/reference_405b_16k.yaml"
SEED = 20260901

#: Widths this repository reports retention at. The ceiling is 8x400G, the
#: widest metro DCI class in deployment; above 12.8 Tbit/s the tier model
#: develops an artifact recorded in validate_span.py's DECLINED list.
REPORTED_GBPS = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
#: Extended ladder for the diminishing-returns sweep only.
SWEEP_GBPS = REPORTED_GBPS + (6400.0, 12800.0, 25600.0, 51200.0, 102400.0, 204800.0)
PROBE_GBPS = 204800.0
PLACEMENTS = ("hierarchical", "blind")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def scenario_digest(scenario: ScenarioConfig) -> str:
    payload = json.dumps(asdict(scenario), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write(name: str, frame: pd.DataFrame, scenario: ScenarioConfig, notes: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.csv"
    frame.to_csv(path, index=False)
    meta = {
        "experiment": name,
        "netcap_version": netcap.__version__,
        "git_commit": git_commit(),
        "seed": SEED,
        "baseline_scenario": scenario.name,
        "baseline_digest": scenario_digest(scenario),
        "rows": len(frame),
        "notes": notes,
    }
    (RAW / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(frame)} rows)")


def exp_s1_atlas(base: ScenarioConfig) -> None:
    """The regime atlas: retention by cut, distance, circuit width, and placement."""
    rows: List[Dict] = []
    for width in REPORTED_GBPS:
        for placement in PLACEMENTS:
            for cell in atlas(base, width, placement=placement):
                rows.append({
                    "circuit_gbps": width,
                    "placement": placement,
                    "regime": cell.regime,
                    "alpha_us": cell.alpha_us,
                    "cut": cell.cut,
                    "applicable": cell.applicable,
                    "retention": cell.retention,
                    "step_ratio": cell.step_ratio,
                    "reason": cell.reason,
                })
    write("s1_regime_atlas", pd.DataFrame(rows), base,
          "Useful-capacity retention for every (width, placement, regime, cut). "
          "Retention is spanned UCF over hall-local UCF with the plan, the world "
          "size, and every reliability parameter held fixed.")


def exp_s2_placement(base: ScenarioConfig) -> None:
    """Blind versus hierarchical rank placement across the distance regimes."""
    rows: List[Dict] = []
    for width in (400.0, 3200.0, PROBE_GBPS):
        for cut in ("pp", "dp"):
            for regime_name, blind, hier in placement_comparison(base, cut, REGIMES, width):
                rows.append({
                    "circuit_gbps": width, "cut": cut, "regime": regime_name,
                    "blind_step_ratio": blind, "hierarchical_step_ratio": hier,
                    "cliff_removed_x": (blind - 1.0) / (hier - 1.0) if hier > 1.0 else float("inf"),
                })
    write("s2_placement", pd.DataFrame(rows), base,
          "The headline. Same fiber, same circuit, same job; only the rank order "
          "changes. The distance cliff is a property of the ring construction.")


def exp_s3_ring_scaling(base: ScenarioConfig) -> None:
    """Blind-placement penalty against the size of the ring crossing the stitch."""
    rows: List[Dict] = []
    b_step = build_ledger(_local_baseline(base)).step.t_step
    for dp, tp, pp in ((128, 8, 16), (256, 8, 8), (512, 8, 4), (1024, 8, 2)):
        plan = replace_nested(base, **{
            "parallelism.tp": tp, "parallelism.pp": pp, "parallelism.dp": dp,
            "parallelism.global_batch_seqs": max(2048, dp * 2)})
        local = build_ledger(_local_baseline(plan)).step.t_step
        for placement in PLACEMENTS:
            spanned = _spanned(_local_baseline(plan), "dp", REGIMES[-1].alpha_us, PROBE_GBPS)
            spanned = replace_nested(spanned, **{"topology.span_placement": placement})
            rows.append({
                "dp": dp, "tp": tp, "pp": pp, "placement": placement,
                "crossing_ring": dp if placement == "blind" else 2,
                "step_ratio": build_ledger(spanned).step.t_step / local,
            })
    write("s3_ring_scaling", pd.DataFrame(rows), base,
          "A blind ring pays the stitch latency 2(k-1) times, so its penalty tracks "
          "the ring it puts across the boundary rather than the distance alone.")


def exp_s4_cluster_growth(base: ScenarioConfig) -> None:
    """Retention on a fixed metro circuit as the cluster it serves grows."""
    rows: List[Dict] = []
    for dp in (32, 64, 128, 256, 512):
        plan = replace_nested(base, **{"parallelism.dp": dp})
        plan = replace(plan, n_pool=plan.parallelism.world_size)
        for cut in ("pp", "dp"):
            local = build_ledger(_local_baseline(plan))
            spanned = _spanned(_local_baseline(plan), cut, 600.0, 800.0)
            if spanned is None:
                continue
            led = build_ledger(spanned)
            rows.append({
                "dp": dp, "world_size": plan.parallelism.world_size, "cut": cut,
                "circuit_gbps": 800.0, "alpha_us": 600.0,
                "retention": led.useful_capacity_fraction / local.useful_capacity_fraction,
                "step_ratio": led.step.t_step / local.step.t_step,
            })
    write("s4_cluster_growth", pd.DataFrame(rows), base,
          "A 120 km 800 Gbit/s stitch, held fixed, serving progressively larger "
          "clusters. Contention grows with the cluster and the circuit does not, "
          "which is why a two-site result at one scale does not transfer upward.")


def exp_s5_width_sweep(base: ScenarioConfig) -> None:
    """Retention against circuit width: where widening stops buying anything."""
    rows: List[Dict] = []
    for cut in ("pp", "dp"):
        for regime in (REGIMES[2], REGIMES[3], REGIMES[4]):
            for placement in PLACEMENTS:
                for width, retention in width_sweep(
                    base, cut, SWEEP_GBPS, regime, placement=placement
                ):
                    rows.append({
                        "cut": cut, "regime": regime.name, "placement": placement,
                        "circuit_gbps": width, "retention": retention,
                    })
    write("s5_width_sweep", pd.DataFrame(rows), base,
          "Eleven widths from 100 Gbit/s to 204.8 Tbit/s. The curve flattens once "
          "the per-rank share hits the NIC line rate; past that, width is inert.")


def exp_s6_exposed_bytes(base: ScenarioConfig) -> None:
    """Bytes across the boundary, raw and exposed, with the circuit lower bound."""
    rows: List[Dict] = []
    for cut in CUTS:
        raw = bytes_across_boundary(base, cut)
        exposed = exposed_bytes_across_boundary(base, cut)
        rows.append({
            "cut": cut,
            "raw_bytes_gb": raw / 1e9,
            "exposed_bytes_gb": exposed / 1e9,
            "hideable_fraction": 1.0 - (exposed / raw) if raw > 0 else float("nan"),
            "circuit_lower_bound_s_at_400g": circuit_lower_bound_s(base, cut, 400.0),
        })
    write("s6_exposed_bytes", pd.DataFrame(rows), base,
          "Why the atlas orders the way it does. Raw bytes across the cut predict "
          "the ordering incorrectly; bytes weighted by what cannot be overlapped "
          "predict it correctly.")


def exp_s7_stitch_sea(base: ScenarioConfig) -> None:
    """Price a stitch in accelerators: how many of the far hall actually arrive."""
    rows: List[Dict] = []
    for width in (400.0, 800.0, 3200.0):
        for cut in ("pp", "dp"):
            for regime in REGIMES:
                v = stitch_equivalent_accelerators(base, cut, regime, width)
                if v is None:
                    continue
                rows.append({
                    "circuit_gbps": width, "cut": cut, "regime": regime.name,
                    "accelerators_connected": v.accelerators_connected,
                    "accelerators_delivered": v.accelerators_delivered,
                    "efficiency": v.efficiency,
                })
    write("s7_stitch_sea", pd.DataFrame(rows), base,
          "Substitution-Equivalent Accelerators with a distance term: a stitch that "
          "connects N accelerators and delivers M of them can be compared directly "
          "against buying M in the hall you already have. No currency, per charter.")


EXPERIMENTS: Dict[str, Callable[[ScenarioConfig], None]] = {
    "s1": exp_s1_atlas,
    "s2": exp_s2_placement,
    "s3": exp_s3_ring_scaling,
    "s4": exp_s4_cluster_growth,
    "s5": exp_s5_width_sweep,
    "s6": exp_s6_exposed_bytes,
    "s7": exp_s7_stitch_sea,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", default=[], help="experiment ids, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="fast subset for CI")
    args = parser.parse_args(argv)

    if args.list:
        for key, fn in EXPERIMENTS.items():
            print(f"{key}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    base = load_scenario(BASELINE)
    chosen = args.experiments or (["s2", "s6"] if args.smoke else list(EXPERIMENTS))
    for key in chosen:
        if key not in EXPERIMENTS:
            print(f"unknown experiment: {key}", file=sys.stderr)
            return 2
        print(f"[{key}] {(EXPERIMENTS[key].__doc__ or '').strip().splitlines()[0]}")
        EXPERIMENTS[key](base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
