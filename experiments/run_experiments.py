"""Experiment runner. Every headline number in the paper is produced here.

Raw outputs are written to ``results/raw/`` and are treated as immutable: the
figure scripts read them and never modify them. Each experiment writes a
sidecar ``.meta.json`` recording the model version, the git commit, the random
seed, and the scenario digest, so a figure can always be traced to the run that
produced it.

Run ``python experiments/run_experiments.py --list`` to see the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import netcap  # noqa: E402
from netcap import build_ledger, load_scenario  # noqa: E402
from netcap.config import ScenarioConfig, replace_nested  # noqa: E402
from netcap.metrics import (  # noqa: E402
    Intervention,
    naive_equivalent_accelerators,
    rescale_pool,
    scaling_curve,
    shapley_attribution,
    substitution_equivalent_accelerators,
)

RAW = ROOT / "results/raw"
BASELINE = ROOT / "configs/scenarios/reference_405b_16k.yaml"
INTERVENTIONS = ROOT / "configs/interventions.yaml"
SEED = 20260805


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


def load_interventions():
    data = yaml.safe_load(INTERVENTIONS.read_text(encoding="utf-8"))
    items = [
        Intervention(
            name=i["name"],
            overrides=i.get("overrides", {}),
            multipliers=i.get("multipliers", {}),
            description=i.get("description", ""),
        )
        for i in data["interventions"]
    ]
    return items, data["regimes"]


def validity_fields(cfg: ScenarioConfig) -> Dict[str, float]:
    """Model-validity metadata attached to every decision-relevant row.

    Rows outside the envelope are kept in the raw output but excluded from
    headline claims by the analysis stage, so the exclusion is auditable rather
    than silent.
    """
    timing = build_ledger(cfg).timing
    return {
        "recovery_pressure": timing.recovery_pressure,
        "within_validity_envelope": timing.within_validity_envelope,
    }


def at_scale(base: ScenarioConfig, n_accelerators: int):
    """Re-plan the baseline for a different job size."""
    overhead = 1.0 + base.reliability.spare_fraction + base.reliability.stranded_fraction
    return rescale_pool(base, int(math.ceil(n_accelerators * overhead)))


# ---------------------------------------------------------------- experiments


def exp_e1_ledger_vs_scale(base: ScenarioConfig) -> None:
    """E1: where accelerator time goes, across scale and oversubscription.

    Hypothesis H1: as scale grows at fixed global batch, the productive share
    falls, and the loss shifts from communication-dominated to failure-dominated.
    """
    rows = []
    for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        for sigma in (1.0, 4.0, 7.0, 16.0):
            cfg = at_scale(replace_nested(base, **{"topology.oversubscription": sigma}), n)
            if cfg is None:
                continue
            ledger = build_ledger(cfg)
            row = ledger.as_dict()
            row.update({"n_accelerators": cfg.parallelism.world_size, "oversubscription": sigma})
            rows.append(row)
    write(
        "e1_ledger_vs_scale",
        pd.DataFrame(rows),
        base,
        "Four-fate ledger across job size and cross-pod oversubscription.",
    )


def exp_e2_sea_by_regime(base: ScenarioConfig) -> None:
    """E2: substitution value of each intervention, in each regime and at each scale.

    Hypothesis H2: the ranking of interventions by SEA is not constant across
    regimes. If it is constant, the decision layer collapses to a rule of thumb
    and the contribution narrows to the accounting framework.
    """
    interventions, regimes = load_interventions()
    rows = []
    for regime in regimes:
        for n in (4096, 16384, 65536):
            cfg = at_scale(replace_nested(base, **regime.get("overrides", {})), n)
            if cfg is None:
                continue
            for interv in interventions:
                result = substitution_equivalent_accelerators(cfg, interv)
                result.update(
                    {
                        "regime": regime["name"],
                        "n_accelerators": cfg.parallelism.world_size,
                        "description": interv.description,
                        **validity_fields(cfg),
                    }
                )
                rows.append(result)
    write(
        "e2_sea_by_regime",
        pd.DataFrame(rows),
        base,
        "SEA of each intervention across regimes and scales. Primary result.",
    )


def exp_e3_decision_map(base: ScenarioConfig) -> None:
    """E3: which intervention ranks first across a two-parameter regime grid.

    Independent variables: node failure rate and cross-pod oversubscription.
    This produces the decision-boundary map that is the paper's headline figure.
    """
    interventions, _ = load_interventions()
    rates = np.geomspace(1e-3, 3e-2, 12)
    sigmas = np.array([1.0, 2.0, 4.0, 7.0, 12.0, 16.0])
    rows = []
    for rate in rates:
        for sigma in sigmas:
            for n in (16384, 65536):
                cfg = at_scale(
                    replace_nested(
                        base,
                        **{
                            "reliability.failure_rate_per_node_day": float(rate),
                            "topology.oversubscription": float(sigma),
                        },
                    ),
                    n,
                )
                if cfg is None:
                    continue
                best_name, best_sea = None, -np.inf
                per = {}
                for interv in interventions:
                    sea = substitution_equivalent_accelerators(cfg, interv)["sea"]
                    per[interv.name] = sea
                    if sea > best_sea:
                        best_name, best_sea = interv.name, sea
                rows.append(
                    {
                        "failure_rate_per_node_day": float(rate),
                        "oversubscription": float(sigma),
                        "n_accelerators": cfg.parallelism.world_size,
                        "winner": best_name,
                        "winner_sea": best_sea,
                        **validity_fields(cfg),
                        **{f"sea_{k}": v for k, v in per.items()},
                    }
                )
    write(
        "e3_decision_map",
        pd.DataFrame(rows),
        base,
        "Highest-SEA intervention over a failure-rate by oversubscription grid.",
    )


def exp_e4_counterexamples(base: ScenarioConfig) -> None:
    """E4: where the thesis fails, and where scaling cannot substitute at all.

    Two negative cases are sought deliberately: regimes where every network
    intervention has near-zero SEA, and regimes where no accelerator count
    reproduces an intervention's effect.
    """
    interventions, _ = load_interventions()
    rows = []
    grid = [
        ("tiny_job", 1024, {}),
        ("short_sequence", 16384, {"model.seq_len": 2048}),
        ("long_sequence", 16384, {"model.seq_len": 32768}),
        ("weak_scaling", 16384, {}),
        (
            "perfect_network",
            16384,
            {"accelerator.nic_bw_gbps": 3200.0, "topology.oversubscription": 1.0},
        ),
        ("perfect_reliability", 16384, {"reliability.failure_rate_per_node_day": 1e-5}),
        ("huge_job", 131072, {}),
        (
            "failslow_fleet",
            16384,
            {
                "reliability.failslow_prob_per_node": 0.01,
                "reliability.failslow_slowdown": 1.5,
            },
        ),
        ("tiny_batch", 16384, {"parallelism.global_batch_seqs": 512}),
        ("huge_batch", 16384, {"parallelism.global_batch_seqs": 8192}),
    ]
    for label, n, overrides in grid:
        cfg = replace_nested(base, **overrides)
        if label == "weak_scaling":
            cfg = replace(cfg, scaling_mode="weak")
        cfg = at_scale(cfg, n)
        if cfg is None:
            continue
        ledger = build_ledger(cfg)
        for interv in interventions:
            result = substitution_equivalent_accelerators(cfg, interv)
            result.update(
                {
                    "case": label,
                    "n_accelerators": cfg.parallelism.world_size,
                    "baseline_ucf": ledger.useful_capacity_fraction,
                    "baseline_blocked_comm_share": ledger.blocked_comm / ledger.total,
                    "baseline_failure_share": (ledger.discarded + ledger.blocked_restart)
                    / ledger.total,
                    "recovery_pressure": ledger.timing.recovery_pressure,
                    "within_validity_envelope": ledger.timing.within_validity_envelope,
                }
            )
            rows.append(result)
    write(
        "e4_counterexamples",
        pd.DataFrame(rows),
        base,
        "Negative cases: regimes where network investment does not pay, and "
        "regimes where accelerators cannot substitute for the intervention.",
    )


def exp_e5_naive_bias(base: ScenarioConfig) -> None:
    """E5: bias of the informal 'equivalent GPUs recovered' metric versus SEA.

    Required experiment: the defense against the charge that the accounting
    framework repackages existing metrics depends on showing that the naive
    composition is quantitatively wrong, not merely inelegant.
    """
    interventions, _ = load_interventions()
    rows = []
    for n in (2048, 8192, 16384, 32768, 65536):
        for rate in (1e-3, 5e-3, 1.5e-2):
            cfg = at_scale(
                replace_nested(base, **{"reliability.failure_rate_per_node_day": rate}), n
            )
            if cfg is None:
                continue
            base_ledger = build_ledger(cfg)
            for interv in interventions:
                treated = build_ledger(interv.apply(cfg))
                result = substitution_equivalent_accelerators(cfg, interv)
                naive = naive_equivalent_accelerators(base_ledger, treated)
                rows.append(
                    {
                        "intervention": interv.name,
                        "n_accelerators": cfg.parallelism.world_size,
                        "failure_rate_per_node_day": rate,
                        "sea": result["sea"],
                        "naive": naive,
                        "ratio_naive_over_sea": naive / result["sea"]
                        if result["sea"] > 0 and math.isfinite(result["sea"])
                        else float("nan"),
                        "baseline_ucf": base_ledger.useful_capacity_fraction,
                        "ettr_times_step_efficiency": base_ledger.effective_training_time_ratio
                        * base_ledger.step.step_efficiency,
                        "recovery_pressure": base_ledger.timing.recovery_pressure,
                        "within_validity_envelope": base_ledger.timing.within_validity_envelope,
                    }
                )
    write(
        "e5_naive_bias",
        pd.DataFrame(rows),
        base,
        "Informal equivalent-GPU metric versus the substitution metric.",
    )


def exp_e6_attribution(base: ScenarioConfig) -> None:
    """E6: how far from additive are interventions when bundled.

    Hypothesis H3: interventions are substitutes, so the sum of individual SEA
    values exceeds the bundle's SEA. A negative additivity error would mean they
    are complements, which would be more surprising and is reported if found.
    """
    rows = []
    bundles = {
        "network_pair": ["bandwidth_2x", "flat_fabric"],
        "recovery_trio": ["fast_detection", "fast_restart", "fast_checkpoint"],
        "mixed_quad": ["bandwidth_2x", "flat_fabric", "fast_restart", "reliability_2x"],
    }
    interventions, regimes = load_interventions()
    by_name = {i.name: i for i in interventions}
    for regime in regimes:
        for n in (16384, 65536):
            cfg = at_scale(replace_nested(base, **regime.get("overrides", {})), n)
            if cfg is None:
                continue
            for bundle_name, members in bundles.items():
                parts = [by_name[m] for m in members]
                attribution = shapley_attribution(cfg, parts)
                row = {
                    "regime": regime["name"],
                    "n_accelerators": cfg.parallelism.world_size,
                    "bundle": bundle_name,
                    "bundle_sea": attribution["bundle_sea"],
                    "sum_of_individual_sea": attribution["sum_of_individual_sea"],
                    "additivity_error": attribution["additivity_error"],
                    "capped": attribution["capped"],
                    **validity_fields(cfg),
                }
                for member in members:
                    row[f"shapley_{member}"] = attribution[member]
                rows.append(row)
    write(
        "e6_attribution",
        pd.DataFrame(rows),
        base,
        "Shapley attribution and additivity error for bundled interventions.",
    )


def exp_e7_uncertainty(base: ScenarioConfig, n_draws: int = 400) -> None:
    """E7: rank stability of interventions under parameter uncertainty.

    Assumption parameters are drawn from documented ranges and the ranking is
    recomputed for each draw. The reported quantity is the fraction of draws in
    which each intervention ranks first, which is a more honest decision aid
    than a point ranking.
    """
    interventions, _ = load_interventions()
    rng = np.random.default_rng(SEED)
    rows = []
    for draw in range(n_draws):
        overrides = {
            "kernel.kernel_efficiency": float(rng.uniform(0.45, 0.70)),
            "topology.net_efficiency": float(rng.uniform(0.70, 0.95)),
            "parallelism.overlap_dp": float(rng.uniform(0.60, 0.95)),
            "parallelism.overlap_tp": float(rng.uniform(0.0, 0.35)),
            "reliability.failure_rate_per_node_day": float(rng.uniform(1.5e-3, 1.5e-2)),
            "reliability.detect_time_s": float(rng.uniform(30.0, 400.0)),
            "reliability.restart_time_s": float(rng.uniform(60.0, 900.0)),
            "reliability.checkpoint_write_s": float(rng.uniform(5.0, 90.0)),
            "reliability.straggler_cv": float(rng.uniform(0.0, 0.05)),
        }
        cfg = at_scale(replace_nested(base, **overrides), 16384)
        if cfg is None:
            continue
        best_name, best_sea = None, -np.inf
        record = {"draw": draw, **overrides}
        for interv in interventions:
            sea = substitution_equivalent_accelerators(cfg, interv)["sea"]
            record[f"sea_{interv.name}"] = sea
            if sea > best_sea:
                best_name, best_sea = interv.name, sea
        record["winner"] = best_name
        record["winner_sea"] = best_sea
        record.update(validity_fields(cfg))
        rows.append(record)
    write(
        "e7_uncertainty",
        pd.DataFrame(rows),
        base,
        f"Rank stability over {n_draws} draws from documented parameter ranges.",
    )


def exp_e8_cross_fidelity(base: ScenarioConfig) -> None:
    """E8: analytical renewal path versus event-driven Monte Carlo path."""
    rows = []
    for n in (2048, 8192, 16384, 65536):
        for rate in (1e-3, 5e-3, 1.5e-2, 3e-2):
            cfg = at_scale(
                replace_nested(base, **{"reliability.failure_rate_per_node_day": rate}), n
            )
            if cfg is None:
                continue
            a = build_ledger(cfg, fidelity="analytical")
            m = build_ledger(cfg, fidelity="monte_carlo", seed=SEED, n_replicates=300)
            rows.append(
                {
                    "n_accelerators": cfg.parallelism.world_size,
                    "failure_rate_per_node_day": rate,
                    "ucf_analytical": a.useful_capacity_fraction,
                    "ucf_monte_carlo": m.useful_capacity_fraction,
                    "rel_error_ucf": abs(m.useful_capacity_fraction - a.useful_capacity_fraction)
                    / a.useful_capacity_fraction,
                    "discarded_analytical": a.discarded,
                    "discarded_monte_carlo": m.discarded,
                    "rel_error_discarded": abs(m.discarded - a.discarded) / max(a.discarded, 1e-9),
                    "failures_analytical": a.timing.n_failures,
                    "failures_monte_carlo": m.timing.n_failures,
                    "recovery_pressure": a.timing.recovery_pressure,
                    "within_validity_envelope": a.timing.within_validity_envelope,
                }
            )
    write(
        "e8_cross_fidelity",
        pd.DataFrame(rows),
        base,
        "Agreement between the two independent reliability implementations.",
    )


def exp_e9_scaling_curves(base: ScenarioConfig) -> None:
    """E9: the throughput-versus-pool-size curves the substitution metric inverts.

    The shape of these curves determines whether an intervention is attainable
    by buying accelerators at all.
    """
    rows = []
    for sigma in (1.0, 4.0, 7.0, 16.0):
        for rate in (1e-3, 5e-3, 1.5e-2):
            cfg = replace_nested(
                base,
                **{
                    "topology.oversubscription": sigma,
                    "reliability.failure_rate_per_node_day": rate,
                },
            )
            cfg = at_scale(cfg, 4096)
            if cfg is None:
                continue
            for pool, productive in scaling_curve(cfg, max_factor=32.0):
                rows.append(
                    {
                        "oversubscription": sigma,
                        "failure_rate_per_node_day": rate,
                        "n_pool": pool,
                        "productive_accelerators": productive,
                        "productive_per_pool": productive / pool,
                    }
                )
    write(
        "e9_scaling_curves",
        pd.DataFrame(rows),
        base,
        "Productive throughput versus pool size under different regimes.",
    )


EXPERIMENTS: Dict[str, Callable[[ScenarioConfig], None]] = {
    "e1": exp_e1_ledger_vs_scale,
    "e2": exp_e2_sea_by_regime,
    "e3": exp_e3_decision_map,
    "e4": exp_e4_counterexamples,
    "e5": exp_e5_naive_bias,
    "e6": exp_e6_attribution,
    "e7": exp_e7_uncertainty,
    "e8": exp_e8_cross_fidelity,
    "e9": exp_e9_scaling_curves,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", default=[], help="experiment ids, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="fast subset for CI")
    args = parser.parse_args(argv)

    if args.list:
        for key, fn in EXPERIMENTS.items():
            first_line = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"{key}: {first_line}")
        return 0

    base = load_scenario(BASELINE)
    chosen = args.experiments or (["e1", "e8"] if args.smoke else list(EXPERIMENTS))
    for key in chosen:
        if key not in EXPERIMENTS:
            print(f"unknown experiment: {key}", file=sys.stderr)
            return 2
        print(f"[{key}] {(EXPERIMENTS[key].__doc__ or '').strip().splitlines()[0]}")
        if key == "e7" and args.smoke:
            exp_e7_uncertainty(base, n_draws=40)
        else:
            EXPERIMENTS[key](base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
