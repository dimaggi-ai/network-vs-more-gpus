"""Substitution metrics, scaling, and attribution.

The central quantity is **Substitution-Equivalent Accelerators (SEA)**: the
number of additional accelerators that, added to the baseline configuration,
would deliver the same productive accelerator-seconds per second as applying an
intervention to the baseline.

Formally, with ``Pi(N, c)`` the productive-accelerator throughput of a pool of
``N`` accelerators under configuration ``c``,

    SEA(c1 ; c0, N0) = the dN solving  Pi(N0 + dN, c0) = Pi(N0, c1)

This differs from the informal "equivalent GPUs recovered" figure obtained by
dividing avoided idle time by the measurement period. That informal figure
assumes a marginal accelerator is fully productive, which contradicts the
premise that accelerators are not fully productive, and it ignores the fact that
adding accelerators changes communication volume and job failure rate. Both
quantities are computed here so the bias can be measured rather than asserted;
see :func:`naive_equivalent_accelerators`.

When ``Pi(N, c0)`` never reaches the intervention's throughput at any pool size
within the search range, SEA is reported as infinite. That is not a modeling
failure; it is the regime in which the improvement cannot be bought with
accelerators at all.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .accounting import CapacityLedger, build_ledger
from .config import ScenarioConfig, replace_nested

MAX_SCALE_FACTOR = 64.0  # widest pool multiple considered when solving for SEA


@dataclass(frozen=True)
class Intervention:
    """A named configuration change with an optional cost in accelerator units.

    ``overrides`` set absolute target values ("deploy a system that detects in
    20 s"). ``multipliers`` scale whatever value the baseline has ("halve the
    failure rate"), so an intervention named 2x stays a 2x on every baseline it
    is applied to, including the regime-shifted baselines of the decision maps.
    An earlier version expressed relative interventions as absolute values,
    which silently changed their strength across regimes; see DECISIONS.md D14.

    ``cost_accelerator_equivalents`` is a normalized cost: how many
    fully-loaded accelerators the intervention costs, for the same pool. Using a
    ratio avoids inventing absolute infrastructure prices. Leave it ``None`` when
    the intervention's cost is the unknown being solved for; the break-even
    value is then the model's output.
    """

    name: str
    overrides: Dict[str, float] = field(default_factory=dict)
    multipliers: Dict[str, float] = field(default_factory=dict)
    cost_accelerator_equivalents: Optional[float] = None
    description: str = ""

    def apply(self, scenario: ScenarioConfig) -> ScenarioConfig:
        combined: Dict[str, float] = dict(self.overrides)
        for key, factor in self.multipliers.items():
            section, attr = key.split(".", 1)
            current = getattr(getattr(scenario, section), attr)
            scaled = current * factor
            combined[key] = int(round(scaled)) if isinstance(current, int) else scaled
        return replace_nested(scenario, **combined)


def rescale_pool(scenario: ScenarioConfig, n_pool_target: int) -> Optional[ScenarioConfig]:
    """Re-plan a scenario for a different pool size.

    Tensor, pipeline, and context parallel degrees are held fixed because they
    are driven by model shape and memory, not by pool size. Data parallelism
    absorbs the change, which is how operators actually scale a fixed model.

    Under ``strong`` scaling the global batch is held fixed, so more data
    parallel replicas mean fewer microbatches each: communication per step is
    unchanged while compute per step falls. Under ``weak`` scaling the global
    batch grows with the replica count up to ``max_global_batch_seqs``.

    Returns ``None`` when the target pool cannot host a valid configuration.
    """
    par = scenario.parallelism
    unit = par.tp * par.pp * par.cp
    overhead = 1.0 + scenario.reliability.spare_fraction + scenario.reliability.stranded_fraction
    n_job_target = int(math.floor(n_pool_target / overhead))
    dp_new = n_job_target // unit
    if dp_new < 1:
        return None

    new_par = dataclasses.replace(par, dp=dp_new)
    if scenario.scaling_mode == "weak":
        scaled = int(par.global_batch_seqs * dp_new / max(1, par.dp))
        new_par = dataclasses.replace(
            new_par, global_batch_seqs=min(scaled, scenario.max_global_batch_seqs)
        )
    # The schedule needs at least one microbatch per replica; with fewer
    # microbatches than stages the pipeline still runs, at a larger bubble,
    # which the bubble term prices.
    seqs_per_dp = new_par.global_batch_seqs / new_par.dp
    if seqs_per_dp / new_par.micro_batch < 1.0:
        return None

    return dataclasses.replace(
        scenario,
        parallelism=new_par,
        n_pool=int(math.ceil(dp_new * unit * overhead)),
        name=f"{scenario.name}@pool{n_pool_target}",
    )


def throughput(scenario: ScenarioConfig, **ledger_kwargs) -> float:
    """Productive accelerators, the currency SEA equalizes."""
    return build_ledger(scenario, **ledger_kwargs).productive_accelerators


def scaling_curve(
    scenario: ScenarioConfig, max_factor: float = MAX_SCALE_FACTOR
) -> List[Tuple[int, float]]:
    """Realizable (pool size, productive accelerators) pairs for a configuration.

    Only pool sizes that admit a valid data-parallel factorization appear;
    :func:`solve_sea` interpolates between adjacent realizable points.

    Sampling is dense for the first sixteen data-parallel steps and geometric
    (7 percent growth) beyond. Small SEA values are resolved by interpolating
    between *adjacent* realizable configurations, so the near-field marginal is
    exact rather than a chord across a coarse grid. A uniform coarse grid was
    found to overstate small SEA values by roughly 13 percent because the
    throughput curve is concave; see DECISIONS.md D13.
    """
    par = scenario.parallelism
    unit = par.tp * par.pp * par.cp
    overhead = 1.0 + scenario.reliability.spare_fraction + scenario.reliability.stranded_fraction
    base_dp = par.dp
    max_dp = max(base_dp + 1, int(base_dp * max_factor))

    dps = set(range(base_dp, min(base_dp + 16, max_dp) + 1))
    dp = float(max(dps))
    while dp < max_dp:
        dp *= 1.07
        dps.add(min(int(round(dp)), max_dp))
    dps.add(max_dp)

    curve: List[Tuple[int, float]] = []
    for dp_i in sorted(dps):
        pool = int(math.ceil(dp_i * unit * overhead))
        variant = rescale_pool(scenario, pool)
        if variant is None:
            continue
        curve.append((variant.n_pool, throughput(variant)))
    return curve


def solve_sea(
    baseline: ScenarioConfig,
    target_throughput: float,
    max_factor: float = MAX_SCALE_FACTOR,
) -> Tuple[float, bool, float]:
    """Solve ``Pi(N0 + dN, baseline) = target`` for ``dN``.

    Returns ``(sea, attainable, best_throughput_seen)``. When the baseline
    configuration cannot reach the target at any pool size up to
    ``max_factor`` times the baseline, ``attainable`` is ``False`` and ``sea`` is
    ``inf``.
    """
    curve = scaling_curve(baseline, max_factor=max_factor)
    if not curve:
        return float("inf"), False, 0.0
    n0 = curve[0][0]
    base_tp = curve[0][1]
    best = max(v for _, v in curve)
    if target_throughput <= base_tp:
        return 0.0, True, best

    prev_n, prev_v = curve[0]
    for n, v in curve[1:]:
        if v >= target_throughput:
            if v == prev_v:
                return float(n - n0), True, best
            frac = (target_throughput - prev_v) / (v - prev_v)
            n_star = prev_n + frac * (n - prev_n)
            return float(n_star - n0), True, best
        prev_n, prev_v = n, v
    return float("inf"), False, best


def substitution_equivalent_accelerators(
    baseline: ScenarioConfig,
    intervention: Intervention,
    max_factor: float = MAX_SCALE_FACTOR,
) -> Dict[str, float]:
    """SEA of one intervention against a baseline, with supporting quantities."""
    base_ledger = build_ledger(baseline)
    treated = intervention.apply(baseline)
    # An intervention that changes the spare or stranded fraction re-partitions
    # the paid pool. Re-plan at the baseline's pool size so the comparison stays
    # a fixed-budget one: more spares means fewer accelerators in the job.
    base_oh = 1.0 + baseline.reliability.spare_fraction + baseline.reliability.stranded_fraction
    new_oh = 1.0 + treated.reliability.spare_fraction + treated.reliability.stranded_fraction
    if abs(new_oh - base_oh) > 1e-12:
        replanned = rescale_pool(treated, baseline.n_pool)
        if replanned is not None:
            treated = replanned
    treated_ledger = build_ledger(treated)

    target = treated_ledger.productive_accelerators
    sea, attainable, best = solve_sea(baseline, target, max_factor=max_factor)

    return {
        "intervention": intervention.name,
        "baseline_pool": float(baseline.n_pool),
        "baseline_productive_accelerators": base_ledger.productive_accelerators,
        "treated_productive_accelerators": target,
        "delta_productive_accelerators": target - base_ledger.productive_accelerators,
        "sea": sea,
        "sea_fraction_of_pool": sea / baseline.n_pool if math.isfinite(sea) else float("inf"),
        "attainable_by_scaling": float(attainable),
        "best_throughput_by_scaling": best,
        "naive_equivalent": naive_equivalent_accelerators(base_ledger, treated_ledger),
        "baseline_ucf": base_ledger.useful_capacity_fraction,
        "treated_ucf": treated_ledger.useful_capacity_fraction,
        "cost_accelerator_equivalents": (
            float("nan")
            if intervention.cost_accelerator_equivalents is None
            else intervention.cost_accelerator_equivalents
        ),
    }


def naive_equivalent_accelerators(
    base: CapacityLedger, treated: CapacityLedger
) -> float:
    """The informal metric: avoided blocked-and-discarded time over the window.

    Computed only so that its bias relative to SEA can be reported. It is not
    used in any decision rule.
    """
    avoided = (base.blocked + base.discarded) - (treated.blocked + treated.discarded)
    return avoided / base.window_s if base.window_s > 0 else 0.0


def bundle_intervention(name: str, parts: Sequence[Intervention]) -> Intervention:
    """Combine interventions into one by merging their overrides and multipliers.

    On conflicting absolute keys, later entries win silently, so bundles with
    conflicting targets should be defined explicitly rather than composed.
    Conflicting multipliers compose by multiplication, which is the natural
    semantics for stacked relative improvements.
    """
    merged: Dict[str, float] = {}
    merged_mult: Dict[str, float] = {}
    for part in parts:
        merged.update(part.overrides)
        for key, factor in part.multipliers.items():
            merged_mult[key] = merged_mult.get(key, 1.0) * factor
    cost = 0.0
    for part in parts:
        if part.cost_accelerator_equivalents is None:
            cost = float("nan")
            break
        cost += part.cost_accelerator_equivalents
    return Intervention(
        name=name,
        overrides=merged,
        multipliers=merged_mult,
        cost_accelerator_equivalents=None if math.isnan(cost) else cost,
        description="bundle of: " + ", ".join(p.name for p in parts),
    )


def shapley_attribution(
    baseline: ScenarioConfig,
    interventions: Sequence[Intervention],
    max_factor: float = MAX_SCALE_FACTOR,
    cap: float = 1e9,
) -> Dict[str, float]:
    """Shapley values of each intervention's contribution to the bundle's SEA.

    SEA is not additive across interventions because throughput is a nonlinear
    function of the configuration: fixing one bottleneck changes how much the
    others are worth. The Shapley value is the standard order-independent
    attribution for exactly this situation, and with a small intervention set the
    full ``2^k`` evaluation is cheap.

    Infinite SEA values are capped at ``cap`` so the attribution stays finite;
    any capping is reported in the ``capped`` key.
    """
    k = len(interventions)
    if k == 0:
        return {}
    names = [i.name for i in interventions]

    cache: Dict[Tuple[int, ...], float] = {}
    capped = 0.0

    def value(subset: Tuple[int, ...]) -> float:
        nonlocal capped
        if subset in cache:
            return cache[subset]
        if not subset:
            cache[subset] = 0.0
            return 0.0
        parts = [interventions[i] for i in subset]
        bundle = bundle_intervention("+".join(interventions[i].name for i in subset), parts)
        result = substitution_equivalent_accelerators(baseline, bundle, max_factor=max_factor)
        v = result["sea"]
        if not math.isfinite(v):
            v = cap
            capped = 1.0
        cache[subset] = v
        return v

    phi = {name: 0.0 for name in names}
    idx = list(range(k))
    for i in idx:
        others = [j for j in idx if j != i]
        for size in range(k):
            weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for combo in combinations(others, size):
                s = tuple(sorted(combo))
                s_with = tuple(sorted(combo + (i,)))
                phi[names[i]] += weight * (value(s_with) - value(s))

    total = value(tuple(idx))
    naive_sum = sum(value((i,)) for i in idx)
    return {
        **phi,
        "bundle_sea": total,
        "sum_of_individual_sea": naive_sum,
        "additivity_error": (naive_sum - total) / total if total > 0 else float("nan"),
        "capped": capped,
    }


def break_even_cost(result: Dict[str, float]) -> float:
    """Cost, in accelerator equivalents, at which an intervention stops paying.

    An intervention is worth funding when its cost expressed in fully-loaded
    accelerator units is below its SEA. The break-even cost is therefore SEA
    itself, which is why the decision rule needs no absolute prices.
    """
    return result["sea"]


def rank_interventions(
    baseline: ScenarioConfig,
    interventions: Iterable[Intervention],
    max_factor: float = MAX_SCALE_FACTOR,
) -> List[Dict[str, float]]:
    """Rank interventions by SEA per unit cost where cost is known, else by SEA."""
    rows = [
        substitution_equivalent_accelerators(baseline, i, max_factor=max_factor)
        for i in interventions
    ]
    for row in rows:
        cost = row["cost_accelerator_equivalents"]
        row["value_ratio"] = (
            row["sea"] / cost if cost and not math.isnan(cost) and cost > 0 else float("nan")
        )
    key: Callable[[Dict[str, float]], float] = lambda r: (
        r["value_ratio"] if not math.isnan(r.get("value_ratio", float("nan"))) else r["sea"]
    )
    return sorted(rows, key=key, reverse=True)
