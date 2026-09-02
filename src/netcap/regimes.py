"""The latency-regime atlas: which parallel cut still produces useful work at
which round-trip time, and what a stitch between halls is worth.

Scale-up and scale-out are bounded by fabrics one vendor sells. **Scale-across**
--- joining halls and buildings that already have dark fiber, ROADMs, and
optical circuit switches between them --- is bounded instead by a decision:
whether a job may use that plant, and in what shape. This module answers the
first half of that decision numerically.

Two questions, two functions:

* :func:`atlas` --- *retention*. Take a job that runs inside one hall and place
  one named cut of it across a stitch, holding the world size and the parallel
  plan fixed. What fraction of its useful capacity survives? This isolates the
  cost of distance from the benefit of more accelerators.
* :func:`stitch_equivalent_accelerators` --- *worth*. A hall-local job is given
  a second hall's accelerators over a circuit. Of the accelerators the stitch
  connected, how many does it actually deliver? The answer is in the same units
  as :func:`netcap.metrics.substitution_equivalent_accelerators`, so "rent a
  circuit" and "buy accelerators in the hall you already have" become one
  comparison.

The regimes are the document's, and the physical reading of each is in
``docs/regime-atlas.md``. Nothing here introduces a new communication model: the
span tier is the existing alpha-beta hierarchy with one more level, and the only
structural addition is that the level's bandwidth is a *shared circuit* rather
than a per-accelerator entitlement (:func:`netcap.performance.span_share_bytes_per_s`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .accounting import build_ledger
from .config import ScenarioConfig, replace_nested
from .performance import span_share_bytes_per_s

# Checkpoint state per parameter: bf16 weights (2) + fp32 master copy (4) +
# two fp32 optimizer moments (8). The same 14 bytes/parameter the memory
# feasibility check in netcap.performance uses.
CHECKPOINT_BYTES_PER_PARAM = 14.0


@dataclass(frozen=True)
class Regime:
    """One rung of the latency map, named by what physically sits at that RTT."""

    name: str
    alpha_us: float
    reading: str


#: The five regimes. ``alpha_us`` is the one-way latency charged to a tier
#: crossing at that distance; the band each is drawn from is in the ``reading``.
REGIMES: Tuple[Regime, ...] = (
    Regime("scale-up", 0.5, "inside one scale-up domain (< 1 us)"),
    Regime("rail", 3.0, "scale-out rail or leaf inside one hall (1-5 us)"),
    Regime("campus-stitch", 40.0, "dedicated OCS or lambda across a campus (10-80 us)"),
    Regime("metro", 500.0, "contended metro ROADM path (0.1-2 ms)"),
    Regime("region", 40000.0, "region to region, coast to coast (10-80 ms)"),
)

#: The cuts an operator can place across a stitch. ``checkpoint`` is not a
#: parallel dimension: it is the job staying hall-local while its state crosses.
CUTS: Tuple[str, ...] = ("tp", "cp", "pp", "dp", "checkpoint")


def _local_baseline(scenario: ScenarioConfig) -> ScenarioConfig:
    """The same job with no cut across a stitch."""
    return replace_nested(
        scenario,
        **{"topology.halls": 1, "topology.span_dimension": "none", "topology.span_bw_gbps": 0.0},
    )


def checkpoint_state_bytes(scenario: ScenarioConfig) -> float:
    """Bytes of durable state one checkpoint of this job writes."""
    return scenario.model.n_params * CHECKPOINT_BYTES_PER_PARAM


def _spanned(
    scenario: ScenarioConfig, cut: str, alpha_us: float, circuit_gbps: float, halls: int = 2
) -> Optional[ScenarioConfig]:
    """Place ``cut`` across a stitch of ``circuit_gbps`` at ``alpha_us``.

    Returns ``None`` when the cut does not exist in this job (a dimension of
    degree 1 cannot be split across two halls).
    """
    if cut == "checkpoint":
        # The job's compute stays hall-local; only its state crosses. The write
        # is one global transfer, so it has the circuit to itself: contention 1.
        base = _local_baseline(scenario)
        share = span_share_bytes_per_s(
            replace_nested(
                base,
                **{
                    "topology.halls": halls,
                    "topology.span_dimension": "dp",
                    "topology.span_bw_gbps": circuit_gbps,
                    "topology.alpha_span_us": alpha_us,
                },
            ).topology,
            1.0,
            scenario.accelerator.nic_bw_gbps,
        )
        if share <= 0:
            return None
        cross_s = checkpoint_state_bytes(scenario) / share + alpha_us * 1e-6
        return replace_nested(
            base,
            **{
                "reliability.checkpoint_write_s": scenario.reliability.checkpoint_write_s
                + cross_s
            },
        )

    degree = getattr(scenario.parallelism, cut, 1)
    if degree < halls:
        return None
    return replace_nested(
        scenario,
        **{
            "topology.halls": halls,
            "topology.span_dimension": cut,
            "topology.span_bw_gbps": circuit_gbps,
            "topology.alpha_span_us": alpha_us,
        },
    )


@dataclass(frozen=True)
class AtlasCell:
    """One (regime, cut) verdict."""

    regime: str
    cut: str
    alpha_us: float
    retention: float  # useful capacity spanned / useful capacity hall-local
    step_ratio: float  # spanned step time / hall-local step time
    applicable: bool
    reason: str = ""


def atlas(
    scenario: ScenarioConfig,
    circuit_gbps: float,
    regimes: Sequence[Regime] = REGIMES,
    cuts: Sequence[str] = CUTS,
    halls: int = 2,
    placement: str = "hierarchical",
    **ledger_kwargs,
) -> List[AtlasCell]:
    """Useful-capacity retention for every (regime, cut) pair.

    The world size, the parallel plan, and every reliability parameter are held
    fixed across the whole table. The only thing that varies is where the job is
    cut and how far apart the two halves sit, which is what makes the cells
    comparable to each other.

    ``placement`` selects how ranks are ordered across the stitch. It defaults
    to ``hierarchical`` -- contiguous, one span-tier member per hall -- because
    that is what a topology-aware launcher produces. Passing ``blind`` gives the
    interleaved ring an unaware launcher produces, and the difference between
    the two tables is the central result of this module.
    """
    base = _local_baseline(scenario)
    base_ledger = build_ledger(base, **ledger_kwargs)
    base_ucf = base_ledger.useful_capacity_fraction
    base_step = base_ledger.step.t_step

    cells: List[AtlasCell] = []
    for regime in regimes:
        for cut in cuts:
            spanned = _spanned(base, cut, regime.alpha_us, circuit_gbps, halls)
            if spanned is not None:
                spanned = replace_nested(spanned, **{"topology.span_placement": placement})
            if spanned is None:
                cells.append(
                    AtlasCell(
                        regime.name,
                        cut,
                        regime.alpha_us,
                        float("nan"),
                        float("nan"),
                        False,
                        f"{cut} has degree {getattr(base.parallelism, cut, 1)} in this job",
                    )
                )
                continue
            ledger = build_ledger(spanned, **ledger_kwargs)
            cells.append(
                AtlasCell(
                    regime.name,
                    cut,
                    regime.alpha_us,
                    ledger.useful_capacity_fraction / base_ucf if base_ucf > 0 else 0.0,
                    ledger.step.t_step / base_step if base_step > 0 else math.inf,
                    True,
                )
            )
    return cells


def atlas_table(cells: Sequence[AtlasCell]) -> Dict[str, Dict[str, float]]:
    """``{regime: {cut: retention}}``, for figures and tests."""
    out: Dict[str, Dict[str, float]] = {}
    for cell in cells:
        out.setdefault(cell.regime, {})[cell.cut] = cell.retention
    return out


def surviving_cuts(cells: Sequence[AtlasCell], floor: float = 0.90) -> Dict[str, List[str]]:
    """Which cuts hold at least ``floor`` of hall-local useful capacity, by regime."""
    out: Dict[str, List[str]] = {}
    for cell in cells:
        if cell.applicable and cell.retention >= floor:
            out.setdefault(cell.regime, []).append(cell.cut)
    for regime in {c.regime for c in cells}:
        out.setdefault(regime, [])
    return out


@dataclass(frozen=True)
class StitchVerdict:
    """What a circuit delivered, against what it connected."""

    regime: str
    cut: str
    circuit_gbps: float
    accelerators_connected: int
    accelerators_delivered: float
    efficiency: float  # delivered / connected
    throughput_local: float
    throughput_spanned: float


def stitch_equivalent_accelerators(
    scenario: ScenarioConfig,
    cut: str,
    regime: Regime,
    circuit_gbps: float,
    **ledger_kwargs,
) -> Optional[StitchVerdict]:
    """Price a stitch in accelerators: SEA with a distance term.

    The baseline is a job that owns one hall. The intervention is a second
    hall's worth of accelerators, reachable only over a circuit. The verdict is
    how many *in-hall* accelerators would have bought the same productive
    throughput --- so a stitch that connects 8,192 accelerators and delivers
    2,000 of them has an efficiency of 0.24, and the operator can compare that
    directly against buying 2,000 accelerators in the hall they already have.

    Returns ``None`` when the cut cannot be placed across two halls.
    """
    from .metrics import rescale_pool, throughput

    half = _local_baseline(scenario)
    n_full = half.n_pool
    small = rescale_pool(half, n_full // 2)
    if small is None:
        return None
    small = _local_baseline(small)

    spanned = _spanned(half, cut, regime.alpha_us, circuit_gbps)
    if spanned is None:
        return None

    pi_local = throughput(small, **ledger_kwargs)
    pi_span = throughput(spanned, **ledger_kwargs)
    connected = n_full - small.n_pool

    # Productive accelerators are linear in pool size at fixed configuration, so
    # the in-hall accelerators equivalent to the stitch's gain is the gain
    # divided by the baseline's productive-accelerators-per-accelerator-owned.
    per_accelerator = pi_local / small.n_pool if small.n_pool else 0.0
    delivered = (pi_span - pi_local) / per_accelerator if per_accelerator > 0 else 0.0
    return StitchVerdict(
        regime=regime.name,
        cut=cut,
        circuit_gbps=circuit_gbps,
        accelerators_connected=connected,
        accelerators_delivered=delivered,
        efficiency=delivered / connected if connected else 0.0,
        throughput_local=pi_local,
        throughput_spanned=pi_span,
    )


# ---------------------------------------------------------------------------
# What actually crosses the boundary
# ---------------------------------------------------------------------------


def bytes_across_boundary(scenario: ScenarioConfig, cut: str, halls: int = 2) -> float:
    """Bytes the named cut pushes across the stitch in one step.

    Computed from the parallel plan rather than read out of the timing model, so
    it is an independent statement about the same configuration and can be used
    to check the timing model rather than restate it. A shared circuit
    serializes everything crossing it, so ``bytes_across_boundary / circuit
    capacity`` is a hard lower bound on the time a spanned step can take --- the
    bound :func:`atlas` is checked against in ``validation/validate_span.py``.
    """
    par, model = scenario.parallelism, scenario.model
    world = par.world_size
    if cut == "checkpoint":
        return checkpoint_state_bytes(scenario)
    if cut == "pp":
        if par.pp < halls:
            return 0.0
        act = par.micro_batch * (model.seq_len / par.cp) * model.hidden * par.act_bytes
        micro = (par.global_batch_seqs / par.dp) / par.micro_batch
        crossings = 2.0 * micro * (halls - 1) / (par.pp - 1)
        return (world / par.pp) * crossings * act
    if cut == "dp":
        if par.dp < halls:
            return 0.0
        from .performance import group_layout

        topo = replace_nested(
            scenario,
            **{
                "topology.halls": halls,
                "topology.span_dimension": "dp",
                "topology.span_bw_gbps": 1.0,
            },
        ).topology
        lay = group_layout(par.dp, par.tp * par.cp * par.pp, topo, "dp", world)
        grad = model.n_params / (par.tp * par.pp) * par.grad_bytes
        below = lay.per_scaleup if lay.per_scaleup > 1 else max(1, lay.scaleups_per_pod)
        chunk = grad / below
        return (world / par.dp) * (2.0 * (halls - 1) / halls) * chunk
    if cut == "tp":
        if par.tp < halls:
            return 0.0
        act = par.micro_batch * (model.seq_len / par.cp) * model.hidden * par.act_bytes
        reps = 4.0 * (model.n_layers / par.pp) * ((par.global_batch_seqs / par.dp) / par.micro_batch)
        return (world / par.tp) * reps * (2.0 * (halls - 1) / halls) * act
    return float("nan")


_OVERLAP_FIELD = {"tp": "overlap_tp", "cp": "overlap_cp", "pp": "overlap_pp", "dp": "overlap_dp"}


def exposed_bytes_across_boundary(
    scenario: ScenarioConfig, cut: str, halls: int = 2
) -> float:
    """Bytes across the stitch that cannot hide behind compute.

    The ordering of the atlas follows this quantity, not the raw byte count. A
    data-parallel gradient reduction moves more bytes than a pipeline handoff in
    some plans and fewer in others, but it is also the most overlappable
    collective in the step, and it is the *unhidden* remainder that lands on the
    critical path. Stated as a separate function because the atlas ordering is
    checked against it in ``validation/validate_span.py``, and a rule that only
    ever restates the timing model would not be a check.
    """
    raw = bytes_across_boundary(scenario, cut, halls)
    if cut == "checkpoint":
        return raw  # a blocking checkpoint write hides behind nothing
    overlap = getattr(scenario.parallelism, _OVERLAP_FIELD[cut], 0.0)
    return raw * (1.0 - overlap)


def circuit_lower_bound_s(
    scenario: ScenarioConfig, cut: str, circuit_gbps: float, halls: int = 2
) -> float:
    """Seconds per step the circuit alone forces, ignoring latency entirely."""
    capacity = circuit_gbps * 1e9 * scenario.topology.net_efficiency / 8.0
    if capacity <= 0:
        return math.inf
    return bytes_across_boundary(scenario, cut, halls) / capacity


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def width_sweep(
    scenario: ScenarioConfig,
    cut: str,
    widths_gbps: Sequence[float],
    regime: Regime,
    placement: str = "hierarchical",
    **ledger_kwargs,
) -> List[Tuple[float, float]]:
    """``[(circuit_gbps, retention)]`` for one cut at one distance."""
    base = _local_baseline(scenario)
    base_ucf = build_ledger(base, **ledger_kwargs).useful_capacity_fraction
    out: List[Tuple[float, float]] = []
    for width in widths_gbps:
        spanned = _spanned(base, cut, regime.alpha_us, width)
        if spanned is None:
            continue
        spanned = replace_nested(spanned, **{"topology.span_placement": placement})
        ucf = build_ledger(spanned, **ledger_kwargs).useful_capacity_fraction
        out.append((width, ucf / base_ucf if base_ucf > 0 else 0.0))
    return out


def placement_comparison(
    scenario: ScenarioConfig,
    cut: str,
    regimes: Sequence[Regime],
    circuit_gbps: float,
    **ledger_kwargs,
) -> List[Tuple[str, float, float]]:
    """``[(regime, blind_step_ratio, hierarchical_step_ratio)]``.

    The headline of the atlas: the published distance cliffs are a property of a
    topology-blind ring, and changing only the rank order removes them without
    touching the fiber.
    """
    base = _local_baseline(scenario)
    base_step = build_ledger(base, **ledger_kwargs).step.t_step
    out: List[Tuple[str, float, float]] = []
    for regime in regimes:
        ratios = []
        for placement in ("blind", "hierarchical"):
            spanned = _spanned(base, cut, regime.alpha_us, circuit_gbps)
            if spanned is None:
                ratios.append(float("nan"))
                continue
            spanned = replace_nested(spanned, **{"topology.span_placement": placement})
            ratios.append(build_ledger(spanned, **ledger_kwargs).step.t_step / base_step)
        out.append((regime.name, ratios[0], ratios[1]))
    return out
