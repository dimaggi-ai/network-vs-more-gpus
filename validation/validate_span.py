"""Validation program for the latency-regime atlas and the span tier.

Same discipline as ``validate_llama3.py``, extended to the three-kind registry
the series uses. Every point declares what kind of statement it is, because the
kinds carry very different evidential weight and a table that mixes them
silently is a highlight reel:

* **calibrated** --- pinned to a figure someone else published. Passing proves
  the model has not drifted from the citation. It does not prove the model
  predicts anything.
* **emergent** --- not tuned, and stated as an *ordering* wherever an ordering
  will carry the claim, so there is no threshold to fit. These must survive
  being recomputed on configurations they were not written against.
* **sanity** --- a property of the model's own structure. ``ref`` is always
  ``"-"``: citing a source beside a self-consistency check would dress it up as
  an external result.

Two anchors were demoted while this registry was being written, and the reasons
are worth keeping in view because they generalise.

The Corning study reports that doubling inter-datacentre bandwidth improves
overlap by at most 0.66 percent. This model reproduces a gain of anywhere from
21 percent to exactly zero for that same doubling, depending only on the circuit
width you start from -- and the study does not publish the width it modelled.
A first draft of this file "passed" that anchor by starting the doubling at 204.8
Tbit/s, which is to say by choosing the answer. It is now DECLINED. What survives
is the study's *short-distance* claim, which can be stated without reference to
any width and therefore has no free parameter to fit.

The zero-gain result itself turned out to be structural rather than empirical:
once the circuit is wider than the ranks' own NICs can fill, widening it further
changes nothing by construction. That is now filed as sanity, not calibration.

What this program does **not** check is printed as a DECLINED list at the end.
A registry that lists only the anchors it passes tells you nothing about the
anchors it avoided.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netcap.accounting import build_ledger  # noqa: E402
from netcap.config import (  # noqa: E402
    AcceleratorSpec,
    KernelSpec,
    ModelSpec,
    ParallelismSpec,
    ReliabilitySpec,
    ScenarioConfig,
    TopologySpec,
    load_scenario,
    replace_nested,
)
from netcap.performance import (  # noqa: E402
    group_layout,
    span_alpha_seconds,
    span_share_bytes_per_s,
)
from netcap.regimes import (  # noqa: E402
    REGIMES,
    Regime,
    _local_baseline,
    _spanned,
    exposed_bytes_across_boundary,
)

# --- constants, hoisted so no tolerance is computed from the run it judges ---

CAMPUS = Regime("campus-stitch", 40.0, "")
REGION = Regime("region", 40000.0, "")
METRO_120KM_US = 600.0  # 120 km at ~5 us/km one way, the OFC 2025 field trial
TEN_KM_US = 50.0
ADJACENT_US = 0.05  # a hall next door: the zero-distance end of the comparison
OFC_CIRCUIT_GBPS = 800.0
CAMPUS_CIRCUIT_GBPS = 400.0
PROBE_GBPS = 204_800.0  # a deliberately unphysical circuit, used ONLY inside
                        # fixed-width comparisons where its artifacts cancel

#: Circuit widths this repository actually reports retention numbers at. The
#: top of the range is 8x400G, the widest metro DCI class in deployment.
REPORTED_GBPS: Tuple[float, ...] = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
#: Extended ladder, including probe widths, for monotonicity only.
LADDER_GBPS: Tuple[float, ...] = REPORTED_GBPS + (6400.0, 12800.0, 25600.0, PROBE_GBPS)

# Published figures. Each is somebody else's number, not ours.
CORNING_NEAR_FIELD_CEILING = 0.005  # "near-complete overlap below 10 km"
CORNING_DOUBLING_CEILING = 0.0066  # used only in the NIC-cap sanity point
OFC_RETENTION_FLOOR = 0.90  # trial measured 0.9941 (PP) and 0.9895 (DP)

CLIFF_REMOVAL_FACTOR = 2.0  # hierarchical must beat blind by at least this much
RTT_SENSITIVITY_THRESHOLD = 0.05  # "materially responds to distance"

# Visibility floors. An ordering alone is satisfied by a flat line, and a flat
# line is what a model with the machinery deleted produces -- so each ordering
# below also requires the effect to be plainly present. These are set an order
# of magnitude under the measured effects on purpose: they discriminate between
# presence and absence, not between magnitudes, and tightening them toward the
# observed values would turn them into fitted parameters. tests/test_span.py
# proves each one is load-bearing by deleting the machinery and requiring red.
CLIFF_FLOOR = 0.5  # blind placement must add at least half a step time
DISTANCE_GROWTH_FLOOR = 1.5  # furthest regime vs nearest, blind
RING_GROWTH_FLOOR = 2.0  # largest crossing ring vs smallest
EXACT = 1e-12
NEAR_EXACT = 1e-9

GPT3_175B = ModelSpec(
    name="gpt3-175b",
    n_layers=96,
    hidden=12288,
    n_params=1.75e11,
    seq_len=2048,
    vocab=50257,
    n_heads=96,
    n_kv_heads=96,
    ffn_multiplier=4.0,
)
H100 = AcceleratorSpec(
    name="H100-SXM",
    peak_flops_bf16=9.895e14,
    memory_gb=80,
    scaleup_bw_gbps=3600,
    nic_bw_gbps=400,
    accelerators_per_node=8,
)


@dataclass
class Point:
    name: str
    kind: str  # calibrated | emergent | sanity
    ref: str
    expected: str
    actual: str
    ok: bool
    note: str = ""


#: Anchors this registry does NOT provide. Kept beside the points it does.
DECLINED: Tuple[Tuple[str, str], ...] = (
    (
        "the 26x magnitude of the published distance cliff",
        "Corning's ASTRA-sim study reports ~26x step-time penalty at 1,000 km, but its "
        "parallel plan, global batch, and ring construction are not published in the "
        "detail a reconstruction needs. This registry pins the qualitative behaviours it "
        "does report and the ring-size scaling law, and declines the number.",
    ),
    (
        "the same study's bandwidth-insensitivity claim at its own operating width",
        "The claim that doubling inter-datacentre bandwidth buys at most 0.66 percent is "
        "conditional on a circuit width the study does not publish. This model produces "
        "gains from 21 percent (doubling 400G) down to exactly zero (doubling past the "
        "NIC cap) for that same operation. Any width that makes the anchor pass would be "
        "a width chosen to make the anchor pass, so the anchor is not checked at all.",
    ),
    (
        "the OFC 2025 field trial's exact 99.41 / 98.95 percent figures",
        "Cluster size, global batch, and micro-batch schedule are not published. The "
        "model is checked against a floor those measurements clear, not against the "
        "measurements. It is conservative by roughly four points at the closest "
        "configuration this project can reconstruct.",
    ),
    (
        "the trial's ordering of the pipeline cut above the data-parallel cut",
        "The two measurements differ by 0.46 points. This model does not resolve a gap "
        "that size and makes no claim about which of the two cuts a metro stitch "
        "favours in that configuration.",
    ),
    (
        "spanning as a strict penalty at circuits wider than 12.8 Tbit/s",
        "Above that width the model makes a split job up to 0.36 percent FASTER than a "
        "hall-local one, because each hall's residual group occupies fewer pods and the "
        "cross-pod hop it drops costs more than the stitch exchange that replaces it. "
        "That is a real consequence of the tier structure, not a rounding error, and it "
        "is why retention is only reported at or below 3.2 Tbit/s. The probe circuit "
        "appears here solely inside fixed-width comparisons, where the effect cancels.",
    ),
    (
        "inference and interactive-SLO spans",
        "There is no inference metric in this model; useful capacity is a training "
        "quantity. The question of which interactive workloads clear which distance "
        "belongs to edge-continuum-placement, which has the latency-domain gates.",
    ),
    (
        "asynchronous and low-communication training across a stitch",
        "The model is synchronous-only. DiLoCo-class methods (SOURCES.md S17) change "
        "the communication structure this atlas assumes, so no cell here describes them.",
    ),
    (
        "optical-layer behaviour: insertion loss, bit-error rate, circuit flap, retune",
        "The stitch is a bandwidth and a latency here. Whether the declared circuit is "
        "the measured circuit, and what happens when it degrades, is a separate "
        "world-model problem and is not modelled anywhere in this repository.",
    ),
    (
        "more than two halls",
        "The span tier accepts halls > 2 and the ring term is general, but no published "
        "measurement was found to check a three-or-more-hall configuration against, so "
        "no result in this repository is reported for one.",
    ),
    (
        "any cost expressed in currency",
        "The research charter forbids currency figures project-wide: SEA is a break-even "
        "ratio and quoting a price would import an assumption the model does not test. "
        "The dollar comparison lives in compute-power-placement.",
    ),
)


def _trial_class_scenario(world: int, tp: int, pp: int, dp: int) -> ScenarioConfig:
    """A configuration in the class of the OFC 2025 two-site trial."""
    return ScenarioConfig(
        name=f"trial-class-{world}",
        model=GPT3_175B,
        accelerator=H100,
        topology=TopologySpec(scaleup_domain=8, pod_size=min(world, 3072), oversubscription=1.0),
        parallelism=ParallelismSpec(
            tp=tp, pp=pp, cp=1, dp=dp, micro_batch=1, global_batch_seqs=max(1024, dp * 8)
        ),
        reliability=ReliabilitySpec(),
        kernel=KernelSpec(),
        window_days=30.0,
    )


def _span(scenario: ScenarioConfig, cut: str, alpha_us: float, gbps: float,
          placement: str) -> Optional[ScenarioConfig]:
    spanned = _spanned(_local_baseline(scenario), cut, alpha_us, gbps)
    if spanned is None:
        return None
    return replace_nested(spanned, **{"topology.span_placement": placement})


def _retention(scenario: ScenarioConfig, cut: str, alpha_us: float, gbps: float,
               placement: str = "hierarchical") -> float:
    base = _local_baseline(scenario)
    b = build_ledger(base).useful_capacity_fraction
    spanned = _span(scenario, cut, alpha_us, gbps, placement)
    if spanned is None:
        return float("nan")
    return build_ledger(spanned).useful_capacity_fraction / b if b > 0 else 0.0


def _step_ratio(scenario: ScenarioConfig, cut: str, alpha_us: float, gbps: float,
                placement: str) -> float:
    base = _local_baseline(scenario)
    b = build_ledger(base).step.t_step
    spanned = _span(scenario, cut, alpha_us, gbps, placement)
    if spanned is None:
        return float("nan")
    return build_ledger(spanned).step.t_step / b


def points(scenario: Optional[ScenarioConfig] = None) -> List[Point]:
    """The registry. Pass a scenario to re-run every point against it."""
    scn = scenario or load_scenario(ROOT / "configs/scenarios/llama3_405b_16k.yaml")
    out: List[Point] = []

    # ---------------- calibrated ----------------

    worst_near = 0.0
    for placement in ("hierarchical", "blind"):
        for cut in ("pp", "dp"):
            for gbps in REPORTED_GBPS:
                r0 = _step_ratio(scn, cut, ADJACENT_US, gbps, placement)
                r10 = _step_ratio(scn, cut, TEN_KM_US, gbps, placement)
                if r0 > 0:
                    worst_near = max(worst_near, r10 / r0 - 1.0)
    out.append(Point(
        "distance-does-not-bind-below-ten-km", "calibrated", "SOURCES.md S20",
        f"<= {CORNING_NEAR_FIELD_CEILING}", f"{worst_near:.5f}",
        worst_near <= CORNING_NEAR_FIELD_CEILING,
        "Corning's ASTRA-sim study reports near-complete compute-communication overlap "
        "below 10 km. Stated as the cost of moving a hall from next-door to 10 km at a "
        "FIXED circuit, which isolates distance and leaves no width to choose: the worst "
        "case over every reported width, both placements, and both cuts is quoted. A "
        "model that showed a penalty here would be wrong about the short end of the "
        "curve no matter what it did at distance.",
    ))

    trial = _trial_class_scenario(512, 8, 8, 8)
    pp_ret = _retention(trial, "pp", METRO_120KM_US, OFC_CIRCUIT_GBPS)
    dp_ret = _retention(trial, "dp", METRO_120KM_US, OFC_CIRCUIT_GBPS)
    both = min(pp_ret, dp_ret)
    out.append(Point(
        "metro-two-site-both-cuts-clear-the-floor", "calibrated", "SOURCES.md S18",
        f">= {OFC_RETENTION_FLOOR}", f"{both:.4f}", both >= OFC_RETENTION_FLOOR,
        "The OFC 2025 field trial ran 175B across two datacentres 120 km apart on an "
        "800 Gbit/s link and measured 99.41 percent (pipeline) and 98.95 percent "
        "(data-parallel) relative training efficiency. This is a floor those "
        "measurements clear comfortably, not a reconstruction of them: see DECLINED.",
    ))

    # ---------------- emergent ----------------

    plans = [
        ("llama3-405b-16k", scn),
        ("trial-class-512", _trial_class_scenario(512, 8, 8, 8)),
        ("trial-class-2048", _trial_class_scenario(2048, 8, 8, 32)),
        ("trial-class-8192", _trial_class_scenario(8192, 8, 8, 128)),
    ]
    violations = []
    for label, plan in plans:
        eb = {c: exposed_bytes_across_boundary(plan, c) for c in ("pp", "dp")}
        rt = {c: _retention(plan, c, CAMPUS.alpha_us, CAMPUS_CIRCUIT_GBPS) for c in ("pp", "dp")}
        cheaper = "pp" if eb["pp"] < eb["dp"] else "dp"
        better = "pp" if rt["pp"] > rt["dp"] else "dp"
        if cheaper != better:
            violations.append(f"{label}: fewer-exposed-bytes={cheaper} but higher-retention={better}")
    out.append(Point(
        "exposed-bytes-order-the-cuts", "emergent", "-",
        "0 violations", f"{len(violations)} violations", not violations,
        "The atlas is not a table of cut types. Across four parallel plans, the cut that "
        "leaves fewer bytes unhidden on the circuit is the cut that retains more useful "
        "capacity -- so which cut a stitch favours is a property of the plan, not of the "
        "cut. Raw bytes get this wrong: on the GPT-3 plans the pipeline cut moves far "
        "fewer bytes than the data-parallel cut and still retains less. Stated as an "
        "ordering because there is nothing in it to tune. "
        + ("Violations: " + "; ".join(violations) if violations else ""),
    ))

    ladder = [(128, 8, 16), (256, 8, 8), (512, 8, 4), (1024, 8, 2)]
    ratios = []
    for dp, tp, pp in ladder:
        plan = replace_nested(scn, **{
            "parallelism.tp": tp, "parallelism.pp": pp, "parallelism.dp": dp,
            "parallelism.global_batch_seqs": max(2048, dp * 2)})
        ratios.append(_step_ratio(plan, "dp", REGION.alpha_us, PROBE_GBPS, "blind"))
    grew = ratios[-1] / ratios[0] if ratios[0] > 0 else 0.0
    monotone = all(b > a for a, b in zip(ratios, ratios[1:])) and grew >= RING_GROWTH_FLOOR
    out.append(Point(
        "blind-penalty-grows-with-the-crossing-ring", "emergent", "-",
        f"strictly increasing, >= {RING_GROWTH_FLOOR}x end to end",
        " -> ".join(f"{r:.2f}" for r in ratios) + f" ({grew:.1f}x)", monotone,
        "A topology-blind ring pays the stitch latency 2(k-1) times, so its penalty must "
        "scale with the ring it puts across the boundary. This is the scaling law that "
        "makes the published cliffs reachable at their scale and modest at this one; the "
        "magnitude itself is declined.",
    ))

    blind_by_distance = [
        _step_ratio(scn, "dp", r.alpha_us, PROBE_GBPS, "blind") for r in REGIMES
    ]
    spread = blind_by_distance[-1] / blind_by_distance[0] if blind_by_distance[0] > 0 else 0.0
    mono_dist = (all(b >= a - EXACT for a, b in zip(blind_by_distance, blind_by_distance[1:]))
                 and spread >= DISTANCE_GROWTH_FLOOR)
    out.append(Point(
        "blind-penalty-grows-with-distance", "emergent", "-",
        f"non-decreasing, >= {DISTANCE_GROWTH_FLOOR}x end to end",
        " -> ".join(f"{r:.2f}" for r in blind_by_distance) + f" ({spread:.1f}x)", mono_dist,
        "The distance axis must bite somewhere, or the atlas has no latency story at all. "
        "Under blind placement it does, monotonically -- and by enough to see. Monotonicity "
        "alone would be satisfied by five identical numbers, which is exactly what a model "
        "that had forgotten about latency would produce.",
    ))

    blind_region = _step_ratio(scn, "dp", REGION.alpha_us, PROBE_GBPS, "blind")
    hier_region = _step_ratio(scn, "dp", REGION.alpha_us, PROBE_GBPS, "hierarchical")
    removed = ((blind_region - 1.0) >= CLIFF_FLOOR
               and (blind_region - 1.0) >= CLIFF_REMOVAL_FACTOR * (hier_region - 1.0))
    out.append(Point(
        "placement-not-distance-causes-the-cliff", "emergent", "-",
        f"blind excess >= {CLIFF_FLOOR} and >= {CLIFF_REMOVAL_FACTOR}x hierarchical",
        f"blind {blind_region:.2f}x vs hierarchical {hier_region:.2f}x", removed,
        "The headline. Holding the fiber, the circuit, and the job fixed and changing "
        "only the order the ranks sit in removes most of the distance penalty. This is "
        "the sense in which scale-across is a software problem: the plant was never the "
        "binding constraint. Both sides are measured at the same width, so the "
        "probe-circuit artifact noted in DECLINED cancels. The absolute floor matters as "
        "much as the ratio here: with no cliff at all, 0 >= 2 x 0 holds and the claim "
        "would pass while saying nothing.",
    ))

    pp_gap = []
    for regime in REGIMES:
        blind = _step_ratio(scn, "pp", regime.alpha_us, CAMPUS_CIRCUIT_GBPS, "blind")
        hier = _step_ratio(scn, "pp", regime.alpha_us, CAMPUS_CIRCUIT_GBPS, "hierarchical")
        pp_gap.append(blind / hier if hier > 0 else 0.0)
    pp_placement = all(g >= CLIFF_REMOVAL_FACTOR for g in pp_gap)
    out.append(Point(
        "pipeline-placement-costs-at-every-distance", "emergent", "-",
        f"blind >= {CLIFF_REMOVAL_FACTOR}x hierarchical in all five regimes",
        " -> ".join(f"{g:.1f}x" for g in pp_gap), pp_placement,
        "A pipeline laid out round-robin puts every one of its pp-1 stage boundaries "
        "on a hall edge; laid out contiguously it puts exactly one there. The gap is "
        "flat across the distance regimes, which is the cleanest statement of the "
        "headline available anywhere in this registry: for the pipeline cut the "
        "penalty is caused by the launcher and not by the fiber at all.",
    ))

    cluster_ladder = [512, 2048, 8192]
    pp_by_size = [
        _retention(_trial_class_scenario(n, 8, 8, n // 64), "pp", METRO_120KM_US, OFC_CIRCUIT_GBPS)
        for n in cluster_ladder
    ]
    falls = all(b < a for a, b in zip(pp_by_size, pp_by_size[1:]))
    out.append(Point(
        "a-metro-stitch-degrades-as-the-cluster-grows", "emergent", "-",
        "strictly decreasing", " -> ".join(f"{r:.3f}" for r in pp_by_size), falls,
        "Contention scales with the cluster while the circuit does not, so a two-site "
        "configuration demonstrated at one scale does not transfer to a larger one on the "
        "same link. This is the reason a successful field trial is not a frontier-scale "
        "result, and it is the model's sharpest warning against extrapolating one.",
    ))

    sens = {}
    for cut in ("tp", "pp", "dp"):
        near = _retention(scn, cut, CAMPUS.alpha_us, PROBE_GBPS)
        far = _retention(scn, cut, REGION.alpha_us, PROBE_GBPS)
        sens[cut] = abs(near - far) / near if near > 0 else math.inf
    only_tp = (
        sens["tp"] > RTT_SENSITIVITY_THRESHOLD
        and sens["pp"] <= RTT_SENSITIVITY_THRESHOLD
        and sens["dp"] <= RTT_SENSITIVITY_THRESHOLD
    )
    out.append(Point(
        "tensor-parallel-is-the-only-distance-bound-cut", "emergent", "-",
        f"tp > {RTT_SENSITIVITY_THRESHOLD}, pp and dp <= {RTT_SENSITIVITY_THRESHOLD}",
        ", ".join(f"{k}={v:.3f}" for k, v in sens.items()), only_tp,
        "On a circuit wide enough that bandwidth has stopped binding, moving from a "
        "campus stitch to a coast-to-coast path costs the pipeline and data-parallel cuts "
        "almost nothing and costs the tensor-parallel cut a great deal, because tensor "
        "parallelism pays the stitch latency once per layer per micro-batch. Measured as "
        "a near-versus-far ratio at one width, so the probe artifact cancels.",
    ))

    # ---------------- sanity ----------------

    local = build_ledger(_local_baseline(scn))
    plain = build_ledger(
        replace_nested(scn, **{"topology.halls": 1, "topology.span_dimension": "none"})
    )
    identical = abs(local.step.t_step - plain.step.t_step) < EXACT
    out.append(Point(
        "hall-local-is-the-published-three-tier-model", "sanity", "-",
        "step times identical", f"delta {abs(local.step.t_step - plain.step.t_step):.3e} s",
        identical,
        "Every span term is exactly zero when halls == 1, which is what lets this "
        "extension ship inside a repository whose published results must not move.",
    ))

    faster = []
    for cut in ("tp", "cp", "pp", "dp", "checkpoint"):
        for regime in REGIMES:
            for gbps in REPORTED_GBPS:
                for placement in ("hierarchical", "blind"):
                    spanned = _span(scn, cut, regime.alpha_us, gbps, placement)
                    if spanned is None:
                        continue
                    if build_ledger(spanned).step.t_step < local.step.t_step - NEAR_EXACT:
                        faster.append(f"{cut}@{regime.name}@{gbps:.0f}G/{placement}")
    out.append(Point(
        "spanning-never-speeds-a-step-up-at-reported-widths", "sanity", "-",
        "no cell faster than hall-local", f"{len(faster)} faster cells", not faster,
        "Splitting a job across a stitch must not make it finish sooner. Checked over "
        "every cut, regime, reported width, and placement. This holds up to 12.8 Tbit/s "
        "and then fails, for a structural reason recorded in DECLINED -- which is why "
        "the reported range stops where it does. "
        + ("Faster: " + "; ".join(faster[:4]) if faster else ""),
    ))

    slower = []
    for cut in ("tp", "cp", "pp", "dp", "checkpoint"):
        for regime in REGIMES:
            for placement in ("hierarchical", "blind"):
                seq = []
                for gbps in LADDER_GBPS:
                    spanned = _span(scn, cut, regime.alpha_us, gbps, placement)
                    if spanned is None:
                        break
                    seq.append(build_ledger(spanned).step.t_step)
                for a, b in zip(seq, seq[1:]):
                    if b > a + EXACT:
                        slower.append(f"{cut}@{regime.name}/{placement}")
                        break
    out.append(Point(
        "widening-a-circuit-never-slows-a-step", "sanity", "-",
        "monotone over the full width ladder", f"{len(slower)} inversions", not slower,
        "Ten widths from 100 Gbit/s to 204.8 Tbit/s, every cut, regime, and placement. "
        "An inversion would mean the contention or NIC-cap arithmetic has a sign error. "
        + ("Inverted: " + "; ".join(slower[:4]) if slower else ""),
    ))

    full_overlap = replace_nested(scn, **{
        "parallelism.overlap_dp": 1.0, "parallelism.overlap_tp": 1.0,
        "parallelism.overlap_pp": 1.0, "parallelism.overlap_cp": 1.0})
    fo_local = build_ledger(_local_baseline(full_overlap)).step.t_step
    fo_span = build_ledger(_span(full_overlap, "dp", REGION.alpha_us, PROBE_GBPS,
                                 "hierarchical")).step.t_step
    out.append(Point(
        "stitch-latency-survives-total-overlap", "sanity", "-",
        "step still grows at overlap = 1.0", f"{fo_span:.4f} s vs {fo_local:.4f} s",
        fo_span > fo_local + EXACT,
        "The serially chained hops drain at the step boundary and cannot hide behind "
        "compute. If setting every overlap fraction to 1.0 made distance free, the model "
        "would be treating latency as though it were bandwidth.",
    ))

    topo = _span(scn, "dp", CAMPUS.alpha_us, 1e12, "hierarchical").topology
    share = span_share_bytes_per_s(topo, 1.0, H100.nic_bw_gbps)
    nic_cap = H100.nic_bw_gbps * 1e9 * topo.net_efficiency / 8.0
    out.append(Point(
        "no-rank-outruns-its-own-nic", "sanity", "-",
        "share == NIC line rate at an unbounded circuit",
        f"{share / 1e9:.2f} vs {nic_cap / 1e9:.2f} GB/s", abs(share - nic_cap) < EXACT,
        "An arbitrarily fat circuit must not let the span tier beat the fabric the ranks "
        "are attached to.",
    ))

    capped = _step_ratio(scn, "dp", REGION.alpha_us, PROBE_GBPS, "blind")
    doubled = _step_ratio(scn, "dp", REGION.alpha_us, PROBE_GBPS * 2, "blind")
    gain = (capped - doubled) / capped if capped > 0 else math.inf
    out.append(Point(
        "widening-past-the-nic-cap-buys-exactly-nothing", "sanity", "-",
        "0.0", f"{gain:.6f}", abs(gain) < EXACT,
        "Once the per-rank share is pinned to the NIC line rate, doubling the circuit "
        "cannot change any term in the step. This is where the Corning "
        "bandwidth-insensitivity result comes from in this model -- and it is why that "
        f"anchor is DECLINED rather than calibrated: the study's {CORNING_DOUBLING_CEILING:.4f} "
        "ceiling is satisfied here by construction, not by agreement.",
    ))

    zero_alpha = span_alpha_seconds(
        group_layout(128, 128, _local_baseline(scn).topology, "dp", 16384),
        _local_baseline(scn).topology, "allreduce")
    out.append(Point(
        "no-stitch-no-stitch-latency", "sanity", "-", "0.0", f"{zero_alpha:.3e}",
        abs(zero_alpha) < EXACT,
        "The span latency term must vanish for a job that never leaves its hall.",
    ))

    return out


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main(argv: Sequence[str] = ()) -> int:
    pts = points()
    width = max(len(p.name) for p in pts) + 2
    print(f"{'point':<{width}}{'kind':<12}{'expected':<40}{'actual':<32}")
    print("-" * (width + 88))
    by_kind: Dict[str, List[Point]] = {}
    for p in pts:
        by_kind.setdefault(p.kind, []).append(p)
    for kind in ("calibrated", "emergent", "sanity"):
        for p in by_kind.get(kind, []):
            mark = "PASS" if p.ok else "FAIL"
            print(f"{p.name:<{width}}{p.kind:<12}{p.expected:<40}{p.actual:<32}{mark}")
    failed = [p for p in pts if not p.ok]

    print()
    print(f"{len(pts)} points: "
          + ", ".join(f"{len(by_kind.get(k, []))} {k}" for k in ("calibrated", "emergent", "sanity"))
          + f" -- {len(pts) - len(failed)} pass, {len(failed)} fail")

    print()
    print("DECLINED -- anchors this registry does not provide:")
    for title, why in DECLINED:
        print(f"  * {title}")
        for line in _wrap(why, 90):
            print(f"      {line}")

    out = ROOT / "results/validation/span_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"points": [asdict(p) for p in pts],
             "declined": [{"title": t, "reason": r} for t, r in DECLINED],
             "all_pass": not failed},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
