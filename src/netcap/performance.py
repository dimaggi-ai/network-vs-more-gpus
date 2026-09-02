"""Healthy-step performance model: compute time, collective time, and the
split of a synchronous training step into productive and blocked time.

"Healthy" means no failure has occurred. Failure, detection, recovery, and
checkpoint effects are layered on top in :mod:`netcap.reliability`.

The formulations here are standard. FLOP counting follows the usual transformer
accounting (6 FLOPs per parameter per token for forward plus backward, with a
separate attention-score term). Collective volumes follow the Megatron-style
mapping of tensor, pipeline, context, and data parallelism onto collectives.
The intent is that an independent tool reproduces these numbers; see
``validation/VALIDATION_REPORT.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from . import collectives as cc
from .config import ModelSpec, ParallelismSpec, ScenarioConfig, TopologySpec

RECOMPUTE_FACTORS = {"none": 1.0, "selective": 1.10, "full": 4.0 / 3.0}


@dataclass(frozen=True)
class GroupLayout:
    """How a parallel group's ranks are distributed across the hierarchy.

    ``halls`` is 1 for every group except the one named by
    :attr:`~netcap.config.TopologySpec.span_dimension`: only the named cut
    crosses the stitch, everything else is laid out inside one hall.

    ``span_contention`` is how many groups of this dimension exist in the job,
    and therefore how many of them share the one circuit at the same point in
    the step. A synchronous step runs every group of a dimension concurrently,
    so the circuit's per-group share is its capacity divided by this number.
    It is the term that makes a stitch cheap for a cut with few groups and
    ruinous for a cut with many.
    """

    size: int
    per_scaleup: int  # members sharing one scale-up domain
    scaleups_per_pod: int  # distinct scale-up domains of this group inside one pod
    pods: int  # distinct pods the group spans
    halls: int = 1  # distinct halls the group spans
    span_contention: float = 1.0  # concurrent groups sharing one circuit
    span_ring: int = 1  # ring members at the span tier: halls, or the whole
    # group when placement is topology-blind


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def group_layout(
    size: int,
    stride: int,
    topology: TopologySpec,
    dimension: str = "",
    world: int = 0,
) -> GroupLayout:
    """Locate a strided rank group within the scale-up / pod / cross-pod / span hierarchy.

    Ranks of the group are ``base + i * stride`` for ``i`` in ``[0, size)``, the
    layout produced by the conventional nesting of parallelism dimensions. A
    dimension whose stride exceeds the scale-up domain places exactly one member
    per domain, which is why data-parallel collectives do not benefit from
    scale-up bandwidth.

    ``dimension`` names which parallel dimension this group is (``"tp"``,
    ``"cp"``, ``"pp"``, ``"dp"``). When it matches the topology's
    ``span_dimension`` the group is split evenly across ``halls`` and the
    intra-hall tiers are computed on the per-hall share; every other dimension
    stays inside one hall. ``world`` is the job's world size, used only to count
    how many groups of this dimension contend for the circuit.
    """
    if size <= 1:
        return GroupLayout(size, 1, 1, 1)

    spans = bool(dimension) and topology.spans and dimension == topology.span_dimension
    halls = min(topology.halls, size) if spans else 1
    size_local = _ceil_div(size, halls)

    g = max(1, topology.scaleup_domain)
    q = max(g, topology.pod_size)

    per_su = min(size_local, max(1, g // stride)) if stride <= g else 1
    n_su_groups = _ceil_div(size_local, per_su)

    su_per_pod_total = max(1, q // g)
    su_stride = max(1, (stride * per_su) // g) if stride * per_su >= g else 1
    per_pod_su = min(n_su_groups, max(1, su_per_pod_total // su_stride))
    pods = _ceil_div(n_su_groups, per_pod_su)

    contention = max(1.0, world / size) if (world and halls > 1) else 1.0
    # A topology-blind ring interleaves the group's ranks across the halls, so
    # the ring crosses the stitch on nearly every hop rather than twice. The
    # intra-hall tiers vanish: there is no local reduction to hide behind.
    blind = halls > 1 and topology.span_placement == "blind"
    span_ring = size if blind else halls
    if blind:
        return GroupLayout(size, 1, 1, 1, halls, contention, span_ring)
    return GroupLayout(size, per_su, per_pod_su, pods, halls, contention, span_ring)


def _tiers_from_layout(
    layout: GroupLayout,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
):
    eff = topology.net_efficiency
    tiers = []
    if layout.per_scaleup > 1:
        tiers.append(
            cc.Tier(
                "scaleup",
                layout.per_scaleup,
                scaleup_bw_gbps * 1e9 * eff / 8.0,
                topology.alpha_scaleup_us * 1e-6,
            )
        )
    if layout.scaleups_per_pod > 1:
        tiers.append(
            cc.Tier(
                "pod",
                layout.scaleups_per_pod,
                nic_bw_gbps * 1e9 * eff / 8.0,
                topology.alpha_pod_us * 1e-6,
            )
        )
    if layout.pods > 1:
        tiers.append(
            cc.Tier(
                "crosspod",
                layout.pods,
                nic_bw_gbps / max(1.0, topology.oversubscription) * 1e9 * eff / 8.0,
                topology.alpha_xpod_us * 1e-6,
            )
        )
    if layout.halls > 1:
        tiers.append(
            cc.Tier(
                "span",
                layout.span_ring,
                span_share_bytes_per_s(topology, layout.span_contention, nic_bw_gbps),
                topology.alpha_span_us * 1e-6,
            )
        )
    return tiers


def span_share_bytes_per_s(
    topology: TopologySpec, contention: float, nic_bw_gbps: float = 0.0
) -> float:
    """Per-group bandwidth over the stitch: circuit capacity split by contention.

    This is the one place where the span tier stops behaving like a fabric tier.
    A pod's bandwidth is quoted per accelerator; a circuit's is quoted for the
    circuit. Dividing by the number of groups crossing it at the same moment is
    what turns "how far is the other hall" into "which cut did you put across
    the stitch".
    """
    if topology.span_bw_gbps <= 0:
        return 0.0
    aggregate = topology.span_bw_gbps * 1e9 * topology.net_efficiency / 8.0
    share = aggregate / max(1.0, contention)
    if nic_bw_gbps > 0:
        # A rank cannot push into the stitch faster than its own NIC, however
        # wide the circuit is. Without this cap an arbitrarily fat circuit lets
        # the span tier outrun the fabric the ranks are actually attached to.
        share = min(share, nic_bw_gbps * 1e9 * topology.net_efficiency / 8.0)
    return share


def allreduce_on_layout(
    size_bytes: float,
    layout: GroupLayout,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
) -> float:
    """Hierarchical all-reduce over a group with a known physical layout."""
    tiers = _tiers_from_layout(layout, topology, scaleup_bw_gbps, nic_bw_gbps)
    if not tiers:
        return 0.0
    if len(tiers) == 1:
        t = tiers[0]
        return cc.ring_allreduce_time(size_bytes, t.group_size, t.bw_bytes_per_s, t.alpha_s)

    lowest = tiers[0]
    total = cc.ring_reduce_scatter_time(
        size_bytes, lowest.group_size, lowest.bw_bytes_per_s, lowest.alpha_s
    )
    chunk = size_bytes / lowest.group_size
    for tier in tiers[1:]:
        total += cc.ring_allreduce_time(chunk, tier.group_size, tier.bw_bytes_per_s, tier.alpha_s)
    total += cc.ring_allgather_time(
        size_bytes, lowest.group_size, lowest.bw_bytes_per_s, lowest.alpha_s
    )
    return total


def pipeline_crossing_fraction(
    layout: GroupLayout, topology: TopologySpec, pp: int
) -> float:
    """Fraction of a pipeline's ``pp - 1`` stage boundaries that cross a hall edge.

    This is the whole of the placement question for a pipeline, so it is named
    rather than inlined. Charging every boundary at the stitch rate
    unconditionally would manufacture the distance cliff this study is testing;
    charging one unconditionally would hide it. DECISIONS.md D17.
    """
    if layout.halls <= 1 or pp <= 1:
        return 0.0
    if topology.span_placement == "blind":
        # Round-robin interleaving puts stage i in hall i mod halls, so
        # consecutive stages are never co-resident and every boundary crosses.
        return 1.0
    # Contiguous stage blocks put exactly halls - 1 boundaries on an edge,
    # whatever the depth of the pipeline.
    return (layout.halls - 1) / (pp - 1)


def span_alpha_seconds(
    layout: GroupLayout, topology: TopologySpec, kind: str = "allreduce"
) -> float:
    """Latency charged at the span tier for one collective, in seconds.

    Split out because it is the one part of a collective's cost that cannot be
    hidden behind compute. The bandwidth term of a ring streams alongside the
    backward pass; the ``2(k-1)`` chained hops are serially dependent and drain
    at the step boundary. That step count is the textbook ring structure stated
    in :mod:`netcap.collectives`, not a claim from any source -- an earlier
    draft hung it on S12, which is a modelling survey and does not support it.
    Charging the stitch's latency at the same
    overlap fraction as its bandwidth term is what makes a naive model conclude
    that distance is free.

    Intra-hall tiers are deliberately excluded: their alphas are microseconds
    and the three-tier model's existing overlap treatment of them is unchanged,
    so every result published before the span work is reproduced exactly.
    """
    if layout.halls <= 1 or layout.span_ring <= 1:
        return 0.0
    hops = layout.span_ring - 1
    alpha = topology.alpha_span_us * 1e-6
    return (2.0 * hops * alpha) if kind == "allreduce" else (hops * alpha)


def allgather_on_layout(
    size_bytes: float,
    layout: GroupLayout,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
) -> float:
    tiers = _tiers_from_layout(layout, topology, scaleup_bw_gbps, nic_bw_gbps)
    total = 0.0
    sizes = []
    chunk = size_bytes
    for tier in tiers:
        sizes.append(chunk)
        chunk = chunk / tier.group_size
    for tier, sz in zip(reversed(tiers), reversed(sizes)):
        total += cc.ring_allgather_time(sz, tier.group_size, tier.bw_bytes_per_s, tier.alpha_s)
    return total


def model_flops_per_token(model: ModelSpec, causal_factor: float = 0.5) -> float:
    """Nominal training FLOPs per token, the denominator convention used by MFU.

    Two terms: ``6 * n_params`` for forward and backward through the weights,
    and an attention-score term that scales with sequence length. The causal
    factor halves the attention term because only the lower triangle is computed.
    """
    dense = 6.0 * model.n_params
    attn = 12.0 * model.n_layers * model.seq_len * model.hidden * causal_factor
    return dense + attn


def shape_derived_params(model: ModelSpec) -> float:
    """Parameter count implied by the declared shape, for a consistency check."""
    per_layer = (4.0 + 2.0 * model.ffn_multiplier) * model.hidden**2
    return model.n_layers * per_layer + 2.0 * model.vocab * model.hidden


def check_param_consistency(model: ModelSpec) -> float:
    """Relative disagreement between declared and shape-derived parameter counts."""
    derived = shape_derived_params(model)
    return abs(derived - model.n_params) / model.n_params


@dataclass(frozen=True)
class StepBreakdown:
    """Per-iteration timing of a healthy synchronous step, in seconds."""

    t_compute: float
    t_tp_exposed: float
    t_cp_exposed: float
    t_pp_exposed: float
    t_dp_exposed: float
    t_bubble: float
    t_sync_wait: float
    t_step: float
    tokens_per_step: float
    comm_raw: Dict[str, float]

    @property
    def t_blocked(self) -> float:
        return (
            self.t_tp_exposed
            + self.t_cp_exposed
            + self.t_pp_exposed
            + self.t_dp_exposed
            + self.t_bubble
            + self.t_sync_wait
        )

    @property
    def step_efficiency(self) -> float:
        """Fraction of a healthy step spent on productive compute."""
        return self.t_compute / self.t_step if self.t_step > 0 else 0.0


def synchronization_tax(world_size: int, cv: float) -> float:
    """Expected slowdown of a synchronous step from per-rank timing jitter.

    A synchronous step waits for the slowest of ``N`` ranks. For per-rank times
    with coefficient of variation ``cv``, the expectation of the maximum of ``N``
    draws grows approximately as ``1 + cv * sqrt(2 ln N)`` (the standard Gaussian
    extreme-value approximation). This makes jitter more expensive at scale,
    which matches reported straggler behavior, but it is an approximation and is
    swept in sensitivity analysis.
    """
    if cv <= 0 or world_size <= 1:
        return 1.0
    return 1.0 + cv * math.sqrt(2.0 * math.log(world_size))


def failslow_tax(n_nodes: int, prob_per_node: float, slowdown: float) -> float:
    """Expected slowdown from at least one degraded but still running node."""
    if prob_per_node <= 0 or slowdown <= 1.0 or n_nodes < 1:
        return 1.0
    p_any = 1.0 - (1.0 - prob_per_node) ** n_nodes
    return 1.0 + (slowdown - 1.0) * p_any


def step_breakdown(scenario: ScenarioConfig) -> StepBreakdown:
    """Time one healthy training iteration and split it into productive and blocked."""
    par = scenario.parallelism
    model = scenario.model
    topo = scenario.topology
    acc = scenario.accelerator

    world = par.world_size
    if world <= 0:
        raise ValueError("world size must be positive")

    seqs_per_dp = par.global_batch_seqs / par.dp
    micro_per_iter = max(1.0, seqs_per_dp / par.micro_batch)
    tokens_per_step = par.global_batch_seqs * model.seq_len

    # Compute time. The recompute factor inflates executed FLOPs above nominal.
    nominal_flops = model_flops_per_token(model) * tokens_per_step
    executed_flops = nominal_flops * RECOMPUTE_FACTORS.get(par.recompute, 1.0)
    peak_total = world * acc.peak_flops_bf16 * scenario.kernel.kernel_efficiency
    t_compute = executed_flops / peak_total

    # Rank layout: tensor parallel is innermost, then context, pipeline, data.
    stride_tp = 1
    stride_cp = par.tp
    stride_pp = par.tp * par.cp
    stride_dp = par.tp * par.cp * par.pp

    lay_tp = group_layout(par.tp, stride_tp, topo, dimension="tp", world=world)
    lay_cp = group_layout(par.cp, stride_cp, topo, dimension="cp", world=world)
    lay_dp = group_layout(par.dp, stride_dp, topo, dimension="dp", world=world)

    layers_per_stage = model.n_layers / par.pp
    seq_local = model.seq_len / par.cp
    act_tensor = par.micro_batch * seq_local * model.hidden * par.act_bytes

    # Tensor parallel: two collectives per layer forward, two backward.
    t_tp = 0.0
    a_tp = 0.0
    if par.tp > 1:
        one = allreduce_on_layout(act_tensor, lay_tp, topo, acc.scaleup_bw_gbps, acc.nic_bw_gbps)
        reps = 4.0 * layers_per_stage * micro_per_iter
        t_tp = reps * one
        a_tp = reps * span_alpha_seconds(lay_tp, topo, "allreduce")

    # Context parallel: ring exchange of key/value chunks, forward and backward.
    t_cp = 0.0
    a_cp = 0.0
    if par.cp > 1:
        kv_ratio = (
            model.n_kv_heads / model.n_heads if model.n_heads and model.n_kv_heads else 1.0
        )
        kv_tensor = 2.0 * par.micro_batch * seq_local * model.hidden * kv_ratio * par.act_bytes
        one = allgather_on_layout(kv_tensor, lay_cp, topo, acc.scaleup_bw_gbps, acc.nic_bw_gbps)
        reps = 3.0 * layers_per_stage * micro_per_iter
        t_cp = reps * one
        a_cp = reps * span_alpha_seconds(lay_cp, topo, "allgather")

    # Pipeline parallel: activation handoff at stage boundaries. A synchronous
    # pipeline is paced by its slowest inter-stage link, so the handoff is
    # costed at the worst tier any boundary crosses; enlarging the scale-up
    # domain therefore does not relieve pp unless the whole pipeline fits in
    # one domain. Consequently the scale-up intervention prices only what a
    # bigger domain does for this fixed plan, not the replanning it enables.
    t_pp = 0.0
    a_pp = 0.0
    if par.pp > 1:
        lay_pp = group_layout(par.pp, stride_pp, topo, dimension="pp", world=world)
        crosses = "crosspod" if lay_pp.pods > 1 else "pod"
        one = cc.point_to_point_time(
            act_tensor, topo, acc.scaleup_bw_gbps, acc.nic_bw_gbps, crosses=crosses
        )
        if lay_pp.halls > 1:
            share = span_share_bytes_per_s(topo, lay_pp.span_contention, acc.nic_bw_gbps)
            spanned = (
                act_tensor / share + topo.alpha_span_us * 1e-6 if share > 0 else math.inf
            )
            # Unlike the cross-pod case above, a spanned pipeline is *not* costed
            # at its worst boundary for every handoff; see the function below.
            frac = pipeline_crossing_fraction(lay_pp, topo, par.pp)
            one = (1.0 - frac) * one + frac * spanned
            a_pp = 2.0 * micro_per_iter * frac * topo.alpha_span_us * 1e-6
        t_pp = 2.0 * micro_per_iter * one

    # Data parallel: gradient reduction over the sharded parameter set. With
    # context parallelism the reduction would also span the cp group; no shipped
    # configuration uses cp > 1, and the omission is recorded in ASSUMPTIONS.md.
    t_dp = 0.0
    a_dp = 0.0
    if par.dp > 1:
        params_local = model.n_params / (par.tp * par.pp)
        grad_bytes = params_local * par.grad_bytes
        t_dp = allreduce_on_layout(grad_bytes, lay_dp, topo, acc.scaleup_bw_gbps, acc.nic_bw_gbps)
        a_dp = span_alpha_seconds(lay_dp, topo, "allreduce")
        if par.zero_stage >= 3:
            t_dp += allgather_on_layout(
                params_local * par.act_bytes, lay_dp, topo, acc.scaleup_bw_gbps, acc.nic_bw_gbps
            )
            a_dp += span_alpha_seconds(lay_dp, topo, "allgather")

    # Stitch latency is serially chained and cannot hide behind compute, so it
    # is exposed in full while only the rest of the collective is overlapped.
    # All four terms are exactly zero when the job is hall-local. DECISIONS.md D16.
    def _expose(total: float, span_alpha: float, overlap: float) -> float:
        span_alpha = min(span_alpha, total)
        return (total - span_alpha) * (1.0 - overlap) + span_alpha

    t_tp_exp = _expose(t_tp, a_tp, par.overlap_tp)
    t_cp_exp = _expose(t_cp, a_cp, par.overlap_cp)
    t_pp_exp = _expose(t_pp, a_pp, par.overlap_pp)
    t_dp_exp = _expose(t_dp, a_dp, par.overlap_dp)

    # 1F1B pipeline bubble, reduced by interleaving (virtual stages per device).
    t_bubble = (
        t_compute * (par.pp - 1) / (micro_per_iter * max(1, par.interleave_factor))
        if par.pp > 1
        else 0.0
    )

    base = t_compute + t_tp_exp + t_cp_exp + t_pp_exp + t_dp_exp + t_bubble
    n_nodes = max(1, world // max(1, acc.accelerators_per_node))
    tax = synchronization_tax(world, scenario.reliability.straggler_cv) * failslow_tax(
        n_nodes,
        scenario.reliability.failslow_prob_per_node,
        scenario.reliability.failslow_slowdown,
    )
    t_sync_wait = base * (tax - 1.0)

    return StepBreakdown(
        t_compute=t_compute,
        t_tp_exposed=t_tp_exp,
        t_cp_exposed=t_cp_exp,
        t_pp_exposed=t_pp_exp,
        t_dp_exposed=t_dp_exp,
        t_bubble=t_bubble,
        t_sync_wait=t_sync_wait,
        t_step=base + t_sync_wait,
        tokens_per_step=tokens_per_step,
        comm_raw={"tp": t_tp, "cp": t_cp, "pp": t_pp, "dp": t_dp},
    )


def model_flops_utilization(scenario: ScenarioConfig, step: StepBreakdown) -> float:
    """MFU as conventionally reported: nominal model FLOPs over peak, per step."""
    nominal = model_flops_per_token(scenario.model) * step.tokens_per_step
    peak = scenario.parallelism.world_size * scenario.accelerator.peak_flops_bf16
    return nominal / (peak * step.t_step) if step.t_step > 0 else 0.0


def memory_per_accelerator_gb(scenario: ScenarioConfig) -> Tuple[float, float]:
    """Rough weights-plus-optimizer and activation footprint, for feasibility checks.

    Used only to flag configurations that could not physically run. It is not a
    detailed memory model and is not used in any headline result.
    """
    par = scenario.parallelism
    model = scenario.model
    shard = model.n_params / (par.tp * par.pp)
    state_bytes = shard * (2 + 4 + 4 + 4)  # bf16 weights, fp32 master, two moments
    if par.zero_stage >= 1:
        state_bytes = shard * 2 + shard * 12 / max(1, par.dp)
    layers_per_stage = model.n_layers / par.pp
    act = (
        par.micro_batch
        * (model.seq_len / par.cp)
        * model.hidden
        * layers_per_stage
        * par.act_bytes
        * 4.0
        / par.tp
    )
    return state_bytes / 1e9, act / 1e9
