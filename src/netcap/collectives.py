"""Hierarchical collective communication timing.

The model is the standard alpha-beta (Hockney) cost model applied tier by tier
across a three-level hierarchy: a scale-up domain, a full-bisection pod, and an
oversubscribed cross-pod layer.

Collectives are decomposed the way production libraries implement them. A
hierarchical all-reduce over ``g`` ranks becomes a reduce-scatter inside the
lowest tier, an all-reduce of the reduced chunk across higher tiers, and an
all-gather back inside the lowest tier. Ring costs are used at every tier:
a ring all-reduce of ``S`` bytes over ``k`` ranks moves ``2(k-1)/k * S`` bytes
per rank in ``2(k-1)`` steps.

Nothing here is novel. It is deliberately a textbook formulation so that the
predictions can be checked against independent tools; see
``validation/VALIDATION_REPORT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import TopologySpec

BITS_PER_BYTE = 8.0
_US = 1e-6


@dataclass(frozen=True)
class Tier:
    """One level of the network hierarchy as seen by a collective."""

    name: str
    group_size: int  # number of participants at this tier
    bw_bytes_per_s: float  # per-rank achievable unidirectional bandwidth
    alpha_s: float  # per-hop latency


def _bw_bytes(gbps: float, efficiency: float) -> float:
    return gbps * 1e9 * efficiency / BITS_PER_BYTE


def decompose_group(
    group_size: int,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
) -> List[Tier]:
    """Split a collective group of ``group_size`` ranks into hierarchy tiers.

    Ranks are assumed to be laid out densely: consecutive ranks fill a scale-up
    domain, consecutive scale-up domains fill a pod, then pods are consumed.
    This is the placement a rail-optimized cluster aims for and is the
    best case for the network; ``ASSUMPTIONS.md`` A6 records the consequence.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    tiers: List[Tier] = []
    eff = topology.net_efficiency

    su = max(1, min(group_size, topology.scaleup_domain))
    if su > 1:
        tiers.append(
            Tier(
                "scaleup",
                su,
                _bw_bytes(scaleup_bw_gbps, eff),
                topology.alpha_scaleup_us * _US,
            )
        )

    # Groups of scale-up domains that still fit inside one full-bisection pod.
    domains_total = _ceil_div(group_size, su)
    domains_per_pod = max(1, topology.pod_size // max(1, topology.scaleup_domain))
    within_pod = max(1, min(domains_total, domains_per_pod))
    if within_pod > 1:
        tiers.append(
            Tier("pod", within_pod, _bw_bytes(nic_bw_gbps, eff), topology.alpha_pod_us * _US)
        )

    pods = _ceil_div(domains_total, within_pod)
    if pods > 1:
        tiers.append(
            Tier(
                "crosspod",
                pods,
                _bw_bytes(nic_bw_gbps / max(1.0, topology.oversubscription), eff),
                topology.alpha_xpod_us * _US,
            )
        )
    return tiers


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def ring_allreduce_time(size_bytes: float, k: int, bw: float, alpha: float) -> float:
    """Ring all-reduce of ``size_bytes`` over ``k`` ranks at one tier."""
    if k <= 1:
        return 0.0
    return 2.0 * (k - 1) / k * size_bytes / bw + 2.0 * (k - 1) * alpha


def ring_reduce_scatter_time(size_bytes: float, k: int, bw: float, alpha: float) -> float:
    if k <= 1:
        return 0.0
    return (k - 1) / k * size_bytes / bw + (k - 1) * alpha


def ring_allgather_time(size_bytes: float, k: int, bw: float, alpha: float) -> float:
    if k <= 1:
        return 0.0
    return (k - 1) / k * size_bytes / bw + (k - 1) * alpha


def allreduce_time(
    size_bytes: float,
    group_size: int,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
) -> float:
    """Hierarchical all-reduce time for ``size_bytes`` per rank.

    Lowest tier does reduce-scatter then all-gather; the tiers above operate on
    the progressively smaller reduced chunk.
    """
    tiers = decompose_group(group_size, topology, scaleup_bw_gbps, nic_bw_gbps)
    if not tiers:
        return 0.0
    if len(tiers) == 1:
        t = tiers[0]
        return ring_allreduce_time(size_bytes, t.group_size, t.bw_bytes_per_s, t.alpha_s)

    lowest = tiers[0]
    total = ring_reduce_scatter_time(
        size_bytes, lowest.group_size, lowest.bw_bytes_per_s, lowest.alpha_s
    )
    chunk = size_bytes / lowest.group_size
    for tier in tiers[1:]:
        # Every higher tier all-reduces the same reduced chunk. With more than
        # two tiers this is conservative (a further reduce-scatter would shrink
        # the chunk again); no configuration in this study exercises three
        # tiers in one collective group.
        total += ring_allreduce_time(chunk, tier.group_size, tier.bw_bytes_per_s, tier.alpha_s)
    total += ring_allgather_time(
        size_bytes, lowest.group_size, lowest.bw_bytes_per_s, lowest.alpha_s
    )
    return total


def reduce_scatter_time(
    size_bytes: float,
    group_size: int,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
) -> float:
    """Hierarchical reduce-scatter, costed as half of the all-reduce path."""
    tiers = decompose_group(group_size, topology, scaleup_bw_gbps, nic_bw_gbps)
    total = 0.0
    chunk = size_bytes
    for tier in tiers:
        total += ring_reduce_scatter_time(chunk, tier.group_size, tier.bw_bytes_per_s, tier.alpha_s)
        chunk = chunk / tier.group_size
    return total


def allgather_time(
    size_bytes: float,
    group_size: int,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
) -> float:
    """Hierarchical all-gather producing ``size_bytes`` per rank."""
    tiers = decompose_group(group_size, topology, scaleup_bw_gbps, nic_bw_gbps)
    total = 0.0
    sizes = []
    chunk = size_bytes
    for tier in tiers:
        sizes.append(chunk)
        chunk = chunk / tier.group_size
    for tier, sz in zip(reversed(tiers), reversed(sizes)):
        total += ring_allgather_time(sz, tier.group_size, tier.bw_bytes_per_s, tier.alpha_s)
    return total


def point_to_point_time(
    size_bytes: float,
    topology: TopologySpec,
    scaleup_bw_gbps: float,
    nic_bw_gbps: float,
    crosses: str = "pod",
) -> float:
    """Single send/recv, used for pipeline-stage activation handoff."""
    eff = topology.net_efficiency
    if crosses == "scaleup":
        return size_bytes / _bw_bytes(scaleup_bw_gbps, eff) + topology.alpha_scaleup_us * _US
    if crosses == "crosspod":
        bw = _bw_bytes(nic_bw_gbps / max(1.0, topology.oversubscription), eff)
        return size_bytes / bw + topology.alpha_xpod_us * _US
    return size_bytes / _bw_bytes(nic_bw_gbps, eff) + topology.alpha_pod_us * _US


def bus_bandwidth_gbps(size_bytes: float, seconds: float, group_size: int) -> float:
    """Algorithmic bus bandwidth, the quantity NCCL benchmarks report.

    Provided so model output can be compared directly against published
    ``nccl-tests`` numbers.
    """
    if seconds <= 0 or group_size <= 1:
        return 0.0
    algo_bw = size_bytes / seconds
    return algo_bw * 2.0 * (group_size - 1) / group_size * BITS_PER_BYTE / 1e9
