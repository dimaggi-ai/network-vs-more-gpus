"""Hierarchical collective communication timing.

The model is the standard alpha-beta (Hockney) cost model applied tier by tier
across a three-level hierarchy: a scale-up domain, a full-bisection pod, and an
oversubscribed cross-pod layer.

This module holds the per-tier ring primitives and the point-to-point cost;
:mod:`netcap.performance` composes them into hierarchical collectives using the
physical layout of each parallel group. A ring all-reduce of ``S`` bytes over
``k`` ranks moves ``2(k-1)/k * S`` bytes per rank in ``2(k-1)`` steps.

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


