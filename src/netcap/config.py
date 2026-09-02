"""Configuration schema for the capacity-accounting model.

All experiment parameters live in dataclasses that round-trip to YAML. No
experiment parameter is hard-coded anywhere else in the package.

Units are stated explicitly in field names. Bandwidth fields are per-accelerator
and expressed in Gbit/s (line rate, before the efficiency derate in
``TopologySpec.net_efficiency``). Time fields are seconds unless suffixed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

GIGA = 1e9
TERA = 1e12
SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class ModelSpec:
    """Dense decoder-only transformer shape.

    ``n_params`` is the total parameter count including embeddings. It is kept
    as an independent field rather than derived so that published model cards
    can be entered verbatim; :func:`netcap.performance.check_param_consistency`
    reports the disagreement with the shape-derived estimate.
    """

    name: str
    n_layers: int
    hidden: int
    n_params: float
    seq_len: int
    vocab: int
    n_heads: int = 0
    n_kv_heads: int = 0
    ffn_multiplier: float = 4.0


@dataclass(frozen=True)
class AcceleratorSpec:
    """Per-accelerator compute and injection bandwidth."""

    name: str
    peak_flops_bf16: float  # dense BF16 FLOP/s, no sparsity
    memory_gb: float
    scaleup_bw_gbps: float  # per-accelerator, unidirectional, into scale-up domain
    nic_bw_gbps: float  # per-accelerator, unidirectional, into scale-out fabric
    accelerators_per_node: int = 8


@dataclass(frozen=True)
class TopologySpec:
    """Four-tier hierarchy: scale-up domain, full-bisection pod, cross-pod, span.

    ``oversubscription`` is the cross-pod ratio expressed as a number >= 1, so a
    1:7 oversubscribed spine layer is ``7.0``. ``net_efficiency`` is the fraction
    of line rate a well-tuned collective library achieves; it absorbs protocol
    overhead, imperfect ring formation, and moderate congestion.

    The **span** tier is the inter-hall stitch: a circuit joining two latency
    domains in different halls or buildings. It differs from the three fabric
    tiers in one way that drives every result in :mod:`netcap.regimes`, and the
    difference is structural rather than a matter of degree:

    * The three fabric tiers are described by a **per-accelerator** bandwidth.
      Buying a wider pod buys every rank more bandwidth.
    * ``span_bw_gbps`` is an **aggregate circuit capacity shared by every group
      that crosses it**. Its per-rank share falls as more groups cross, so the
      cost of a stitch depends on *which* parallel dimension is cut across it,
      not only on how far away the other hall is.

    ``halls`` of 1 (the default) disables the span tier entirely and reproduces
    the three-tier model exactly; every scenario shipped in ``configs/`` before
    the span work leaves it at the default.

    ``span_dimension`` names the cut placed across the stitch --- the "named
    cut" an operator chooses when a job is allowed to span. ``"none"`` keeps the
    job hall-local. Only the named dimension crosses; every other dimension is
    laid out inside one hall.

    ``span_placement`` is how the crossing group's ranks are ordered, and it is
    the difference between a stitch that works and one that does not:

    * ``"hierarchical"`` --- contiguous placement. The group reduces inside each
      hall first and the ring at the span tier has one member per hall, so the
      stitch latency is paid ``2(halls - 1)`` times per collective.
    * ``"blind"`` --- a topology-blind ring that interleaves ranks across halls,
      so the ring crosses the stitch on almost every hop and the latency is paid
      ``2(size - 1)`` times. This is not a strawman: it is what a collective
      library does when nothing tells it where the hall boundary is, and it is
      the configuration in which published geo-distributed training studies
      report their distance cliffs. Reproducing that cliff and then removing it
      by changing only the placement is the point of ``docs/regime-atlas.md``.
    """

    scaleup_domain: int  # accelerators sharing the scale-up fabric
    pod_size: int  # accelerators under full bisection bandwidth
    oversubscription: float = 1.0
    alpha_scaleup_us: float = 2.0
    alpha_pod_us: float = 6.0
    alpha_xpod_us: float = 15.0
    net_efficiency: float = 0.85
    halls: int = 1  # latency domains the pool is split across
    alpha_span_us: float = 0.0  # one-way latency of the inter-hall stitch
    span_bw_gbps: float = 0.0  # aggregate circuit capacity, shared by all crossers
    span_dimension: str = "none"  # none | tp | cp | pp | dp
    span_placement: str = "hierarchical"  # hierarchical | blind

    def __post_init__(self) -> None:
        if self.span_dimension not in ("none", "tp", "cp", "pp", "dp"):
            raise ValueError(f"unknown span_dimension {self.span_dimension!r}")
        if self.span_placement not in ("hierarchical", "blind"):
            raise ValueError(f"unknown span_placement {self.span_placement!r}")
        if self.halls < 1:
            raise ValueError("halls must be >= 1")
        if self.halls > 1 and self.span_dimension == "none":
            raise ValueError("halls > 1 requires a named span_dimension")
        if self.spans and self.span_bw_gbps <= 0.0:
            raise ValueError("a span_dimension requires span_bw_gbps > 0")

    @property
    def spans(self) -> bool:
        """True when a cut is actually placed across a stitch."""
        return self.halls > 1 and self.span_dimension != "none"


@dataclass(frozen=True)
class ParallelismSpec:
    """Parallelism degrees and overlap assumptions.

    Overlap fields are the fraction of that collective's time hidden behind
    compute. They are assumptions, swept in sensitivity analysis.
    """

    tp: int = 8
    pp: int = 1
    cp: int = 1
    dp: int = 1
    micro_batch: int = 1
    global_batch_seqs: int = 2048
    zero_stage: int = 1
    interleave_factor: int = 1  # virtual pipeline stages per device
    # Descriptive only: the tensor-parallel collective volumes above already
    # assume a sequence-parallel layout, so this flag does not change the model.
    sequence_parallel: bool = True
    recompute: str = "selective"  # none | selective | full
    overlap_dp: float = 0.85
    overlap_tp: float = 0.10
    overlap_pp: float = 0.70
    overlap_cp: float = 0.30
    grad_bytes: int = 4  # gradient/optimizer reduction precision in bytes
    act_bytes: int = 2  # activation precision in bytes

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.cp * self.dp


@dataclass(frozen=True)
class ReliabilitySpec:
    """Failure, detection, recovery, checkpoint, and degradation parameters.

    ``failure_rate_per_node_day`` is the job-stopping failure rate per compute
    node per day of runtime. The default is calibrated to Meta HPCA 2025
    (approximately 5e-3 per GPU-node-day); see ``SOURCES.md`` S2.

    ``checkpoint_interval_s`` of ``None`` selects the Daly first-order optimum.
    """

    failure_rate_per_node_day: float = 5.0e-3
    detect_time_s: float = 120.0
    restart_time_s: float = 300.0
    checkpoint_interval_s: Optional[float] = None
    checkpoint_write_s: float = 60.0
    checkpoint_blocking_fraction: float = 1.0  # 1.0 fully blocking, ~0.05 async
    repair_time_s: float = 3600.0
    spare_fraction: float = 0.0
    straggler_cv: float = 0.0
    failslow_prob_per_node: float = 0.0
    failslow_slowdown: float = 1.0
    stranded_fraction: float = 0.0


@dataclass(frozen=True)
class KernelSpec:
    """Compute-side efficiency, held constant across network interventions.

    Separating this from the network and reliability terms is what lets the
    model report a network-and-reliability capacity fraction that is not
    contaminated by kernel quality. ``kernel_efficiency`` is the fraction of
    peak FLOP/s achieved by the compute kernels themselves when not waiting on
    anything.
    """

    kernel_efficiency: float = 0.72


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete, self-describing experiment point."""

    name: str
    model: ModelSpec
    accelerator: AcceleratorSpec
    topology: TopologySpec
    parallelism: ParallelismSpec
    reliability: ReliabilitySpec
    kernel: KernelSpec = field(default_factory=KernelSpec)
    n_pool: int = 0  # accelerators paid for; 0 means "equal to world size"
    window_days: float = 30.0
    scaling_mode: str = "strong"  # strong | weak
    max_global_batch_seqs: int = 8192
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_pool == 0:
            # Accelerators paid for: the job's world size plus the spare and
            # stranded overhead that the operator also owns but cannot compute on.
            overhead = 1.0 + self.reliability.spare_fraction + self.reliability.stranded_fraction
            object.__setattr__(
                self, "n_pool", int(math.ceil(self.parallelism.world_size * overhead))
            )

    @property
    def window_seconds(self) -> float:
        return self.window_days * SECONDS_PER_DAY


_SECTION_TYPES = {
    "model": ModelSpec,
    "accelerator": AcceleratorSpec,
    "topology": TopologySpec,
    "parallelism": ParallelismSpec,
    "reliability": ReliabilitySpec,
    "kernel": KernelSpec,
}


def _coerce(section_type: type, values: Dict[str, Any]) -> Dict[str, Any]:
    """Cast YAML-parsed values to the declared field types.

    PyYAML follows YAML 1.1, which does not recognize ``4.05e11`` as a float
    because the exponent lacks a sign. Rather than requiring configs to be
    written defensively, values are coerced against the dataclass annotations.
    """
    hints = {f.name: f.type for f in dataclasses.fields(section_type)}
    out: Dict[str, Any] = {}
    for key, value in values.items():
        target = hints.get(key)
        if isinstance(value, str) and target is not None and "float" in str(target):
            out[key] = float(value)
        elif isinstance(value, str) and target in ("int", int):
            out[key] = int(float(value))
        elif target in ("float", float) and isinstance(value, int):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def scenario_from_dict(data: Dict[str, Any]) -> ScenarioConfig:
    """Build a :class:`ScenarioConfig` from a plain dict (typically parsed YAML)."""
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTION_TYPES:
            section_type = _SECTION_TYPES[key]
            kwargs[key] = section_type(**_coerce(section_type, value))
        elif key == "window_days" and isinstance(value, (int, str)):
            kwargs[key] = float(value)
        else:
            kwargs[key] = value
    return ScenarioConfig(**kwargs)


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load a scenario from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        return scenario_from_dict(yaml.safe_load(handle))


def save_scenario(scenario: ScenarioConfig, path: str | Path) -> None:
    """Write a scenario to YAML."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(scenario), handle, sort_keys=False)


def replace_nested(scenario: ScenarioConfig, **overrides: Any) -> ScenarioConfig:
    """Return a copy of ``scenario`` with dotted-path overrides applied.

    Example: ``replace_nested(cfg, **{"topology.oversubscription": 1.0})``.
    """
    sections: Dict[str, Dict[str, Any]] = {}
    top: Dict[str, Any] = {}
    for key, value in overrides.items():
        if "." in key:
            section, attr = key.split(".", 1)
            sections.setdefault(section, {})[attr] = value
        else:
            top[key] = value

    new_kwargs: Dict[str, Any] = {}
    for section, attrs in sections.items():
        current = getattr(scenario, section)
        new_kwargs[section] = dataclasses.replace(current, **attrs)
    new_kwargs.update(top)
    return dataclasses.replace(scenario, **new_kwargs)
