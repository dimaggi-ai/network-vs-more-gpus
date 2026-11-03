"""Offline declaration checks before using accelerator evidence in netcap.

This module does not fetch sources, run a model, authenticate evidence, or
qualify a device. Passing means only that supplied declarations fit the narrow
model interface. Callers must independently establish their truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
import re

UNMODELED_DIMENSIONS = ("host", "SRAM", "topology_calibration", "power", "price")
LIMITATIONS = (
    "Declaration matching is not measured comparability or calibration.",
    "Source locations, dates, revisions and hashes are supplied, not authenticated or fetched.",
    "Retrieval date is not a measurement date or an automatic freshness guarantee.",
    "No execution permission, full-stack support or production readiness is established.",
    "Host, SRAM, topology calibration, power and price remain outside this gate.",
)


@dataclass(frozen=True)
class Provenance:
    source: str
    revision: str
    retrieved_on: str  # ISO YYYY-MM-DD; no implicit current clock
    sha256: str  # supplied source digest; never a signature
    evidence_class: str  # synthetic | vendor_specification | measured
    valid_until: str | None = None  # optional explicit caller-supplied expiry


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    scope: str  # per_accelerator only
    direction: str  # unidirectional bandwidth; not_applicable compute/memory


@dataclass(frozen=True)
class ComparisonContext:
    workload: str
    quality_target: str
    protocol: str
    model_family: str  # dense_decoder_only
    topology_model: str  # netcap_four_tier; not a hardware topology calibration
    parallelism: tuple[int, int, int, int]  # tp, pp, cp, dp
    window_seconds: float


@dataclass(frozen=True)
class CapabilityEvidence:
    hardware: str
    precision: str
    sparsity: str
    compute: Quantity | None
    memory: Quantity | None
    scaleup: Quantity | None
    scaleout: Quantity | None
    context: ComparisonContext
    provenance: Provenance | None
    scaleup_topology: str | None = None  # explicit flat_switched declaration required
    # These dimensions are retained as explicit unknowns, never substituted by 0.
    unknown_dimensions: tuple[str, ...] = UNMODELED_DIMENSIONS


@dataclass(frozen=True)
class ModelInputs:
    peak_flops_bf16: float
    memory_gb: float
    scaleup_bw_gbps: float
    nic_bw_gbps: float


@dataclass(frozen=True)
class Assessment:
    status: str  # applicable_to_declared_model | incomplete | unsupported | invalid
    reasons: tuple[str, ...]
    inputs: ModelInputs | None
    record_sha256: str | None
    unknown_dimensions: tuple[str, ...]
    limitations: tuple[str, ...] = LIMITATIONS


def _text(value: object) -> bool:
    return (type(value) is str and 0 < len(value) <= 2048
            and value == value.strip() and all(ord(c) >= 32 and ord(c) != 127 for c in value))


def _positive(value: object) -> bool:
    # Bound integers before float conversion (huge Python ints can overflow).
    return type(value) in (int, float) and 0 < value <= 1e100 and math.isfinite(value)


def _iso(value: object) -> date | None:
    if type(value) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _context_valid(ctx: object) -> bool:
    return (type(ctx) is ComparisonContext
            and all(_text(getattr(ctx, f)) for f in
                    ("workload", "quality_target", "protocol", "model_family", "topology_model"))
            and type(ctx.parallelism) is tuple and len(ctx.parallelism) == 4
            and all(type(n) is int and 0 < n <= 1_000_000 for n in ctx.parallelism)
            and _positive(ctx.window_seconds))


def assess(evidence: CapabilityEvidence, expected: ComparisonContext, *,
           as_of: str) -> Assessment:
    """Assess declarations against a supplied context and explicit ISO date.

    No JSON ingestion or implicit coercion. Missing required evidence refuses
    applicability. Expiry is checked only when explicitly declared; the source
    must still be reviewed independently. No rounding, aggregate division,
    bidirectional halving, precision conversion or sparsity conversion occurs.
    """
    invalid, missing, unsupported = [], [], []
    result = lambda status, reasons, inputs=None, digest=None: Assessment(
        status, tuple(reasons), inputs, digest, UNMODELED_DIMENSIONS)
    if type(evidence) is not CapabilityEvidence:
        return result("invalid", ["evidence must be CapabilityEvidence"])
    when = _iso(as_of)
    if when is None:
        invalid.append("as_of must be an explicit ISO calendar date")
    if not _context_valid(expected) or not _context_valid(evidence.context):
        invalid.append("contexts require named workload/quality/protocol and finite positive scale/window")
    elif evidence.context != expected:
        unsupported.append("workload, quality, protocol, model, topology abstraction, scale and window must match")
    if _context_valid(evidence.context):
        if evidence.context.model_family != "dense_decoder_only":
            unsupported.append("only dense_decoder_only is represented")
        if evidence.context.topology_model != "netcap_four_tier":
            unsupported.append("only the netcap_four_tier abstraction is represented; no topology calibration inferred")
    if evidence.scaleup_topology is None or evidence.scaleup_topology == "unknown":
        missing.append("scaleup_topology must explicitly declare flat_switched; domain size alone is insufficient")
    elif not _text(evidence.scaleup_topology):
        invalid.append("scaleup_topology must be bounded text or None")
    elif evidence.scaleup_topology != "flat_switched":
        unsupported.append("only the declared flat_switched scale-up abstraction fits; no torus, Boardfly, dragonfly, hop or bisection model is implemented")
    if not all(_text(v) for v in (evidence.hardware, evidence.precision, evidence.sparsity)):
        invalid.append("hardware, precision and sparsity must be nonempty bounded text")
    if evidence.precision != "BF16":
        unsupported.append("dense BF16 throughput required; no conversion from FP4 or other precision")
    if evidence.sparsity != "dense":
        unsupported.append("dense throughput required; no sparse-to-dense conversion")
    if evidence.unknown_dimensions != UNMODELED_DIMENSIONS or type(evidence.unknown_dimensions) is not tuple:
        invalid.append("retain host, SRAM, topology_calibration, power and price as explicit unknown dimensions")
    p = evidence.provenance
    if p is None:
        missing.append("provenance is required")
    elif type(p) is not Provenance:
        invalid.append("provenance must be Provenance")
    else:
        if not all(_text(v) for v in (p.source, p.revision)):
            missing.append("source and revision must be explicit nonempty bounded text")
        retrieved = _iso(p.retrieved_on)
        if retrieved is None or (when is not None and retrieved > when):
            invalid.append("retrieved_on must be a valid ISO date no later than as_of")
        if type(p.sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", p.sha256):
            missing.append("supply the lowercase 64-hex source SHA-256")
        if p.evidence_class not in ("synthetic", "vendor_specification", "measured"):
            unsupported.append("evidence_class must explicitly distinguish synthetic, vendor_specification or measured")
        if p.valid_until is not None:
            expiry = _iso(p.valid_until)
            if expiry is None or (retrieved is not None and expiry < retrieved):
                invalid.append("valid_until must be a valid ISO date on/after retrieval")
            elif when is not None and expiry < when:
                unsupported.append("declared evidence expiry has passed; refresh and independently review the source")
    conversions = (
        ("compute", {"FLOP/s": 1., "TFLOP/s": 1e12, "PFLOP/s": 1e15}, "not_applicable"),
        ("memory", {"GB": 1., "GiB": 2**30 / 1e9}, "not_applicable"),
        ("scaleup", {"Gbit/s": 1., "Tbit/s": 1e3, "GB/s": 8., "TB/s": 8e3}, "unidirectional"),
        ("scaleout", {"Gbit/s": 1., "Tbit/s": 1e3, "GB/s": 8., "TB/s": 8e3}, "unidirectional"),
    )
    normalized = []
    for name, units, direction in conversions:
        q = getattr(evidence, name)
        if q is None:
            missing.append(f"{name} evidence is required")
            continue
        if type(q) is not Quantity or not _positive(q.value):
            invalid.append(f"{name} requires a finite positive numeric quantity (not bool)")
            continue
        if not all(_text(v) for v in (q.unit, q.scope, q.direction)):
            invalid.append(f"{name} requires explicit bounded unit, scope and direction")
            continue
        if q.unit not in units or q.scope != "per_accelerator" or q.direction != direction:
            unsupported.append(f"{name}: require units {tuple(units)}, per_accelerator, {direction}; no inferred conversion of scope/direction")
            continue
        converted = q.value * units[q.unit]
        if not math.isfinite(converted) or converted <= 0:
            invalid.append(f"{name} normalization must remain finite and positive")
            continue
        normalized.append(converted)
    if invalid:
        return result("invalid", invalid + missing + unsupported)
    if missing:
        return result("incomplete", missing + unsupported)
    if unsupported:
        return result("unsupported", unsupported)
    payload = {"gate": "netcap-declaration/v1", "evidence": asdict(evidence),
               "expected": asdict(expected), "as_of": as_of}
    record_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    return result("applicable_to_declared_model", [], ModelInputs(*normalized), record_hash)


def assess_pair(left: CapabilityEvidence, right: CapabilityEvidence,
                expected: ComparisonContext, *, as_of: str) -> tuple[Assessment, Assessment]:
    """Return two assessments; both must pass, with matching evidence classes.

    Matching declarations do not establish an observed hardware comparison.
    """
    a, b = assess(left, expected, as_of=as_of), assess(right, expected, as_of=as_of)
    if a.status == b.status == "applicable_to_declared_model":
        if left.provenance.evidence_class != right.provenance.evidence_class:
            reason = ("evidence classes differ; do not compare vendor peaks, synthetic estimates and measurements",)
            return tuple(Assessment("unsupported", reason, None, x.record_sha256,
                                    x.unknown_dimensions) for x in (a, b))
    return a, b
