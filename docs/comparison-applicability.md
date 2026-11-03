# Checking declarations before a model comparison

`netcap.capability_evidence` is a pure, offline Python API for checking whether
supplied accelerator declarations fit this model's input conventions. It does
not alter or automatically feed the existing numerical model.

The current model expects dense BF16 compute and per-accelerator unidirectional
injection bandwidth. An FP4 peak cannot populate `peak_flops_bf16`. A domain's
aggregate rate cannot be divided by its chip count to infer injection bandwidth;
a bidirectional rate cannot simply be halved. Unknown units and scope are refused.

The scale-up tier uses one per-rank rate and a constant latency; it has no hop,
diameter, bisection or expert all-to-all model. A profile must explicitly declare
`scaleup_topology="flat_switched"` in addition to the `netcap_four_tier` context.
Missing/unknown topology is incomplete. Torus, Boardfly, dragonfly and other
nonflat topology declarations are unsupported. Domain size alone cannot qualify
such a profile. This is a declaration of the modeled flat abstraction, not
authentication that a physical fabric is switched or nonblocking; topology
calibration remains unknown even when the declaration passes.

## Synthetic success, refusal and correction

These are invented fixture values, not specifications or measurements of a vendor
product. Run from an installed checkout, or with `PYTHONPATH=src`:

```python
from dataclasses import replace
from netcap.capability_evidence import (
    CapabilityEvidence, ComparisonContext, Provenance, Quantity, assess,
)

context = ComparisonContext(
    workload="synthetic-training", quality_target="fixed-quality-A",
    protocol="protocol-A", model_family="dense_decoder_only",
    topology_model="netcap_four_tier", parallelism=(8, 1, 1, 1),
    window_seconds=3600,
)
profile = CapabilityEvidence(
    hardware="synthetic-device", precision="BF16", sparsity="dense",
    compute=Quantity(1, "PFLOP/s", "per_accelerator", "not_applicable"),
    memory=Quantity(80, "GB", "per_accelerator", "not_applicable"),
    scaleup=Quantity(100, "GB/s", "per_accelerator", "unidirectional"),
    scaleout=Quantity(400, "Gbit/s", "per_accelerator", "unidirectional"),
    context=context,
    provenance=Provenance("synthetic:example", "fixture-v1", "2026-11-03",
                          "a" * 64, "synthetic"),
    scaleup_topology="flat_switched",  # explicit synthetic model assumption
)
result = assess(profile, context, as_of="2026-11-03")
assert result.status == "applicable_to_declared_model"
assert result.inputs.peak_flops_bf16 == 1e15
assert result.inputs.scaleup_bw_gbps == 800
assert "power" in result.unknown_dimensions  # not zero, not calibrated

fp4_only = replace(profile, precision="FP4")
refused = assess(fp4_only, context, as_of="2026-11-03")
assert refused.status == "unsupported" and refused.inputs is None

# Correction means supplying evidence with the required declarations, NOT
# relabelling an actual FP4 vendor measurement as BF16. Here both are fixtures.
assert assess(profile, context, as_of="2026-11-03") == result
```

`assess_pair(left, right, expected, as_of=...)` returns two assessments. **Both**
must have `applicable_to_declared_model` status. The shared expected context
binds workload, quality target, protocol, model family, topology abstraction,
parallelism degrees and time window. Evidence classes must also match: measured,
synthetic and vendor-specification records cannot be mixed. Matching labels do
not establish that two observed experiments are genuinely comparable, or that
an analyst's declarations are correct.

## Result contract

- `invalid`: wrong type, non-finite/nonpositive or out-of-bound number, malformed
  date, unbounded/control-character text, or invalid typed context.
- `incomplete`: required quantity, provenance or scale-up topology declaration is missing. No normalized inputs
  are returned; a missing value never becomes zero.
- `unsupported`: precision, sparsity, units, direction, scope, declared model or
  context do not fit, or an explicitly declared evidence expiry has passed.
- `applicable_to_declared_model`: only these supplied declarations fit the narrow
  input interface. Normalized `ModelInputs` are returned without running it.

Exact decimal conversions are supported for FLOP/s, TFLOP/s and PFLOP/s;
Gbit/s, Tbit/s, GB/s and TB/s; and memory GB or explicitly binary GiB. Conversion
does not imply any hardware efficiency, collective bandwidth or precision
conversion. Numeric inputs must be builtin integers or floats, not booleans or
numeric strings, positive and finite, and at most `1e100` before unit conversion.
Converted values must remain finite and positive; the bound is not a physical
plausibility check (`1e100 PFLOP/s` becomes `1e115 FLOP/s`, and tiny positive
values are allowed). Parallelism degrees
are integer values from 1 to 1,000,000. These are defensive API bounds, not
validated hardware operating ranges. The API accepts typed dataclasses, not
arbitrary JSON; it supplies no permissive parser or unknown-field dropping.

Host capacity, SRAM, topology calibration, power and price are always disclosed
as unknown dimensions of this gate. `netcap_four_tier` identifies a model
abstraction, not proof that a particular fabric follows it. Alternative model
families and topology abstractions are unsupported. Applicability does not
establish full-stack support, calibration, observed superiority, production
readiness, regulatory assurance or permission to execute.

## Provenance and freshness limits

The API retains a source label/location, revision, retrieval date, source SHA-256
and evidence class. It never visits the source or checks that these statements
are true. The source hash is supplied metadata, not authentication. The returned
`record_sha256` identifies the supplied evidence, expected context, explicit
assessment date and gate-version label; it is not a signature or proof of
execution. It identifies the record as supplied, not normalized semantic identity:
for example, numeric `1` and `1.0` produce different hashes even when their
normalized inputs are equal. Any changed record must be assessed again.

There is no implicit clock or invented age cutoff. `as_of` is caller-supplied,
and a future retrieval date is invalid relative to it. An optional, explicitly
declared `valid_until` produces refusal after that date. An old retrieval date
without such an expiry can still match the input interface; it does **not** mean
the evidence is current. Independent source/revision/freshness review remains
necessary, including when all checks pass. A reference from an older generation
must not be relabelled as evidence for a newer device.

## Checks and remaining work

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p test_capability_evidence.py -v
```

The synthetic fault matrix exercises omitted quantities/provenance, precision and
sparsity substitution, ambiguous dimensions, explicit expiry, malformed dates,
invalid numeric values, missing/nonflat scale-up topology, workload/quality/context
mismatch, correction, repeatable
hashes and input preservation. It makes no network, subprocess or device calls.
These checks qualify this metadata contract only. They do not qualify host or
network isolation, silicon, cooling, a stateful agent loop, or any physical
failure scenario. Independent review and designated technical acceptance remain
separate from a passing unit suite.
