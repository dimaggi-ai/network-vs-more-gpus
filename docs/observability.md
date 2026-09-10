# Offline collective evidence

`netcap.observability` compares normalized JSON observations from two matched
collective benchmark campaigns. It extends the networking model's evidence
boundary; it is not a separate fabric simulator or an NCCL instrumentation agent.

From the repository root, in the normal netcap Python environment:

```bash
PYTHONPATH=src python -m netcap.observability examples/observability/baseline.json examples/observability/candidate.json
```

The examples are **synthetic**, not NCCL hardware results. Their 100-to-75 GB/s
change produces a 25% relative algorithm-bandwidth drop and a regression flag
at the default 15% threshold. That threshold is an exposed screening input,
not a calibrated fault detector. The result explicitly leaves cause undetermined.

## Input contract

Use `netcap-collective-observation/v1`, with `evidence_kind` (`synthetic` or
`benchmark`), source artifact identifier, context and samples. The checked-in
examples are complete format references; unknown fields are rejected.

Context must identify collective, dtype, rank count, hardware, topology,
transport, library version and measurement protocol. The protocol identifier
must encode comparable warmup, iterations, software settings, affinity and
background load. Matching strings do not independently verify these conditions.

Each sample names message size in bytes, in-place/out-of-place mode,
`algbw_GBps`, `busbw_GBps` and correctness error count (`wrong_elements`). Both
bandwidth fields use decimal **GB/s**, not Gbps. Bus bandwidth is a collective
normalization, not a physical link reading. No raw NCCL debug-log parser or
`nccl-tests` text-format parser is implemented: preserve and normalize the actual
source artifact explicitly before importing it.

Only equal contexts, evidence kinds and message-size/mode grids are compared.
Repeated samples are summarized by median within each matched group; values
from different message sizes are not averaged. Nonfinite/negative bandwidth,
invalid ranks/modes and missing fields are refused. A zero baseline yields
`insufficient-baseline`; correctness errors invalidate the performance evidence.
Malformed input exits 2. A valid regression report exits 0 because it is a
diagnostic result, not a deployment gate.

## Interpretation and reuse

A regression is not proof of NIC, InfiniBand, GPU or optical failure. Library
settings, workload changes and noisy neighbors can also change performance.
The tool does not tune `TopologySpec.net_efficiency`, estimate stranded GPUs,
compute a capacity gain or execute remediation. Use the existing netcap models
for explicit scenario economics, separately from these observations.

Optical telemetry continues to belong in `optical-circuit-intent`'s drift/HEDGE
modules. Future correlation needs synchronized event windows, rank-to-device
and topology mappings, matched control runs, missing-data handling and chaos
fidelity identifiers. Merely coincident observations are not causal evidence.
Those inputs and the correlator are not implemented in this increment.

Tests: `python -m pytest tests/test_observability.py`; included automatically in
the repository's existing full-suite CI command.
