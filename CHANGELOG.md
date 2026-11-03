# Changelog

## 1.1.0 (release candidate; not released)

- Adds the existing span tier, latency-regime atlas and span validation to the release scope since 1.0.0.
- Includes bounded WAN accounting, observability, installed CLI repair and historical experiment outputs.
- Preserves disconnected-state provenance and refuses nonfinite WAN outputs; repairs finite inversion across overflowing lower brackets. `required_bandwidth` resolves widths only to about `maximum_gbps/2^70`: at `1e300`, the default workload/target 0.5 returns about `8.5e278` Gbit/s in every regime despite different minima. These feasible answers convey no useful minimum-width information. Keep the bound near the expected answer; the default `1e6` resolution is about `8.5e-16` Gbit/s.
- Validates span numeric/integer input domains without silent string coercion; constrains active halls to the named degree and retains nondivisible supported cuts.
- Compatibility: span values accept Python `int`/`float` only (not NumPy scalars). YAML 1.1 `4e2` parses as text and is refused; use `400` or `4.0e+2` for bandwidth. Halls and active degrees require integer values.
- Adds an optional offline capability-declaration applicability API with explicit flat-switched abstraction, precision/unit/provenance checks. It does not authorize or execute a model.
- Clarifies fixed H100-class tp8/pp16 study plans, absent memory-capacity feasibility and automatic replanning, aggregate span bandwidth and model-only claims.
- Includes intervening article, README, validation CI and cover-art changes in the candidate manifest.

See [the candidate manifest and remaining gates](docs/releases/v1.1.0.md). Existing result artifacts retain their historical producer identities and are not fresh candidate measurements.

## 1.0.0 (2026-08-05)

First public release.

### Model
- Four-fate accelerator-time accounting with numerically enforced closure.
- Alpha-beta hierarchical collective model over a scale-up, pod, cross-pod hierarchy, with rank-layout awareness so data-parallel collectives do not incorrectly receive scale-up bandwidth.
- Failure, detection, restart, repair, spare, and checkpoint model with two independent implementations.
- Substitution-Equivalent Accelerators, Shapley attribution, and break-even cost.

### Validation
- Kernel efficiency calibrated on the published 8,192-accelerator Llama 3 configuration; held-out 16,384-accelerator prediction within 1.6 percent.
- Daly's optimal checkpoint interval and Meta's expected-ETTR expression both recovered.
- Scope guard at recovery pressure 0.25; the two independent reliability implementations agree to 0.02 percent at every tested severity (an earlier event-driven accounting bug, found in independent review, is documented in DECISIONS.md D14).
- One validation failure recorded and diagnosed rather than tuned away (see DECISIONS.md D5).
- Relative interventions use multiplier semantics so a 2x is a 2x in every regime (D15); the substitution solver samples densely near the baseline (D13).

### Experiments
- Nine experiment families, all raw outputs immutable with provenance sidecars.
- Seven figures generated from raw results only.

### Known limitations
- No hardware measurement anywhere in the project.
- Failure model is per-node Poisson and is optimistic below roughly 4,096 accelerators.
- Cross-campus extension not implemented.
