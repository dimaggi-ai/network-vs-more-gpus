# Changelog

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
