# Decision Log

Every decision that changed scope, methodology, tooling, a metric, or a contribution. Dated, with the evidence that drove it.

---

## D1. Adopt scope F (investment decision framework) built on A and E, single campus
**Date:** 2026-08-05. **Phase:** 0.

Six candidate scopes were scored against the mandate's rubric in `strategy_comparison.md`. The combined decision framework scored highest (7.80) because the decision layer is the part no inspected prior work provides, and because it builds on the communication and reliability scopes rather than competing with them.

Cross-campus pooling scored 6.50 but was rejected as a primary scope on validation feasibility: the only public anchors are a vendor blog with an accompanying paper and one field-trial abstract, neither carrying the parameter detail needed for calibration. A headline cross-campus claim would rest on unvalidated extrapolation, which Gate 5 forbids. Retained as a bounded optional extension.

---

## D2. Do not build a new simulator; build an analytical model with two fidelities
**Date:** 2026-08-05. **Phase:** 3.

The mandate forbids a new general-purpose simulator absent evidence that existing ones cannot serve. The evidence goes the other way: ASTRA-sim 3.0, SimAI, Echo, ATLAHS, and Charon are all actively maintained and model communication at fidelities a solo researcher cannot match.

But none of them models failures, and the reliability-side tool that does (AIReSim) explicitly models no network at all. The gap is not fidelity, it is the *combination* plus a decision layer. A compact analytical model with an independent event-driven cross-check serves that better than extending a large simulator, and it keeps the whole experiment program runnable in ten seconds, which is what makes the uncertainty analysis affordable.

Consequence: the communication and reliability formulations are deliberately textbook, so they can be checked against independent tools and closed forms rather than trusted.

---

## D3. Replace the supplied metric with a substitution-based definition
**Date:** 2026-08-05. **Phase:** 2.

The supplied "equivalent accelerators recovered" divides avoided idle and repeated accelerator-seconds by the measurement period. That is only an accelerator count if a marginal accelerator is fully productive, which contradicts the project's own premise, and it cannot represent regimes where no purchase matches the intervention.

Replaced by Substitution-Equivalent Accelerators, defined by solving for the accelerator count that delivers equal productive throughput. The supplied metric is retained and computed solely so its bias can be measured. It understates value by 1.4x at 2,048 accelerators and 6.2x at 65,536 (figures restated after D13 corrected the solver sampling).

---

## D4. Model telemetry as detection time, not as a resolution parameter
**Date:** 2026-08-05. **Phase:** 3.

Telemetry resolution does not affect capacity directly; it affects capacity only by changing how quickly a fault is detected. Modeling "better telemetry" as its own variable would be unfalsifiable. Detection time is measurable, comparable across published systems, and is what the accounting actually consumes.

---

## D5. Correct the Llama 3 reliability inputs after the first validation run failed
**Date:** 2026-08-05. **Phase:** 6.

The first validation run passed the throughput and interruption-count checks and failed the ETTR check, predicting 0.600 against a published "higher than 90 percent".

Diagnosis: two inputs were misspecified, not the model. The configuration assumed zero replacement capacity, so every failure waited a full physical repair; and it assumed a 60-second fully blocking checkpoint write. Both contradict the source paper, which states the cluster held 24,000 GPUs while the job used up to 16,000, that failures were handled by automation with significant manual intervention only three times in 54 days, and that per-GPU checkpoint state is 1 MB to 4 GB written against a high-throughput storage fabric.

Corrected to a 3 percent spare fraction and a 20-second blocking write, both marked `[derived]` in the config with their justification. Predicted ETTR became 0.888 and the check passed. The pre-correction result is reported in the validation report rather than discarded.

An arithmetic consequence worth recording: at the Daly optimum, checkpoint plus lost-work overhead is approximately `sqrt(2 w / MTTF)`. With a 60-second blocking write and a 2.34-hour job MTTF that is about 12 percent, so a published ETTR above 90 percent is not reachable with fully blocking 60-second checkpoints at this scale regardless of anything else in the model.

---

## D6. Calibrate one compute-side parameter, hold out a second configuration
**Date:** 2026-08-05. **Phase:** 6.

Kernel efficiency is a compute-side unknown that no public source reports. Rather than assume it, it is calibrated on the published 8,192-accelerator configuration and then held fixed to predict the 16,384-accelerator configuration, which doubles data parallelism at fixed global batch and spreads the gradient all-reduce across twice as many pods.

Held-out prediction: 406.3 TFLOP/s per accelerator against a published 400, a 1.6 percent error; MFU 41.06 percent against a published 41 percent. No network or reliability parameter was tuned. The calibrated value, 0.571, is plausible for realized large-GEMM efficiency including non-GEMM work.

---

## D7. Exclude the third published scaling row from quantitative validation
**Date:** 2026-08-05. **Phase:** 6.

The source paper's Table 4 third row lists TP=8, CP=16, PP=16, DP=4 alongside 16,384 GPUs. Those degrees multiply to 8,192. The row is internally inconsistent as printed, and resolving it would require guessing which field is wrong. It is excluded from the quantitative comparison and the discrepancy is reported.

---

## D8. Separate the validated configuration from the study baseline
**Date:** 2026-08-05. **Phase:** 4.

`configs/scenarios/llama3_405b_16k.yaml` is the validation target and is not used for any study result. `configs/scenarios/reference_405b_16k.yaml` is the study baseline: it adopts the calibrated kernel efficiency, enables a nonzero straggler term, and shortens the window to 30 days. Keeping them separate prevents a study parameter from silently contaminating a validation claim.

The straggler coefficient of variation is set to 0.02, which produces roughly a 9 percent synchronization tax at 16,384 accelerators, consistent with the reported magnitude that 42.5 percent of production jobs run at least 10 percent slower due to stragglers.

---

## D9. Define and enforce a validity envelope for the fast path
**Date:** 2026-08-05. **Phase:** 6.

Cross-fidelity comparison showed the analytical and event-driven paths agreeing to under 1 percent across most of the parameter space but diverging to 29.6 percent in the corner where mean time to failure approaches the recovery time.

Rather than report the corner with a caveat or quietly drop it, a dimensionless validity parameter was defined: recovery pressure, the share of runtime consumed by discarded work plus restart. Below 0.25 the two paths agree to within 1.4 percent. Rows above it are flagged in the raw output and excluded from headline claims; where such a case is load-bearing it is recomputed with the event-driven path. The threshold is pinned by a test that fails if the fast path ever drifts more than 5 percent while still declaring itself valid.

---

## D10. Report a refuted hypothesis rather than restating it
**Date:** 2026-08-05. **Phase:** 7.

Hypothesis H3 predicted that interventions are substitutes. Network interventions are (mean additivity error +6.5 percent), but recovery interventions are complements (-8.9 percent; figures restated after D13): faster checkpointing shortens the optimal interval, which shrinks the discarded window, which raises the value of faster detection and restart. The hypothesis is recorded as partially refuted in the charter and the mechanism is reported in the paper.

---

## D11. Use a Python port of the visualization palette validator
**Date:** 2026-08-05. **Phase:** 7.

The reference palette validator ships as JavaScript, and node on this machine fails to start because its ICU dependency is missing. Installing a mismatched ICU version would risk breaking the user's toolchain for an unrelated reason.

The validator's computations were ported to `figures/palette_check.py` and verified against the documented expected values: the port reproduces the documented worst adjacent-pair figures exactly (CVD delta-E 9.1 light, 8.4 dark), which confirms the port is faithful. The decision map is a categorical heatmap, so its four hues were selected to pass the stricter all-pairs rule rather than the adjacent-pair default.

---

## D12. Title changed from assertion to qualified finding
**Date:** 2026-08-05. **Phase:** 7.

The supplied title asserts that network capacity is compute capacity. The results do not support it as a general claim. At 16,384 accelerators under parameter uncertainty, bandwidth ranks first in 0.6 percent of draws; reliability ranks first in 52.3 percent. Bandwidth does win at 65,536 accelerators in low-failure, heavily-oversubscribed regimes.

The working title now states the qualified version. Recording this explicitly because the mandate requires that the title not be forced to confirm itself.

---

## D13. Densify the scaling-curve sampling near the baseline
**Date:** 2026-08-05. **Phase:** 10 quality pass.

The substitution solver interpolated the throughput-versus-pool curve across a
uniform grid whose first step was 84 data-parallel replicas (about 11,000
accelerators). The curve is concave, so interpolating across that chord
understated the marginal productivity of the next accelerator (0.444 as a chord
against 0.502 measured one replica out) and inflated every near-field SEA value
by roughly 13 percent.

Fixed by sampling the first sixteen data-parallel steps densely and growing
geometrically beyond. Effects: small SEA values fell (doubling bandwidth at the
16K baseline: 318 to 281), the informal-metric understatement factor at 65,536
accelerators fell from 9.9x to 6.2x, and recovery complementarity strengthened
from -4.1 to -8.9 percent. The decision map, the rank-stability shares, the
capacity ledger, and every validation result were unchanged, which is itself a
useful robustness observation: the sampling bias shifted magnitudes, not
comparisons. All quoted numbers were regenerated and the claims test updated.

---

## D14. Correct a detection-time accounting bug in the event-driven path; reframe the envelope as a scope guard
**Date:** 2026-08-06. **Phase:** 10 quality pass, from an independent code review.

The Monte Carlo path subtracted detection time from the running and checkpoint
buckets on each failure, although detection wall-time had never been added to
those buckets, and then rescaled every bucket so their sum matched the window.
The rescale hid the deficit and biased all buckets at high failure rates: at
65,536 accelerators and 3e-2 failures per node-day the path under-reported
running time by roughly 11 percent while the analytical path was within 0.06
percent of an independently written event simulation.

Consequences of the correction:

1. The two implementations now agree to within 0.02 percent at every tested
   severity, so the "validity envelope" is no longer a fidelity boundary. The
   0.25 recovery-pressure threshold is retained as a modeling-scope guard:
   beyond it the independence assumptions are outside their calibrated regime
   and no operator would run such a configuration uncorrected. D9's original
   justification is superseded by this entry.
2. The 131,000-accelerator counterexample was restated with corrected numbers:
   throughput peaks 1.2 percent above the baseline pool (31,566 at 1.25x)
   before declining, rather than declining monotonically as previously
   reported. The unattainability conclusion itself survives: the best
   reachable throughput (31,566) remains far below the halved-failure-rate
   target (38,640), and both implementations agree on this case to 0.03
   percent.
3. The Monte Carlo path now asserts wall-clock conservation internally and
   raises instead of rescaling, and a cross-fidelity test asserts agreement at
   every cell rather than skipping the harsh ones.

---

## D15. Give relatively-worded interventions multiplier semantics; reprice pool-repartitioning interventions at fixed budget
**Date:** 2026-08-06. **Phase:** 10 quality pass, from the same review.

Interventions named as ratios ("bandwidth 2x", "failure rate halved",
"straggler control") were implemented as absolute target values. On the
regime-shifted baselines of the decision map that silently changed their
strength: at a cell with 0.012 failures per node-day, "halved" (implemented as
an absolute 0.0025) was nearly a 5x improvement, and at 0.001 it was a
degradation clamped to zero. The decision map and the uncertainty analysis were
therefore comparing interventions whose strength varied by cell in ways their
names did not state.

Relative interventions are now true multipliers applied to whatever the cell's
baseline is, and absolute-target interventions ("deploy a system that detects
in 20 s") remain absolute, which is their honest semantics. Interventions that
re-partition the paid pool (spares) are re-planned at the baseline's pool size,
so adding spares shrinks the job rather than growing the budget.

Effects on results, all regenerated: the decision map now has three distinct
winners rather than four (faster restart no longer ranks first anywhere,
although it remains worth hundreds to thousands of accelerators); reliability
dominates most of the 65,536-accelerator map; the rank-stability leader at
16,384 accelerators becomes the 5-second checkpoint capability (45.8 percent of
draws) ahead of halving the failure rate (29.1 percent); and the informal
metric's worst-case understatement is 4.6x rather than 6.2x. The qualitative
conclusions (rank reversals exist, bandwidth rarely wins at 16K scale, recovery
improvements are complements) are unchanged.

---

## D16. Add a span tier to the topology, inert unless a job is explicitly split
**Date:** 2026-09-01. **Phase:** 6.

The three-tier model (scale-up, pod, cross-pod) describes one hall. Answering
whether a job can be spread across halls needs a fourth tier, and the fourth
tier is not like the other three: its bandwidth is an *aggregate shared
circuit*, not a per-accelerator entitlement, so the number of groups crossing it
determines what each one gets. `TopologySpec` therefore gains `halls`,
`alpha_span_us`, `span_bw_gbps`, `span_dimension`, and `span_placement`.

Every one of those defaults to the hall-local case, and every span term is
exactly zero when `halls == 1`. This is load-bearing rather than tidy: this
repository's published results must not move, and
`validate_span.py`'s `hall-local-is-the-published-three-tier-model` and
`tests/test_span.py`'s `test_published_results_do_not_move` both assert
byte-identical step times against the pre-span model. The Llama 3 validation
program passes unchanged.

A per-rank share is also capped at the rank's own NIC line rate. Without that
cap an arbitrarily wide circuit lets the span tier outrun the fabric the ranks
are actually attached to, and an early draft of the model duly reported that
splitting a job across two halls made it *faster*.

---

## D17. Expose stitch latency rather than overlapping it, and make the pipeline's crossing fraction a placement question
**Date:** 2026-09-01. **Phase:** 6.

Two decisions about how a stitch is charged, both of which change the answer.

**Latency is exposed.** The bandwidth term of a ring collective streams
alongside the backward pass, but the `2(k-1)` chained hops are serially
dependent and drain at the step boundary. Charging the stitch's latency at the
same overlap fraction as its bandwidth term makes a model conclude that distance
is free, which is exactly the error this study exists to test for. So
`span_alpha_seconds` is split out and added back in full after the overlap
fraction is applied to the rest. Intra-hall alphas are deliberately excluded:
they are microseconds, and touching them would move published results.

**The pipeline's crossing fraction follows placement.** A spanned pipeline is
not costed at its worst boundary for every handoff, the way the cross-pod case
is. How many of the `pp - 1` stage boundaries land on a hall edge depends
entirely on how stages were laid out: contiguous blocks put exactly `halls - 1`
there, round-robin interleaving puts every one there. Charging all of them
unconditionally would manufacture the distance cliff this study is testing;
charging one unconditionally would hide it. The first draft did the latter, and
the atlas duly reported that rank order is irrelevant to a pipeline — a 15x
error at the study baseline, and a false one. The logic now lives in
`pipeline_crossing_fraction`, named rather than inlined so that
`tests/test_span.py` can delete it and require the registry to notice.

---

## D18. Report retention only at or below 3.2 Tbit/s, and decline the Corning bandwidth anchor
**Date:** 2026-09-01. **Phase:** 6.

Two limits found while building the validation registry, both recorded rather
than worked around.

**The reported width ceiling.** Above 12.8 Tbit/s the tier model makes a split
job up to 0.36 percent *faster* than a hall-local one: each hall's residual
group occupies fewer pods, and the cross-pod hop it drops costs more than the
stitch exchange replacing it. That is a real consequence of the tier structure,
not a rounding error. Retention is therefore reported only at or below 3.2
Tbit/s — 8x400G, the widest metro DCI class in deployment — and the wider probe
circuit appears only inside fixed-width comparisons, where the effect cancels on
both sides. The onset width and the magnitude are both printed in the registry's
DECLINED list.

**The declined anchor.** Corning's study (S20) reports that doubling inter-DC
bandwidth improves overlap by at most 0.66 percent, but does not publish the
circuit width it modelled. This model produces gains from 21 percent (doubling
400G) down to exactly zero (doubling past the NIC cap) for that same operation.
A first draft of the registry "passed" the anchor by starting the doubling at
204.8 Tbit/s, which is to say by choosing the answer; the number would have
looked like calibration and been nothing of the sort. It is now DECLINED. What
survives is the study's short-distance claim, which can be stated without
reference to any width — the cost of moving a hall from next-door to 10 km at a
*fixed* circuit — and therefore has no free parameter to fit. The zero-gain
result is filed as sanity rather than calibration, because once the per-rank
share is pinned to the NIC line rate it is true by construction.

Of Corning's four headline numbers this repository reproduces one, declines the
26x magnitude and the bandwidth claim, and does not test the hollow-core figure.
