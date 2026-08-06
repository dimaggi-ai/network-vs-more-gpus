# Metric Framework

This document defines every metric used in the project, states what each one is good for, and records where the supplied metric proposal was changed and why.

## 1. The accounting identity

For a pool of `N` accelerators over a window of `T` seconds, the total resource is `N x T` accelerator-seconds. Every accelerator-second is assigned exactly one **terminal fate**:

| Fate | Definition |
|---|---|
| **Productive** | Computation whose result is present in the delivered model state. |
| **Blocked** | Allocated and powered, but not computing: exposed collective communication, synchronization wait, pipeline bubble, checkpoint stall, restart. |
| **Discarded** | Computation performed and then thrown away: everything between the last durable checkpoint and failure detection. |
| **Unavailable** | Owned but not usable by the job: down awaiting repair, held as a spare, or stranded behind a domain, power, or fragmentation limit. |

`Productive + Blocked + Discarded + Unavailable = N x T` exactly. This is asserted numerically on every model evaluation by `CapacityLedger.check_invariants`, with a relative tolerance of 1e-9 on the analytical path.

Two conventions make the buckets genuinely exclusive:

**Classification is by terminal fate, not by activity.** A second spent computing inside a window that was later discarded is Discarded, not Productive. Without this rule the same second can be counted twice.

**Re-execution is counted once.** When a job restarts from a checkpoint and repeats lost progress, the *original* attempt is Discarded and the *re-execution* is ordinary Productive and Blocked time. Charging both would double-count the loss. This is the most common error in informal goodput accounting and is the reason the identity closes.

## 2. Useful Capacity Fraction

```
UCF = Productive / (N x T)
```

The share of what was paid for that became useful work. Denominator is the pool, not the job, so spares and stranded capacity reduce it.

## 3. Relationship to existing metrics

Two metrics are already established. Neither is replaced; both are computed for comparison.

**Model FLOPs Utilization** relates achieved model FLOPs to peak over the wall-clock of a running step. It is blind to failures: a job that loses a third of its time to restarts can still report a high MFU.

**Effective Training Time Ratio**, as defined by Meta (HPCA 2025), is productive runtime over the job's available wallclock. It counts normal stepping as productive, so exposed communication and synchronization wait are credited as productive time, and it does not see pool capacity the job never held.

The two overlap, and their product is not a capacity fraction: MFU's denominator is peak FLOPs over running time, ETTR's is available job wallclock, and neither denominator is the accelerator-seconds paid for. The practical consequence is measurable. At the study baseline of 16,384 accelerators the model reports ETTR 0.889 and UCF 0.627, a gap of 26 percentage points. An operator reading only ETTR would conclude the fleet is 89 percent effective while 37 percent of the accelerator-seconds purchased produced nothing.

## 4. Substitution-Equivalent Accelerators

Let `Pi(N, c)` be productive accelerator-seconds per second for a pool of `N` accelerators under configuration `c`. For a baseline `c0` at pool size `N0` and an intervention producing configuration `c1`:

```
SEA(c1 ; c0, N0) = the dN solving   Pi(N0 + dN, c0) = Pi(N0, c1)
```

SEA answers the question an operator actually asks: how many accelerators would I have to buy to get what this improvement gives me.

Properties that matter:

* **It prices the alternative correctly.** A purchased accelerator inherits the fleet's inefficiency; it does not arrive fully productive.
* **It accounts for the cost of growth.** Adding accelerators raises communication volume and shortens job MTTF, both of which the model captures because `Pi` is recomputed at the larger size.
* **It can be unbounded.** Where `Pi(N, c0)` never reaches the target, SEA is infinite: the improvement cannot be bought with accelerators at any quantity. This is reported as an outcome, not an error.
* **It requires no prices.** SEA *is* the break-even cost. An intervention is worth funding when it costs less than SEA fully-loaded accelerators. Because the decision rule needs only a ratio, the project never invents absolute infrastructure prices.

Practically, SEA equals the productive gain divided by the marginal productivity of an added accelerator. At the study baseline that marginal productivity is 0.502, so a gain of 141 productive accelerators from doubling bandwidth is worth 281 accelerators of purchase.

## 5. Why the supplied metric was replaced

The supplied proposal was:

```
Equivalent accelerators recovered = (accelerator-seconds of idle, blocked, or repeated work avoided) / (measurement period)
```

It was rejected as a decision metric for three reasons, and retained only as a comparison target so its error could be measured rather than asserted.

1. **It assumes the marginal accelerator is fully productive.** Dividing avoided accelerator-seconds by the window yields a count of accelerators only if a new accelerator contributes a full accelerator-second of useful work per second. That contradicts the premise of the project.
2. **It ignores the cost of growth.** It cannot represent the fact that adding accelerators makes communication and failures worse.
3. **It cannot express unattainability.** It always returns a finite number, including in regimes where no purchase can match the intervention.

The measured consequence: the informal figure is 0.73 of SEA at 2,048 accelerators and 0.16 at 65,536. It understates intervention value by 1.4x at small scale and 6.2x at large scale, and the bias grows precisely where the decision matters most.

## 6. Attribution when interventions interact

`Pi` is nonlinear in the configuration, so SEA is not additive across interventions: once one bottleneck is relieved, the others are worth a different amount. Contributions are attributed with the Shapley value over the set of interventions, evaluating all `2^k` subsets. Shapley is used because it is the standard order-independent attribution for exactly this situation and because it satisfies efficiency, which is asserted as a test.

The measured non-additivity, excluding cases where SEA is unbounded:

| Bundle | Mean additivity error | Interpretation |
|---|---|---|
| Bandwidth plus flat fabric | +6.5 percent | Substitutes, as expected |
| Detection plus restart plus checkpoint | -8.9 percent | **Complements**, contrary to the stated hypothesis |
| Mixed network and reliability | +7.5 percent | Substitutes |

Recovery improvements compound: faster checkpointing shortens the optimal checkpoint interval, which shrinks the window of discarded work, which makes faster detection and restart worth more. Reported as a refutation of hypothesis H3's stated direction.

## 7. Uncertainty and rank stability

Point estimates are not reported as decisions. Assumption parameters are drawn from documented ranges and the ranking is recomputed per draw; the reported quantity is the share of draws in which each intervention ranks first. Across 323 in-envelope draws at 16,384 accelerators: halving the failure rate 52.3 percent, straggler control 24.8 percent, faster checkpointing 19.2 percent, faster restart 3.1 percent, quadrupled bandwidth 0.6 percent.

## 8. Validity envelope

The fast analytical path is trusted only where **recovery pressure**, the share of runtime consumed by discarded work plus restart, is below 0.25. Inside that envelope the analytical and event-driven paths agree to within 1.4 percent on UCF; outside it they diverge to 29.6 percent. Rows outside the envelope are retained in the raw output, flagged, and excluded from headline claims; where such a case matters, it is recomputed with the event-driven path.

## 9. Metrics for inference

Inference is out of scope for this version. The analogous decomposition would classify accelerator-seconds by whether they served a token that met its latency target, and the substitution metric would equalize served throughput at a fixed tail-latency constraint rather than productive accelerator-seconds. No inference metric in this document has been implemented or tested, and none is used in any result.

## 10. Guard against misuse

The framework converts infrastructure improvements into accelerator units, which invites overstatement. Three rules constrain it:

1. SEA is always reported with the regime it was computed in. A single headline number across regimes would be meaningless given the rank reversals.
2. SEA is a break-even cost, not a saving. It says how much an intervention may cost before it stops paying, not how much money it recovers.
3. Where SEA is unbounded, the correct statement is that the improvement cannot be bought with accelerators, not that it is worth infinite accelerators.
