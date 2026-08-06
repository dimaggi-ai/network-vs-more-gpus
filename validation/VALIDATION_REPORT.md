# Validation Report

Generated from `results/validation/llama3_validation.json`, `results/raw/e8_cross_fidelity.csv`, and the test suite. Reproduce with `make validate`.

## Summary

| Layer | What it checks | Result |
|---|---|---|
| Invariants | Accounting closes; limiting cases; monotonicity | Pass, 46 tests |
| Closed forms | Daly optimal checkpoint interval; Meta expected-ETTR expression | Pass, within 10 percent and 5 percent respectively |
| External measurement | Held-out prediction of a published configuration | Pass, 1.6 percent throughput error |
| Cross-fidelity | Analytical against event-driven | Pass inside the declared envelope, 1.4 percent maximum |

Nothing in this project has been validated against hardware the author operated. Every result is analytical or simulated.

## 1. Invariants

Asserted on every model evaluation, not only in tests:

* **Time conservation.** Productive plus blocked plus discarded plus unavailable equals accelerator-seconds paid for, to a relative tolerance of 1e-9 on the analytical path and 1e-6 on the Monte Carlo path.
* **Non-negativity.** No bucket goes negative under any tested combination of oversubscription and failure rate.
* **Infinite bandwidth.** As link bandwidth grows without bound, exposed communication falls below 1e-6 of the total.
* **Zero failures.** With a zero failure rate, discarded work and restart time are exactly zero.
* **Ideal cluster.** With no communication cost, no failures, no checkpoint cost, and no pipeline, the useful capacity fraction exceeds 0.999.
* **Monotonicity.** Useful capacity is non-decreasing in bandwidth and non-increasing in both failure rate and oversubscription.
* **Metric ordering.** For a pool equal to the job size, ETTR is never below UCF, since ETTR credits exposed communication as productive.

## 2. Independently derived closed forms

### 2.1 Daly's optimal checkpoint interval

Daly's first-order optimum, `sqrt(2 w M) - w`, is derived independently of this model. The test recovers it by brute-force search over the model's own predicted productive time across 4,000 candidate intervals, for checkpoint costs of 10, 30, and 120 seconds.

**Predefined threshold: within 10 percent. Result: pass at all three costs.**

### 2.2 Meta's expected-ETTR expression

The expression published in HPCA 2025,

```
E[ETTR] ~= (1 - N r_f (u0 + dt_cp/2)) / (1 + w_cp / dt_cp)
```

is reproduced verbatim in `netcap.reliability.meta_ettr_closed_form` and compared against the model at 2,048, 8,192, and 16,384 accelerators. The published expression is a small-loss approximation that ignores detection time and treats losses as additive, so agreement is checked in that regime with detection set to zero.

**Predefined threshold: within 5 percent. Result: pass at all three scales.**

### 2.3 Job mean time to failure

Using RSC-1's measured rate of 6.50 failures per thousand node-days:

| Scale | Reported | Model | Error |
|---|---|---|---|
| 16,384 accelerators | 1.8 h | 1.803 h | 0.2 percent |
| 131,072 accelerators | 0.23 h | 0.225 h | 2.2 percent |
| 1,024 accelerators | 7.9 h | 28.8 h | **265 percent** |

The first two are the scales at which the source reports projections; the model reproduces them. The third is a direct measurement and the model misses it badly. This is reported as a limitation, not smoothed over: the source's own figures are not mutually consistent under inverse scaling (16-fold size change, 4.4-fold MTTF change), which indicates that small-job interruptions are dominated by causes this model does not represent as per-node hardware failures. A test pins the divergence so it cannot drift unnoticed.

**Consequence for claims:** results below roughly 4,096 accelerators are indicative only.

## 3. External validation against published measurements

### 3.1 Design

One compute-side parameter, kernel efficiency, is calibrated on a single anchor and then held fixed while the model predicts a different configuration. No network or reliability parameter is tuned at any point.

* **Calibration anchor:** 8,192 accelerators, TP 8, CP 1, PP 16, DP 64, 16M tokens per batch, published 430 TFLOP/s per accelerator.
* **Held-out target:** 16,384 accelerators, same model and batch, DP 128, published 400 TFLOP/s per accelerator and 41 percent BF16 MFU.

The held-out case doubles data parallelism at fixed global batch. That halves compute per step while spreading the gradient all-reduce across twice as many pods, so it exercises exactly the communication and scaling behavior the project depends on.

### 3.2 Results

| Quantity | Published | Model | Error | Threshold | Verdict |
|---|---|---|---|---|---|
| Held-out throughput | 400 TFLOP/s/acc | 406.3 | 1.6 percent | 10 percent | Pass |
| Held-out MFU | 41 percent | 41.06 percent | 0.1 percent | derived from above | Pass |
| Interruptions in 54 days | 419 | 526.6 | 25.7 percent | 35 percent | Pass |
| Job-level ETTR | above 90 percent | 88.8 percent | 1.2 points low | at least 85 percent | Pass |
| Calibrated kernel efficiency | not published | 0.571 | n/a | 0.50 to 0.85 | Pass |

The interruption threshold was deliberately set loose because the failure rate comes from a different cluster than the one being predicted. The rate implied by the published interruption count is 3.79e-3 per node-day, which sits inside the range the other source reports across its two clusters (2.34e-3 to 6.50e-3). The model's 25.7 percent overprediction is consistent with the target fleet being somewhat more reliable than the cluster the rate was taken from.

### 3.3 A validation failure and what it changed

The first run **failed** the ETTR check, predicting 0.600 against a published value above 0.90.

The cause was two misspecified inputs, not the model. The configuration assumed no replacement capacity, so every failure waited a full physical repair, and it assumed 60-second fully blocking checkpoint writes. Both contradict the source paper, which states the cluster held 24,000 GPUs while the job used up to 16,000, that failures were handled by automation with significant manual intervention only three times, and that per-GPU checkpoint state is 1 MB to 4 GB written to a high-throughput storage fabric.

An independent arithmetic check confirms the diagnosis. At the Daly optimum, combined checkpoint and lost-work overhead is approximately `sqrt(2 w / M)`. With a 60-second blocking write and a 2.34-hour job MTTF that is about 12 percent, so an ETTR above 90 percent is unreachable with fully blocking 60-second checkpoints at this scale regardless of any other parameter.

The inputs were corrected to 3 percent spares and a 20-second blocking write, both marked `[derived]` with their justification in the configuration file. The throughput and interruption results are unaffected, because those quantities do not depend on either parameter.

### 3.4 Excluded comparison

The source's Table 4 contains a third row listing TP 8, CP 16, PP 16, DP 4 alongside 16,384 GPUs. Those degrees multiply to 8,192. The row is internally inconsistent as printed and is excluded rather than repaired by guessing which field is wrong.

## 4. Cross-fidelity agreement

The analytical renewal path and the event-driven Monte Carlo path share only the parameter dataclass. Compared across 16 configurations spanning 2,048 to 65,536 accelerators and failure rates from 1e-3 to 3e-2:

| Region | Configurations | Maximum relative error in useful capacity |
|---|---|---|
| Inside validity envelope | 13 | 1.4 percent |
| Outside validity envelope | 3 | 29.6 percent |

The envelope is defined by **recovery pressure** below 0.25, where recovery pressure is the share of runtime consumed by discarded work plus restart. The threshold was set from this data and is pinned by a test that fails if the analytical path ever drifts more than 5 percent while still declaring itself valid.

Out-of-envelope rows remain in the raw output with a flag and are excluded from headline claims. Where such a case is load-bearing it is recomputed with the event-driven path: the claim that no accelerator count matches an infrastructure improvement at approximately 131,000 accelerators was confirmed that way, with productive throughput falling from 29,682 to 25,016 as the pool grew by half again.

## 5. Validated, extrapolated, and unsupported scope

**Validated.** Communication and throughput modeling for dense transformer training at 8,192 and 16,384 accelerators on a three-tier RoCE fabric with an 8-accelerator scale-up domain, 3,072-accelerator pods, and 1:7 cross-pod oversubscription. Job MTTF at 16,384 and above. Checkpoint optimization behavior. Accounting closure everywhere.

**Extrapolated.** Scales from 1,024 to 131,072 accelerators outside the two validated points. Bandwidths other than 400 Gbps. Oversubscription ratios other than 1:7. Failure rates other than the two published values. Every substitution-equivalent accelerator figure, since no published source reports the counterfactual the metric requires.

**Unsupported.** Any absolute cost claim. Any inference result. Any cross-campus result. Any claim about correlated or cascading failures. Any claim about jobs below roughly 4,096 accelerators. Any claim about non-transformer workloads, mixture-of-experts routing, or asynchronous training.

## 6. Consequences for each headline claim

| Claim | Status | Rests on |
|---|---|---|
| Useful capacity fraction is far below ETTR, by 26 points at 16,384 accelerators | Validated inputs, derived output | Validated throughput and MTTF at this exact configuration; the gap follows from the definitions |
| Productive share falls from 0.78 to 0.38 between 1,024 and 65,536 accelerators | Partly extrapolated | Validated at 8,192 and 16,384; the endpoints are extrapolation, and the 1,024 figure is optimistic per the small-job limitation |
| The best marginal investment changes with regime; four different interventions win | Extrapolated | No published counterfactual exists. Robust across the uncertainty analysis, but this is a model result |
| The informal equivalent-accelerator metric understates value by 1.4x to 6.2x | Extrapolated, but structural | Follows from the marginal productivity of an added accelerator, which is a property of the validated scaling behavior |
| Recovery interventions are complements, not substitutes | Extrapolated | Mechanism is explicable and consistent across regimes, but unmeasured |
| Beyond roughly 131,000 accelerators no purchase matches an infrastructure improvement | Extrapolated, confirmed at high fidelity | Outside the fast path's envelope; recomputed with the event-driven path. Depends on the strong-scaling assumption |

## 7. What would change the conclusions

* A multi-node hardware measurement of exposed communication as a share of step time would move the communication model from "validated against one published system" to "measured".
* Published failure traces including non-hardware interruptions would fix the small-job limitation.
* Any published counterfactual, a system measured before and after a network or reliability change, would allow the substitution metric itself to be validated rather than only its inputs.
