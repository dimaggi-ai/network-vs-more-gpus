# Research Charter

Status: active. Written 2026-08-05, after Phase 0 intake and Phase 1 literature analysis, before the headline experiments were run. Amendments are recorded in `DECISIONS.md` with dates.

## Working title

**Network capacity is sometimes compute capacity: an accounting framework and decision boundaries for infrastructure investment in large-scale AI training**

The supplied title, "Network Capacity Is Compute Capacity", is treated as a hypothesis to be tested rather than a conclusion to be demonstrated. The results determine the final title. As of the first full experiment run the evidence supports a qualified version: network capacity substitutes for compute capacity in identifiable regimes and not in others, and at the most common operating points studied here it is not the strongest marginal investment.

## Problem statement

Operators of large accelerator fleets must repeatedly choose between buying more accelerators and improving the infrastructure around the accelerators they already own. The metrics in common use do not support that comparison. Model FLOPs Utilization measures how well a step runs but ignores time lost to failures. Effective Training Time Ratio measures availability but credits communication stall as productive. Neither expresses an infrastructure improvement in units comparable to an accelerator purchase, so the comparison is usually made informally.

## Central research question

For a large synchronous training job, when does an improvement to network bandwidth, topology, failure rate, failure detection, recovery speed, or checkpointing deliver more productive accelerator time than spending the same budget on additional accelerators, and where are the boundaries between those regimes?

## Secondary questions

1. How is paid-for accelerator time actually distributed between productive work, blocked time, discarded work, and unavailable capacity, and how does that distribution change with scale?
2. How large is the error in the informal practice of converting avoided idle time directly into an equivalent accelerator count?
3. Are infrastructure interventions additive in their effect, and if not, in which direction and by how much?
4. Under parameter uncertainty, how stable is the ranking of interventions?
5. Are there regimes in which no accelerator purchase can reproduce the effect of an infrastructure improvement?

## Falsifiable hypotheses

* **H1** As job size grows at fixed global batch, the productive share of paid-for accelerator time falls, and the dominant loss shifts from communication to failure-related causes.
  *Status: supported.* Productive share falls from 0.781 at 1,024 accelerators to 0.382 at 65,536. Exposed communication stays near 8 to 10 percent of the total throughout while pipeline bubble and restart time grow from about 1 percent combined to about 29 percent.
* **H2** The ranking of interventions by substitution value is not constant across operating regimes.
  *Status: supported.* Four distinct interventions rank first across the 96 in-envelope cells of the decision map, and the set of winners differs between 16,384 and 65,536 accelerators.
* **H3** Interventions are substitutes, so the sum of their individual substitution values exceeds the value of the bundle.
  *Status: partially refuted.* Network interventions are substitutes as predicted (mean additivity error +6.5 percent), but recovery interventions are complements (mean -8.9 percent): detection, restart, and checkpoint improvements are together worth more than the sum of their parts. Reported as a null result against the stated direction.
* **H4** The informal equivalent-accelerator metric is biased, and the bias grows with scale.
  *Status: supported.* The informal figure is 0.73 of the substitution value at 2,048 accelerators and 0.16 at 65,536, an understatement of 1.4x rising to 6.2x.
* **H5** There exist regimes in which no accelerator count reproduces an intervention's effect.
  *Status: supported, outside the fast path's validity envelope and therefore confirmed with the event-driven path.* At approximately 131,000 accelerators under the study baseline, productive throughput decreases as the pool grows, so the substitution has no solution.

## Intended contributions

1. A four-fate decomposition of accelerator-seconds (productive, blocked, discarded, unavailable) that is mutually exclusive and collectively exhaustive by construction and is checked numerically on every model evaluation.
2. Substitution-Equivalent Accelerators, a metric that expresses an infrastructure change as the accelerator count delivering equal productive throughput, together with the demonstration that the informal alternative is biased and by how much.
3. An order-independent attribution rule for interacting interventions, and the measurement of how far from additive they are.
4. Decision-boundary maps identifying which intervention is the strongest marginal investment in each regime, with the negative cases stated as prominently as the positive ones.
5. An open implementation with two independent fidelities, an explicit validity envelope, and external validation against published measurements.

The contribution is **not** a new simulator, a new communication model, a new reliability model, or a claim that networks affect training performance. All of those exist and are cited.

## Unit of analysis

One large synchronous training job on a dedicated pool of accelerators over a measurement window, with the pool defined as the accelerators paid for rather than the accelerators the job holds.

## Variables

**Independent.** Accelerator count; per-accelerator scale-out bandwidth; scale-up domain size; cross-pod oversubscription; node failure rate; failure detection time; restart time; checkpoint write cost; spare fraction; straggler coefficient of variation; parallelism degrees; global batch; sequence length.

**Dependent.** Useful Capacity Fraction; productive accelerators; substitution-equivalent accelerators; break-even cost; the four-fate shares; Model FLOPs Utilization and Effective Training Time Ratio, computed for comparison with published values.

**Controlled.** Kernel efficiency is held fixed across every intervention, so that no network or reliability result can be contaminated by an implicit change in compute efficiency. Model shape and precision are fixed within a sweep.

## Baselines

* The unmodified study configuration at each scale and regime.
* Additional accelerators at the baseline configuration, which is the alternative every intervention is measured against.
* The informal equivalent-accelerator metric, retained specifically as a comparison target.
* Published Model FLOPs Utilization and interruption counts, as external anchors.

## Ablations

Interventions applied individually, in bundles, and in all subsets for the Shapley computation; interventions applied where the corresponding loss has already been eliminated (perfect network, perfect reliability), which is how the zero-value cases are produced.

## Validation approach

Four layers, reported in `validation/VALIDATION_REPORT.md`:

1. Invariants: time conservation, non-negativity, limiting cases, monotonicity in bandwidth, failure rate, and oversubscription.
2. Independently derived closed forms: Daly's optimal checkpoint interval recovered by numerical optimization of the model, and the Meta HPCA 2025 expected-ETTR expression.
3. External measurement: kernel efficiency calibrated on the published 8,192-accelerator Llama 3 configuration, then used unchanged to predict the 16,384-accelerator configuration; job MTTF compared against Meta's reported projections.
4. Cross-fidelity: the analytical renewal path against the event-driven Monte Carlo path, which defines the validity envelope.

## Workload scope

Dense decoder-only transformer pre-training, synchronous, with tensor, pipeline, context, and data parallelism. Sequence lengths from 2,048 to 32,768. Model sizes anchored on a 405B-parameter configuration.

## Infrastructure scope

A single campus: a scale-up domain, a full-bisection pod, and an oversubscribed cross-pod layer. Node-level failures with detection, restart, repair, and spares.

## Non-goals

Inference serving. Data staging and storage systems beyond a checkpoint write cost. Multi-tenant scheduling and autoscaling. Packet-level congestion control. Optical and physical-layer design. Carbon and power modeling beyond a per-site cap in the optional extension. Absolute currency-denominated cost claims. Any product, service, or commercial framing.

## Threats to validity

* **Construct.** The four-fate decomposition classifies by terminal fate, which is a choice; an activity-based classification would attribute differently. The substitution metric depends on how parallelism is re-planned as the pool grows, and strong scaling at fixed global batch is the assumed policy.
* **Internal.** The straggler term uses a Gaussian extreme-value approximation. The failure process is per-node Poisson, which underestimates correlated failures and, as Meta's own 1,024-GPU measurement shows, misses whatever causes small-job interruptions to exceed the hardware rate by roughly a factor of four.
* **External.** Calibration rests on one published system. The model is validated for large synchronous jobs and is expected to be optimistic for small ones.
* **Conclusion.** Rankings depend on parameter ranges that are themselves assumptions, which is why rank stability under uncertainty is reported rather than a single ranking.

## Required compute

None beyond a laptop. The full experiment program runs in about ten seconds; the test suite in under a second. No accelerator access is used, and no result in this project is a hardware measurement.

## Expected figures

Seven, all generated from immutable raw results by `figures/make_figures.py`: the capacity ledger against scale, scaling and marginal productivity, the decision map, substitution value by regime, the informal-metric bias, rank stability under uncertainty, and cross-fidelity agreement.

## Publication criteria

An arXiv preprint is justified if the novelty statement survives a repeated literature check, the external validation passes its predefined thresholds, and at least one rank reversal survives the uncertainty analysis. A workshop submission is justified additionally if the decision-boundary result proves robust to the parameter ranges. A stronger venue would require hardware validation, which this project does not have and does not claim.

## Go, narrow, or stop conditions

* **Go.** All met as of the first full run: novelty check passed, external validation passed, rank reversals present.
* **Narrow (N1).** If every regime ranked the same intervention first, the contribution would narrow to the accounting framework and the validation study. Not triggered.
* **Narrow (N2).** If the informal metric had turned out to be a good approximation of the substitution metric, the metric contribution would narrow to the accounting framework. Not triggered; the gap reaches a factor of 6.2.
* **Stop.** If prior work were found occupying the full intersection of communication modeling, failure modeling, cost comparison, and accelerator substitution, the project would pivot to the validation-and-accounting contribution. Not triggered as of the 2026-08-05 search; the search is to be repeated before submission.
