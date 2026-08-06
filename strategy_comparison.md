# Strategy Adjudication

Date: 2026-08-05. Adjudicator: project lead (autonomous execution).

## What was supplied

Three items were supplied, not three competing strategies:

1. An **execution mandate** describing process, integrity rules, phases, and gates. Classification: process instruction, not a research proposal. Accepted in full.
2. A **strategy assignment** containing the central idea, the proposed metric ("Equivalent Accelerator Capacity Recovered"), researcher context and constraints, and **six candidate first-paper scopes**. Classification: proposal. The six scopes are the actual competing strategies and are adjudicated below.
3. A **motivating source**: the Google Cloud networking blog post. Classification: background and motivation. It is vendor material. Every claim in it is treated as a vendor claim (see `SOURCES.md` S1).

There were no contradictions between items 1 and 2. Item 2 contains two assumptions that the evidence check contradicted or qualified; these are recorded in `ASSUMPTIONS.md` (A1, A2) and resolved in `DECISIONS.md` (D2, D3).

## Rubric scoring of the six candidate scopes

Weights are from the mandate. Scores are 1 to 10, assigned after the literature pass in `docs/literature_matrix.md`. Weighted total is the sum of score times weight.

| Scope | Importance (15%) | Novelty (20%) | Decision value (20%) | Solo feasibility (15%) | Validation feasibility (15%) | Reproducibility (5%) | Publication strength (10%) | **Weighted** |
|---|---|---|---|---|---|---|---|---|
| **A. Single-campus distributed-training communication and failures** | 9 | 5 | 6 | 8 | 9 | 9 | 6 | **6.90** |
| **B. Cross-campus pooling of power-constrained capacity** | 8 | 7 | 8 | 5 | 3 | 7 | 7 | **6.50** |
| **C. Network-aware distributed inference across heterogeneous pools** | 7 | 6 | 6 | 5 | 4 | 7 | 6 | **5.75** |
| **D. Data staging and accelerator idle time** | 5 | 4 | 4 | 8 | 6 | 8 | 4 | **5.15** |
| **E. Reliability, telemetry, and recovery as effective capacity** | 8 | 5 | 7 | 8 | 8 | 9 | 6 | **7.00** |
| **F. Combined infrastructure-investment decision framework** | 9 | 8 | 10 | 6 | 6 | 8 | 8 | **7.80** |

Scoring notes, with the evidence behind each low score:

- **A scores 5 on novelty.** Communication modeling for distributed training is a crowded field: ASTRA-sim 3.0 (S8), SimAI (S9), Calculon (S10), Echo, ATLAHS, Charon all model it, several with packet-level fidelity that a solo researcher cannot match. Failure modeling on top of communication is less crowded but AIReSim (S11) occupies the reliability half.
- **B scores 3 on validation feasibility.** The only public quantitative anchors are a DeepMind blog with an accompanying paper (S17) and one OFC field trial abstract (S18). Neither provides the parameter detail needed to calibrate a cross-campus model, and no public trace exists. A cross-campus headline claim would rest almost entirely on unvalidated extrapolation, which Gate 5 forbids.
- **C scores 4 on validation feasibility** for the same reason, compounded by inference serving having its own large and fast-moving literature (continuous batching, disaggregated prefill/decode) that a network-focused solo project would have to engage with credibly.
- **D scores 4 on novelty.** Data-loading and storage stalls are well covered in the storage systems literature and are a smaller share of lost accelerator time than communication or failures at the scales of interest.
- **E scores 5 on novelty.** Meta HPCA 2025 (S2) already provides the ETTR framework, the failure-rate data, and a closed-form availability expression. AIReSim (S11) already provides a discrete-event reliability simulator with capacity-planning what-if analysis. A reliability-only paper would be re-treading both.
- **F scores highest** because the decision layer is the part that no inspected prior work provides, and because it can be built **on top of** A and E rather than instead of them. Its risk is scope explosion, which the scope decision below controls.

## Strongest components taken from each scope

- From **A**: the alpha-beta collective communication model mapped onto a concrete parallelism strategy and a concrete oversubscribed Clos topology. This is the mechanism that connects a network parameter to a step time.
- From **E**: the availability model built on published failure rates, checkpoint interval, detection time, and restart time, calibrated to the Meta HPCA data.
- From **F**: the decision layer, the substitution metric, and the rank-reversal analysis. This is the contribution.
- From **B**: retained only as a bounded **regime-boundary extension**, not as a headline claim. See the extension definition below.

## Elements rejected, with reasons

| Rejected element | Reason |
|---|---|
| Building a new general-purpose AI-network simulator | The mandate forbids it absent evidence, and the evidence goes the other way. ASTRA-sim 3.0, SimAI, Echo, ATLAHS, and Charon all exist and are actively maintained. A solo re-implementation would be strictly worse and would consume the entire budget. |
| Packet-level network simulation as the primary path | Packet-level runs at the scales of interest (10k to 100k accelerators, multi-day jobs) are computationally infeasible for a solo researcher. The research question is about multi-day capacity accounting, not microsecond queueing. Retained only as a possible future fidelity check. |
| The proposed metric formula as supplied | The supplied expression (avoided idle-or-repeated accelerator-seconds divided by measurement period) is not a defensible equivalence. It assumes a marginal accelerator is fully productive, which contradicts the project's own premise. Replaced by a substitution-based definition. See `DECISIONS.md` D3 and `docs/metric_framework.md`. |
| Inference in version one | Rejected on validation feasibility and scope control. Recorded as excluded, not as unimportant. |
| Data staging and storage in version one | Rejected on novelty and materiality. Modeled only as a fixed stall term that can be swept in sensitivity analysis. |
| Telemetry resolution as an independent variable | Telemetry does not affect capacity directly. It affects capacity only through detection time. Modeled as detection time, which is measurable and comparable across sources. This avoids an unfalsifiable "better telemetry" variable. |
| A weighted composite score ranking interventions | The mandate asks for decision boundaries and Pareto frontiers instead, and the assignment is right that a single composite score would hide the regime dependence that is the actual finding. |
| Any framing organized around a job description | Excluded per the mandate. The contribution is a metric and a decision method, both of which stand alone. |

## Consolidated direction

**Primary scope: F built on A and E, restricted to single-campus synchronous large-model training.**

The project builds a two-path capacity-accounting model (fast analytical path, Monte Carlo discrete-event failure path) that decomposes accelerator-time into four mutually exclusive fates, and a decision layer on top of it that answers which marginal infrastructure investment yields the most productive accelerator-time. The contribution is the accounting framework, the substitution metric, the attribution rule for interacting interventions, and the resulting decision-boundary maps, not the underlying performance model, which deliberately reuses standard formulations so it can be checked against existing tools.

**Optional extension: B, narrowed.** Cross-campus pooling is admitted only as a bounded regime-boundary analysis answering one question: under a per-site power cap, at what WAN bandwidth and synchronization frequency does pooling across two sites produce more productive accelerator-time than leaving capacity stranded at one site. It is explicitly labeled as extrapolation with weak validation and is excluded from headline claims.

**Excluded from version one:** inference serving, data staging and storage systems, autoscaling and multi-tenant scheduling, packet-level congestion control research, optical and physical-layer design, carbon and power modeling beyond a per-site cap, and any absolute currency-denominated cost claim.

## Largest remaining uncertainty

Whether the substitution metric produces **rank reversals** across realistic operating regimes. If every regime ranks the same intervention first, the decision layer collapses to a rule of thumb and the paper's contribution weakens to an accounting framework plus a validation study. This is a real possibility and is a predefined narrowing condition, not a failure to hide. It is tested early, in experiment E4, before the paper is drafted.

## Contribution most vulnerable to prior-art challenge

The **capacity-accounting decomposition** is the most vulnerable. ETTR (S2, S5), goodput, and MFU are established, and a reviewer will reasonably ask what the four-fate decomposition adds beyond ETTR times MFU. The defense must be specific: ETTR and MFU are ratios defined over different denominators and do not compose across loss sources without double counting, and neither supports counterfactual substitution against accelerator count. That defense has to be demonstrated numerically, not asserted, which is why quantifying the naive-additivity error is a required result rather than an optional one.

## Novelty-validation plan

1. Systematic search across the simulator, reliability, goodput, cost-modeling, and network-architecture literatures. Done, recorded in `docs/literature_matrix.md`.
2. For each nearest neighbor, record explicitly which of the four required capabilities it has: communication modeling, failure and recovery modeling, cost or investment comparison, and accelerator-count substitution. The white space is the intersection that is empty.
3. Phrase the claim as "we found no prior public work that ..." rather than "first". Enforced in the paper checklist.
4. Re-run the search before submission and record the date, since this literature moves fast.
5. Treat any newly found work occupying the intersection as a Gate 1 failure and pivot toward the validation-and-accounting contribution, which survives even if the decision layer is anticipated.
