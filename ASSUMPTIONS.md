# Assumptions

Every quantity in the model is one of: **published** (traceable to a primary source in `SOURCES.md`), **derived** (computed from a published quantity, with the computation stated), **calibrated** (fitted to a published measurement, with the fit and the held-out test stated), or **assumed** (a modeling choice with a documented range and a sensitivity result).

No headline claim rests on an assumed value that was not swept.

## Assumptions carried into the results

| ID | Assumption | Default | Swept range | Basis and effect |
|---|---|---|---|---|
| A1 | Kernel efficiency, the fraction of dense peak achieved by compute kernels | 0.571 | 0.45 to 0.70 | Calibrated on the 8,192-accelerator anchor (D6). Affects absolute throughput but not the ranking of interventions, since it is held constant across every intervention. |
| A2 | Achievable fraction of link line rate for collectives | 0.85 | 0.70 to 0.95 | Standard range for well-tuned collective libraries. Swept in E7. |
| A3 | Data-parallel collective overlap with backward pass | 0.85 | 0.60 to 0.95 | Overlapping the gradient all-reduce is standard practice. Swept in E7. |
| A4 | Tensor-parallel collective overlap | 0.10 | 0.0 to 0.35 | Tensor-parallel collectives sit on the critical path and overlap poorly. Swept in E7. |
| A5 | Per-rank timing jitter, coefficient of variation | 0.02 | 0.0 to 0.05 | Chosen to produce roughly a 9 percent synchronization tax at 16,384 accelerators, matching the reported magnitude that 42.5 percent of production jobs run at least 10 percent slower from stragglers (S6). Swept in E7. |
| A6 | Rank placement is dense and regular: consecutive ranks fill a scale-up domain, then a pod | dense | not swept | The best case for the network. Fragmented placement would make every network intervention look better, so this assumption is conservative with respect to the project's own thesis. |
| A7 | Failure detection time | 120 s | 30 to 400 s | No public source states it for the validated system. Swept in E7; "fast detection" intervention sets 20 s. |
| A8 | Restart and reinitialization time | 300 s | 60 to 900 s | Recovery-systems literature reports figures from tens of seconds upward. Swept in E7. |
| A9 | Node repair time when no spare is available | 3600 s | not swept | Matters only through spare availability; with the default 3 percent spares the probability a spare is free is essentially one, so the value is not load-bearing. |
| A10 | Checkpoint interval | Daly optimum | also fixed values | The first-order optimum is used unless a value is given. The model recovers Daly's formula by numerical optimization, which is a validation check rather than an assumption. |
| A11 | Latency terms: 2 us scale-up, 6 us intra-pod, 15 us cross-pod | as stated | not swept | At the message sizes in these workloads, bandwidth dominates latency; the latency terms change step time by well under a percent. |
| A12 | Attention FLOPs use a causal factor of 0.5 | 0.5 | not swept | Standard for causal decoder-only models. Affects the FLOP denominator identically for all configurations. |
| A13 | Strong scaling: global batch is held fixed as the pool grows | strong | weak scaling tested in E4 | This is the policy for "buy more accelerators to finish the same run sooner". Weak scaling is included as a counterexample case. |
| A14 | Interruptions follow a per-node Poisson process | Poisson | rate swept 1e-3 to 3e-2 | See the limitation below; this is the model's weakest assumption. |
| A17 | Gradient reduction spans the data-parallel group only | cp groups excluded | not applicable | With context parallelism the reduction would also span the cp group. Latent: cp = 1 in every shipped configuration; recorded so a future cp > 1 study does not inherit it silently. |

## Assumptions contradicted by evidence and corrected

| ID | Original assumption | Evidence against it | Resolution |
|---|---|---|---|
| A15 | No spare capacity, so each failure waits a full repair | The source paper states the cluster held 24,000 GPUs while the job used up to 16,000, and that failures were handled by automation with manual intervention only three times in 54 days | Set to a 3 percent spare fraction, marked `[assumed]` since the exact reserve is not published. See D5. |
| A16 | 60-second fully blocking checkpoint writes | The source paper states per-GPU checkpoint state is 1 MB to 4 GB written against a high-throughput storage fabric, and that they minimize pause time. Independently, a 60-second blocking write makes the published ETTR arithmetically unreachable at this MTTF | Set to 20 seconds, marked `[assumed]`, bounded by the paper's stated 1 to 4 GB per GPU. See D5. |

## Known limitation: the failure process

The model treats job-stopping interruptions as a per-node Poisson process with a rate taken from published hardware failure data. Two pieces of evidence bound how far that can be trusted.

**It reproduces large-scale behavior well.** Using the RSC-1 measured rate of 6.50 failures per thousand node-days, the model reproduces Meta's reported job MTTF at 16,384 accelerators (1.803 hours predicted against 1.8 reported) and at 131,072 accelerators (0.225 against 0.23).

**It fails at small scale, and Meta's own numbers show why.** Their measured 1,024-GPU MTTF of 7.9 hours is roughly 3.7 times shorter than inverse scaling from the same per-node rate predicts. Their own reported figures are not mutually consistent under inverse scaling: 1,024 to 16,384 accelerators is a 16-fold change in size but only a 4.4-fold change in reported MTTF. Whatever dominates interruptions for smaller jobs is not per-node hardware failure, and this model does not represent it.

**Consequence.** The model is calibrated and validated for large synchronous jobs and is optimistic for small ones. Results below roughly 4,096 accelerators should be read as indicative only. This is asserted by a test that fails if the divergence changes, so the limitation cannot silently drift.

The model also assumes failures are independent. Correlated failures, from a shared power event, a rack-level fault, or a bad firmware rollout, would produce worse outcomes than modeled and would raise the value of fault-isolation interventions that this project does not represent.

## What no assumption can rescue

The model has never been run against hardware. Every number in this project is analytical or simulated. The external validation compares against published measurements of systems the author did not operate, using configurations reconstructed from papers. Where a published value depends on a parameter the paper does not report, the comparison tests the joint plausibility of the model and that parameter, not the model alone. This is stated in the validation report per claim.
