# Literature and White-Space Analysis

Search date: 2026-08-05. Every work listed was opened at the level noted in `SOURCES.md`. Works marked "abstract-level" have had only the abstract or an authoritative summary read; none of their numbers are used in a headline claim without a full read first.

## Capability columns

The four capabilities that matter for this project:

- **COMM**: models communication time as a function of network parameters (bandwidth, topology, oversubscription, collective algorithm).
- **FAIL**: models failures, detection, recovery, checkpointing, and their effect on delivered work.
- **COST**: compares interventions on a cost or investment basis.
- **SUBST**: expresses an improvement as an equivalent change in accelerator count, or otherwise supports the question "is this worth more than N more accelerators".

## Nearest-neighbor matrix

| Work | Research question | Workload | Method | Network fidelity | COMM | FAIL | COST | SUBST | Validation scale | Public impl. | Difference from this project |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ASTRA-sim 3.0 (S8) | How do ML systems perform across HW/SW/network co-design? | Training and inference | Simulation, multi-fidelity | High (analytical, congestion-aware, packet backends); adds InfraGraph and cache-line-granularity GPU model | Yes | No | No | No | Component-level and published-system comparisons | Yes | It answers "how fast", not "what should I buy next". No failure or investment axis. |
| SimAI (S9) | Same, with Alibaba production alignment | Training and inference | Analytical + ns-3 packet | High | Yes | No | No | No | Vendor-reported cluster alignment | Yes, Apache-2.0 | Same gap. Analytical mode is a good cross-check backend for our comm path. |
| Calculon (S10) | What system designs are optimal for LLM training/inference? | Transformer training and inference | Closed-form analytical | Medium (bandwidth and hierarchy, not topology-level congestion) | Yes | No | Partially (efficiency and scalability framing) | No | Compared against published system performance | Yes | Closest performance-model neighbor. Optimizes hardware and parallelism jointly, does not model reliability or recovery, and does not express results as accelerator substitution. |
| Echo, ATLAHS, Charon, memoized packet-level simulation (abstract-level) | Faster or more faithful simulation of large-model training | Training | Trace-driven and packet simulation | High | Yes | No | No | No | Varies | All improve simulation fidelity or speed. None add a failure or investment axis. |
| MLSYSIM (S14) | Can first-principles dimensional models replace profiling? | ML systems generally | Analytical, "systems walls" | Medium | Yes | Not stated | No | No | Not stated | Not stated | Analytical modeling neighbor, no reliability or decision layer. |
| AI Trinity (S13) | How do compute, bandwidth, and memory trade off? | General | Conceptual framework | Conceptual | Conceptual | No | No | No | None | No | Conceptual sibling of our thesis, without quantification, failures, or decisions. |
| AIReSim (S11) | How do failure, recovery, scheduling, and repair choices affect cluster reliability? | Single large AI job | Discrete-event (SimPy) | **None** (explicitly no topology, communication, or bandwidth modeling) | **No** | Yes | No | No | Authors state parameters are "hypothetical and not based on any observations" | Not stated | Closest reliability-side neighbor and the sharpest boundary of our white space: it does capacity-planning what-if analysis with no network model and no cost or substitution axis. |
| Meta HPCA 2025 (S2) | How reliable are large ML research clusters in practice? | Multi-tenant research jobs | Measurement + closed-form ETTR model | Not modeled (failures attributed to IB links among other causes) | No | Yes | No | No | 2 clusters, 24k A100, 11 months, >150M GPU-hours | Data not public | Provides our calibration target and the closed-form availability check. Measures reality, does not evaluate interventions against accelerator purchase. |
| ByteRobust / SOSP 2025 (S5, abstract-level) | How to keep ETTR high in production LLM training? | Production LLM training | System + deployment measurement | Fault localization includes network faults | Partially | Yes | No | No | 9,600 GPUs, 3 months, 97% ETTR | No | Builds a robustness system and measures it. Not a comparative investment model. |
| Straggler what-if analysis, OSDI 2025 (S6, abstract-level) | How much do stragglers cost, counterfactually? | Production LLM training | Trace-driven what-if simulation | Trace-implicit | Partially | Partially | No | No | 5-month ByteDance trace | No | Methodological precedent for counterfactual accounting. Scoped to stragglers, single organization, no network or purchase comparison. |
| Crux, SIGCOMM 2024 (S16) | Can communication scheduling raise GPU utilization? | Multi-tenant DL training | Scheduler + testbed + trace simulation | High (contention-aware) | Yes | No | No | No | 96-GPU testbed plus production traces | Partially | Demonstrates the comm-to-utilization link we rely on. Optimizes a scheduler rather than comparing infrastructure investments. |
| Rail-only, Ghobadi et al. (S15) | Can LLM networks be built more cheaply at equal performance? | LLM training | Traffic analysis + simulation | High | Yes | No | **Yes** (38-77% network cost, 37-75% power) | No | Simulation with published model configs | Partially | The closest cost-aware neighbor and the dual of our question: it minimizes network cost at iso-performance, while we compare marginal network, reliability, and accelerator spend at iso-budget. No failure modeling. |
| Survey of end-to-end modeling and TCO (S12) | What is the state of workload, simulator, and TCO modeling? | Training | Survey | Survey | Survey | Survey | Yes (TCO models catalogued) | No | N/A | N/A | Confirms the taxonomy and confirms that TCO models and performance simulators are catalogued separately, which is itself evidence for the gap. |
| Decoupled DiLoCo (S17), OFC 120 km field trial (S18) | Can training run across regions at low WAN bandwidth? | Cross-region training | System + measurement | WAN bandwidth as the variable | Yes | Partially (resilience framing) | No | No | 12B across 4 regions at 2-5 Gbps; 175B across 120 km at 800 Gb/s | Partially | Establishes the cross-campus regime that our optional extension probes. Algorithmic contributions, not capacity accounting. |
| Recovery systems: TrainMover, Gemini, FFTrainer, ElasWave, LowDiff (S19, abstract-level) | How fast can training recover from failure? | Training | Systems | Varies | Partially | Yes | No | No | Varies | Varies | Each reduces one recovery cost. They supply realistic intervention parameter ranges. None compares recovery investment against network or accelerator investment. |

## What the matrix shows

Reading down the capability columns:

- **COMM without FAIL**: ASTRA-sim, SimAI, Calculon, Echo, ATLAHS, Charon, Crux, Rail-only, MLSYSIM. This is the large majority of the simulation literature.
- **FAIL without COMM**: AIReSim, Meta HPCA, ByteRobust, the recovery systems literature.
- **COST**: only Rail-only (network capital and power at iso-performance) and the TCO models catalogued in the survey (which are not coupled to failure behavior).
- **SUBST**: none of the inspected works.

The cells that are empty in every row are the combination of all four. No inspected work models communication and failure behavior in one accounting, prices the interventions against each other, and expresses the result as a substitution against accelerator count.

## White-space statement

> We found no prior public work that (a) decomposes accelerator-time into mutually exclusive productive, blocked, discarded, and unavailable fates within a single model that is simultaneously sensitive to network parameters and to failure, detection, and recovery parameters; (b) defines the value of an infrastructure intervention as the equivalent change in accelerator count required to deliver the same productive accelerator-time under the same configuration; and (c) uses that definition to map the operating regimes in which bandwidth, reliability, recovery speed, or accelerator count is the preferred marginal investment.

This is deliberately narrow. It does not claim novelty for communication modeling, for failure modeling, for ETTR, for goodput, or for the observation that networks affect training performance. All of those are established and are cited as such.

## Gate 1 assessment

**Gate 1 (novelty): PASS, with a named vulnerability.**

The contribution remains distinguishable from its nearest neighbors: AIReSim on the reliability side has no network model and no cost or substitution axis and states its parameters are hypothetical; Calculon and ASTRA-sim on the performance side have no failure model and no substitution axis; Rail-only prices networks but at iso-performance with no failure model; Meta HPCA measures reliability but does not evaluate interventions.

The named vulnerability, carried forward to `RESEARCH_CHARTER.md` and to the paper's threats section: a reviewer may argue that the four-fate decomposition is a repackaging of ETTR times MFU. Answering that requires showing numerically where the composition of ratios breaks, which is why experiment E5 (naive additivity error) is a required experiment rather than an optional one.

## Sources searched and found empty

Queries run on 2026-08-05 that returned no work occupying the intersection: "equivalent GPU capacity network reliability", "GPU-equivalent capacity", "effective capacity translate accelerator units", "network bandwidth versus buying more GPUs tradeoff decision framework", "what-if capacity planning goodput checkpoint failure simulator". Absence of results in these searches is recorded as weak evidence only, consistent with the mandate's instruction not to treat a limited search as confirmation of a gap. The stronger evidence is the per-work capability audit above, which is based on reading the works themselves.
