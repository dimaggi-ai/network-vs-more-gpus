# Evidence Ledger

All sources below were opened and inspected on 2026-08-05 unless noted. Each entry lists what claims it supports in this project. Vendor material is marked as such and is used for motivation only, never as independent evidence.

Status codes: [P] peer-reviewed or conference-accepted, [A] arXiv preprint, [V] vendor material, [C] code repository.

## Motivating source

### S1. Google Cloud, "AI infrastructure: Data center and global networks built for the AI era" [V]
- URL: https://cloud.google.com/blog/products/networking/data-center-and-global-networks-built-for-ai-era
- Accessed 2026-08-05.
- Vendor blog. Claims inspected: three network domains (scale-up, scale-out/Virgo fabric with 47 Pb/s claim, frontend/Jupiter); synchronized millisecond bursts and tail-latency sensitivity; automated hang detection and sub-millisecond telemetry; cross-campus pooling to overcome per-site power limits; WAN growth 10x 2020-2025; a verifiable arithmetic example (100 Gbps to 3.2 Tbps cuts a 1 PB transfer from 22.2 h to 0.7 h).
- Use in project: motivation and framing only. None of its reliability or pooling claims are treated as evidence.

## Measurement studies (calibration and validation data)

### S2. Kokolis et al. (Meta), "Revisiting Reliability in Large-Scale Machine Learning Research Clusters", HPCA 2025 [P]
- arXiv: https://arxiv.org/abs/2410.21680 (HTML v2 inspected)
- Key extracted facts (quoted from paper):
  - ETTR defined as "the ratio of productive runtime to available wallclock time of a job run".
  - Large job runs (>1024 GPUs) on RSC-1 show ETTR around 0.9.
  - MTTF of 1024-GPU jobs is 7.9 hours; projected MTTF 1.8 h at 16,384 GPUs and 0.23 h at 131,072 GPUs.
  - Failure rate r_f approximately 5e-3 failures per GPU-node-day (RSC-1: 6.50 per thousand node-days; RSC-2: 2.34).
  - Top failure contributors: IB links, filesystem mounts, GPU memory errors, PCIe errors.
  - Closed-form expectation: E[ETTR] ~= (1 - N_nodes * r_f * (u0 + dt_cp/2)) / (1 + w_cp / dt_cp), with u0 restart overhead, w_cp checkpoint write time, dt_cp checkpoint interval.
  - Scale: 2 clusters (16k + 8k A100), 11 months, 4M jobs, >150M A100 GPU-hours.
- Use in project: primary calibration target for the availability model; the E[ETTR] formula is a Gate 2 reproduction target.

### S3. Grattafiori et al. (Meta), "The Llama 3 Herd of Models", 2024 [A]
- arXiv: https://arxiv.org/abs/2407.21783 (ar5iv HTML inspected)
- Key extracted facts:
  - 405B model pre-trained on up to 16K H100 GPUs (700W TDP, 80GB HBM3).
  - Network: RoCE fabric, 400 Gbps between GPUs, Arista switches; three-layer Clos; pods of 3,072 GPUs with full bisection; 1:7 oversubscription between pods.
  - 54-day snapshot: 466 total interruptions, 419 unexpected; 78% of unexpected attributed to confirmed hardware; GPU issues 58.7% of all issues.
  - Higher than 90% effective training time.
  - BF16 MFU 38-43%; 41% at full 16K scale.
  - 4D parallelism (TP, PP, CP, DP).
- Use in project: primary joint calibration target (communication model MFU + availability model ETTR at a public large-scale config).

### S4. Jiang et al. (ByteDance), "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs", NSDI 2024 [P]
- arXiv: https://arxiv.org/abs/2402.15627 (abstract inspected)
- Key facts: 55.2% MFU training a 175B model on 12,288 GPUs, 1.34x over Megatron-LM baseline; production straggler/failure diagnosis experience.
- Use in project: upper-bound check for the communication model on a heavily optimized stack; qualitative failure taxonomy.

### S5. ByteDance, "Robust LLM Training Infrastructure at ByteDance" (ByteRobust), SOSP 2025 [P]
- arXiv: https://arxiv.org/abs/2509.16293; ACM DOI 10.1145/3731569.3764838 (search-level inspection; full PDF to be read before citation in paper)
- Key facts: 97% cumulative ETTR for a three-month job on 9,600 GPUs; defines cumulative and sliding-window (1 h) ETTR; hierarchical detection and demarcation approach.
- Use in project: availability model validation point; ETTR metric variants.

### S6. Lin et al., "Understanding Stragglers in Large Model Training Using What-if Analysis", OSDI 2025 [P]
- arXiv: https://arxiv.org/abs/2505.05713 (search-level inspection; full paper to be read before citation)
- Key facts: five-month ByteDance trace; what-if simulation of straggler-free runs; 42.5% of jobs at least 10% slower due to stragglers; tail jobs waste up to 45% of allocated resources.
- Use in project: magnitude prior for straggler/fail-slow degradation factor; methodological precedent for counterfactual what-if framing.

### S7. Lablup, "From Detection to Recovery: Operational Analysis on LLM Pre-training with 504 GPUs", 2026 [A]
- arXiv: https://arxiv.org/abs/2605.09370 (search-level inspection)
- Key facts: 63-node B200 cluster, 55 days telemetry, 224 sessions.
- Use in project: mid-scale reliability datapoint; shows failure behavior below hyperscale.

## Simulators and performance models (tool candidates and neighbors)

### S8. Won et al., "ASTRA-sim 3.0: Next-Level Distributed Machine Learning Simulations", 2026 [A]
- arXiv: https://arxiv.org/abs/2606.10440; project site https://astra-sim.github.io/
- Adds high-fidelity GPU execution model, InfraGraph infrastructure representation, latency-sensitive collective modeling. No failure, reliability, checkpoint, or cost modeling stated.
- Use in project: candidate cross-model validation backend; nearest neighbor on the simulation side.

### S9. Alibaba, SimAI (NSDI 2025) [P][C]
- GitHub: https://github.com/aliyun/SimAI (README inspected; Apache-2.0; SimAI 1.6 released April 2026, active)
- Components: AICB workload generator, SimCCL, astra-sim-alibabacloud analytical engine, ns-3-alibabacloud packet engine. Analytical mode uses bus-bandwidth abstraction. No failure or cost modeling in docs.
- Use in project: candidate cross-model validation backend (analytical mode); evidence that public simulators stop at performance.

### S10. Isaev, McDonald, Dennison, Vuduc, "Calculon: a Methodology and Tool for High-Level Codesign of Systems and Large Language Models", SC 2023 [P][C]
- ACM DOI 10.1145/3581784.3607102 (abstract and metadata inspected)
- Parameterized analytical performance model of transformer training/inference; joint hardware-software design-space search; open-source Python tool.
- Use in project: closest performance-model neighbor; planned cross-model check for communication/MFU predictions. Does not model failures, recovery, or investment decisions.

### S11. Pattabiraman, Patel, Lin, "AIReSim: A Discrete Event Simulator for Large-scale AI Cluster Reliability Modeling", DSN 2026 industry track [A]
- arXiv: https://arxiv.org/html/2603.07041v1 (full HTML inspected)
- SimPy DES of server failures, repair queues, spares, warm standby; capacity-planning what-if case study.
- Explicitly does NOT model network topology, communication time, or bandwidth effects; does NOT convert reliability to dollars or GPU-equivalents; parameters "hypothetical and not based on any observations"; single-job assumption; no stated public code.
- Use in project: closest reliability-side neighbor; defines the boundary of our white space.

### S12. Svedas et al., "A Survey of End-to-End Modeling for Distributed DNN Training: Workloads, Simulators, and TCO", 2025 [A]
- arXiv: https://arxiv.org/abs/2506.09275
- Surveys workload representation, simulators, TCO/emissions models.
- Use in project: positioning and related-work coverage; source of the simulator/TCO taxonomy.

### S13. Fan, Weng, Li, "Computation-Bandwidth-Memory Trade-offs: A Unified Paradigm for AI Infrastructure" (AI Trinity), Dec 2025 [A]
- arXiv: https://arxiv.org/abs/2601.11577
- Conceptual three-way tradeoff framework. No failures, reliability, cost, or capacity-equivalence modeling in abstract.
- Use in project: related conceptual work; distinguishes our quantitative decision layer.

### S14. Janapa Reddi, "MLSYSIM: First-Principles Infrastructure Modeling for Machine Learning Systems", June 2026 [A]
- arXiv: https://arxiv.org/abs/2607.02558
- Dimensional-analysis engine, 22 "Systems Walls", 28 composable models. No failure/reliability/investment modeling stated.
- Use in project: related analytical-modeling work.

## Network architecture and cost

### S15. Wang, Ghobadi, Shakeri, Zhang, Hasani, "Rail-only: How to Build Low-Cost Networks for Large Language Models", 2023/2024 [A]
- arXiv: https://arxiv.org/abs/2307.12169
- Rail-only design achieves same training performance while cutting network cost 38-77% and network power 37-75%; LLM traffic is sparse and does not need any-to-any full bisection; MoE all-to-all overhead 8.2-11.2%.
- Use in project: relative network-cost anchors; the iso-performance cost-reduction dual of our question; evidence for traffic sparsity assumptions.

### S16. Cao et al. (Alibaba), "Crux: GPU-Efficient Communication Scheduling for Deep Learning Training", SIGCOMM 2024 [P]
- https://dl.acm.org/doi/10.1145/3651890.3672239 (search-level inspection)
- 36.3% of DLT jobs may experience communication contention; scheduling raises GPU utilization up to 23% in trace-driven simulation, 8.3-14.8% on a 96-GPU testbed.
- Use in project: evidence that network contention converts directly into GPU utilization loss (single-campus, multi-tenant).

## Cross-campus / WAN training

### S17. Google DeepMind, "Decoupled DiLoCo: Resilient, Distributed AI Training at Scale", blog, April 2026, with paper arXiv:2604.21428 [V + A]
- https://deepmind.google/blog/decoupled-diloco/ (inspected)
- 12B model trained across four US regions over 2-5 Gbps WAN; asynchronous outer loop; claimed >20x faster than conventional synchronization at that bandwidth.
- Use in project: establishes the low-bandwidth cross-campus regime; motivates treating WAN pooling as a separate operating regime rather than an extension of synchronous scaling.

### S18. OFC 2025 field trial, "Multi-Datacenter Distributed Training for LLM ... over 120km 800Gbit/s C+L OTN" [P]
- https://opg.optica.org/abstract.cfm?uri=ofc-2025-Th1A.3 (search-level inspection)
- 175B-parameter training across two DCs 120 km apart; up to 99.41% (PP) and 98.95% (DP) relative training efficiency.
- Use in project: shows high-bandwidth metro pooling can be near-lossless; boundary datapoint between metro and WAN regimes.

## Recovery systems (context for intervention parameters)

### S19. TrainMover (UCCL project blog, inspected via search 2026-08-05), Gemini (SOSP 2023, Amazon Science page), FFTrainer (arXiv 2512.03644), ElasWave (arXiv 2510.00606), LowDiff (arXiv 2509.04084) [A]
- Search-level inspection only. These systems reduce detection/restart/recompute costs (e.g., TrainMover ~20 s migration; FFTrainer claims ~98% recovery-time cut).
- Use in project: define the plausible ranges for intervention parameters (detection time, restart time, checkpoint overhead). Each will be opened fully before any specific number is cited in the paper.

## Notes

- "Search-level inspection" means the abstract or an authoritative summary was read but the full text has not yet been read end to end. Any number from such a source is provisional and must be verified against the full text before appearing in the paper. Tracked as an open item in STATUS.md.
- No source above is treated as evidence for a "first ever" novelty claim. The white-space statement is phrased as "we found no prior public work that ...", see docs/literature_matrix.md.
