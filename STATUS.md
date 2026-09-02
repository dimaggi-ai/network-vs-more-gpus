# Status

Last updated: 2026-09-01.

## Current phase

**Published, with a scale-across extension added 2026-09-01.** The v1.0.0 body of
work was approved by the project owner on 2026-08-06 after a multi-iteration
quality pass and released at https://github.com/dimaggi-ai/network-vs-more-gpus.
Nothing in it has moved: the span tier added in phase 11 is inert when
`halls == 1`, and both the Llama 3 validation program and the published step
times reproduce byte-identically.

Preprint-server submission is no longer planned; that line is withdrawn rather
than pending.

## Completed

| Phase | Output |
|---|---|
| 0. Intake and adjudication | `strategy_comparison.md`, `SOURCES.md`, `DECISIONS.md`, `ASSUMPTIONS.md` |
| 1. Literature and white space | `docs/literature_matrix.md`, Gate 1 passed |
| 2. Charter | `RESEARCH_CHARTER.md`, `docs/metric_framework.md` |
| 3-4. Design and implementation | `src/netcap/`, configs, Makefile, 65 tests |
| 5. Experiments | 9 experiment families, immutable raw results with provenance sidecars |
| 6. Validation | `validation/VALIDATION_REPORT.md`, Gates 2 and 3 passed |
| 7. Analysis and figures | 7 figures, `results/processed/` |
| 8. Paper | `paper/main.tex`, compiled and inspected, 12 pages |
| 9. Article | `article/article.md`, 2,032 words |
| 10. Publication checks | See below |
| 11. Scale-across span tier | `docs/regime-atlas.md`, `src/netcap/regimes.py`, `validation/validate_span.py`, 7 span experiments, fig8-fig9, D16-D18, A18-A22, S20 |

## Decision gates

| Gate | Status | Evidence |
|---|---|---|
| 1. Novelty | Pass | Per-work capability audit in `docs/literature_matrix.md`; no inspected work combines communication modeling, failure modeling, cost comparison, and accelerator substitution. Claim worded as "we found no prior public work that", not "first". |
| 2. Baseline credibility | Pass | Daly's optimal checkpoint interval and Meta's expected-ETTR closed form both recovered; held-out throughput prediction within 1.6 percent. |
| 3. Accounting correctness | Pass | Time conservation asserted on every evaluation; limiting cases and monotonicity tested. |
| 4. Finding robustness | Pass | Rank reversals survive the uncertainty analysis; out-of-scope rows excluded; the one load-bearing out-of-scope claim computed with both implementations, which agree to 0.03 percent. |
| 5. Publication integrity | Pass | Each headline claim tagged validated, extrapolated, or unsupported in the validation report. No absolute cost claim anywhere. |
| 6. Span-tier evidential honesty (phase 11) | Pass | 16-point registry split 2 calibrated / 7 emergent / 7 sanity, with a printed 10-entry DECLINED list. Seven mutation tests delete span machinery and require the registry to go red, each printing its actual red set; `test_registry_blind_spots` prints the points no mutation kills. Two anchors were demoted mid-build rather than kept: see D18. |

## Phase 10 checklist

| Check | Result |
|---|---|
| Full test suite | 65 tests, all pass |
| Clean-environment reproduction | Fresh `git clone` into a new venv reproduced every raw CSV and the summary byte-identically |
| Figure regeneration | All 7 figures regenerate from raw results only |
| Claim-to-evidence audit | Every quoted number verified against `results/processed/summary.json`; enforced by `tests/test_claims.py`, which `make reproduce` runs after regenerating results |
| Citation and link verification | 18 of 19 URLs resolve; the exception is ACM's bot block, and all 5 DOIs resolve through doi.org |
| License review | MIT for code; no third-party code vendored; no third-party dataset redistributed |
| Secret scan | Clean. Only false positives on the word "token" in the LLM sense |
| Data-provenance review | `results/raw/` immutable with `.meta.json` sidecars recording version, commit, seed, and config digest |
| README installation test | `make venv` then `make smoke-test` from a clean checkout |
| Paper PDF inspection | Compiled and read; figures, tables, equations, references, and layout checked |
| Article fact check | Every number traced to `results/processed/summary.json` |
| Release notes | `CHANGELOG.md` |
| Citation file | `CITATION.cff` |
| Product or commercial content | None introduced |

## Quality pass before release (2026-08-05 to 2026-08-06)

Four iterations, recorded in `DECISIONS.md` D13 to D15:

1. **Solver sampling** (D13): the substitution solver interpolated across a coarse chord of a concave curve, inflating small SEA values about 13 percent. Fixed; decision map and validation unchanged.
2. **Claims and prose audit**: an article arithmetic error (blocked share misstated as 23.5 percent, actually 29.9) and an overstatement fixed; a newly found adjacent paper (Fernandez et al., arXiv 2411.13055) added to related work; novelty re-checked and still clear.
3. **Fresh-clone reproduction**: clean clone plus new venv reproduced every raw CSV and the summary byte-for-byte.
4. **Independent adversarial code review** (subagent with no context): found a real accounting bug in the event-driven reliability path (D14) and a semantics flaw in relatively-worded interventions (D15). Both fixed; all numbers regenerated; the two reliability implementations now agree to 0.02 percent at every tested severity.

## Decisions still open

1. **arXiv submission.** `cs.DC` primary with `cs.NI` cross-list recommended. Needs the owner's arXiv account.
2. **Author metadata.** `CITATION.cff` and the paper carry the author name with no affiliation, ORCID, or contact address.
3. **Hardware validation.** Not required for the published claims, which are scoped accordingly; a small multi-node benchmark would most strengthen the work and needs approval to rent capacity.

## Optional work not done

* **Cross-campus extension.** Scoped in `strategy_comparison.md` as a bounded regime-boundary analysis and deliberately not implemented. It would need a WAN model and a per-site power cap. Excluded from version one on validation feasibility; the public anchors are too thin to calibrate against.
* **Hardware validation.** A small multi-node benchmark measuring exposed communication as a share of step time would move the communication model from "validated against one published system" to "measured". This needs rented accelerator capacity and therefore your approval to spend money. A single-node result would not support any claim in the paper and should not be attempted as a substitute.
* **Cross-model validation against Calculon or ASTRA-sim.** Not performed. The model is checked against two independently derived closed forms and a held-out published measurement, so this would be corroboration rather than a gap, but it is the natural next validation layer.

## Latest successful reproduction command

```bash
make reproduce
```

Run 2026-08-06 from a fresh clone with a newly built virtual environment. Regenerated 9 raw result sets, 6 processed tables, 7 figures, and the paper PDF, byte-identical to the committed results. Total runtime under a minute plus the LaTeX build.

Phase 11 re-verified 2026-09-01: 83 tests pass, `validate_llama3.py` reports
`all_checks_pass`, `validate_span.py` reports 16 of 16, and `make smoke-test`
completes end to end. `make reproduce` now also runs the span experiments,
registry, and figures.

## Continuation instruction

If work resumes in a new session: the v1.0.0 research is complete and no phase
needs redoing. Read this file and `DECISIONS.md` (through D18) first.

Phase 11 added the span tier and the latency-regime atlas. Its headline is that
rank placement, not fiber length, sets the cost of spreading a job across halls
--- a 1,000x change in round-trip time costs the data-parallel cut 1.2 points,
while reordering the ranks at fixed distance costs it 2.7x and the pipeline cut
6.5x. Start at `docs/regime-atlas.md`, then the DECLINED list printed by
`validation/validate_span.py`, which is where the model's limits are recorded.

The paper and article still describe v1.0.0 only and were deliberately not
touched; folding the atlas into them is a separate decision for the owner.
