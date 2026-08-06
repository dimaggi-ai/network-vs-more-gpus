# Status

Last updated: 2026-08-05.

## Current phase

**Phase 10 complete. Awaiting publication approval.**

The repository, paper, and article are finished and internally consistent. Nothing has been published, pushed to a remote, or submitted anywhere. No money has been spent and no credentials have been used.

## Completed

| Phase | Output |
|---|---|
| 0. Intake and adjudication | `strategy_comparison.md`, `SOURCES.md`, `DECISIONS.md`, `ASSUMPTIONS.md` |
| 1. Literature and white space | `docs/literature_matrix.md`, Gate 1 passed |
| 2. Charter | `RESEARCH_CHARTER.md`, `docs/metric_framework.md` |
| 3-4. Design and implementation | `src/netcap/`, configs, Makefile, 55 tests |
| 5. Experiments | 9 experiment families, immutable raw results with provenance sidecars |
| 6. Validation | `validation/VALIDATION_REPORT.md`, Gates 2 and 3 passed |
| 7. Analysis and figures | 7 figures, `results/processed/` |
| 8. Paper | `paper/main.tex`, compiled and inspected, 11 pages |
| 9. Article | `article/article.md`, 2,032 words |
| 10. Publication checks | See below |

## Decision gates

| Gate | Status | Evidence |
|---|---|---|
| 1. Novelty | Pass | Per-work capability audit in `docs/literature_matrix.md`; no inspected work combines communication modeling, failure modeling, cost comparison, and accelerator substitution. Claim worded as "we found no prior public work that", not "first". |
| 2. Baseline credibility | Pass | Daly's optimal checkpoint interval and Meta's expected-ETTR closed form both recovered; held-out throughput prediction within 1.6 percent. |
| 3. Accounting correctness | Pass | Time conservation asserted on every evaluation; limiting cases and monotonicity tested. |
| 4. Finding robustness | Pass | Rank reversals survive the uncertainty analysis; out-of-envelope rows excluded; the one load-bearing out-of-envelope claim recomputed at high fidelity. |
| 5. Publication integrity | Pass | Each headline claim tagged validated, extrapolated, or unsupported in the validation report. No absolute cost claim anywhere. |

## Phase 10 checklist

| Check | Result |
|---|---|
| Full test suite | 55 tests, 53 pass, 2 skip (correctly, outside the validity envelope) |
| Clean-environment reproduction | `make reproduce` from an emptied `results/` and `figures/` regenerates everything |
| Figure regeneration | All 7 figures regenerate from raw results only |
| Claim-to-evidence audit | 29 quoted numbers verified against `results/processed/summary.json`; now enforced by `tests/test_claims.py` |
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

## Decisions required from you

1. **Publication approval.** Nothing goes public without it. The package is: GitHub repository at v1.0, the paper as an arXiv preprint, and the article. My recommendation is to release the repository and preprint together, and the article only after the preprint has a stable link.
2. **GitHub account and repository name.** The repository has local commits on `main` and no remote configured. I did not create a remote or push.
3. **arXiv category.** `cs.DC` (Distributed, Parallel, and Cluster Computing) is the closest fit, with `cs.NI` as a cross-list. Needs your arXiv account.
4. **Author metadata.** `CITATION.cff` and the paper currently carry your name with no affiliation, ORCID, or contact address. Tell me what to add.
5. **Whether to attempt hardware validation before publishing.** Not required for the current claims, which are scoped accordingly, but it is the single change that would most strengthen the work. See below.

## Blockers

None for the research. The only blockers are the approvals above, all of which are irreversible public actions or require your credentials.

## Next three actions

1. On your approval, add the git remote, push `main`, and tag `v1.0.0`.
2. On your approval, build the arXiv submission bundle (the paper already compiles standalone with `tectonic`) and submit.
3. Re-run the prior-art search immediately before submission and record the date in `docs/literature_matrix.md`, since this literature moves fast and the current search is dated 2026-08-05.

## Optional work not done

* **Cross-campus extension.** Scoped in `strategy_comparison.md` as a bounded regime-boundary analysis and deliberately not implemented. It would need a WAN model and a per-site power cap. Excluded from version one on validation feasibility; the public anchors are too thin to calibrate against.
* **Hardware validation.** A small multi-node benchmark measuring exposed communication as a share of step time would move the communication model from "validated against one published system" to "measured". This needs rented accelerator capacity and therefore your approval to spend money. A single-node result would not support any claim in the paper and should not be attempted as a substitute.
* **Cross-model validation against Calculon or ASTRA-sim.** Calculon is cloned to a scratch directory but not wired in. The model is already checked against two independently derived closed forms and a held-out published measurement, so this would be corroboration rather than a gap, but it is the natural next validation layer.

## Latest successful reproduction command

```bash
make reproduce
```

Run 2026-08-05 from an emptied `results/raw/`, `results/processed/`, and `figures/`. Regenerated 9 raw result sets, 6 processed tables, 7 figures, and the paper PDF. Total runtime approximately 45 seconds including the LaTeX build.

## Continuation instruction

If work resumes in a new session: the research is complete and no phase needs redoing. Read this file and `DECISIONS.md` first. Do not re-run the adjudication, the literature analysis, or the validation. The only outstanding items are the five approval decisions above and the three next actions, all of which are gated on your input.
