"""Condense the raw results into the tables the paper and README quote.

Writes ``results/processed/summary.json`` and a set of small CSV tables. Every
number quoted in ``paper/`` and ``README.md`` must appear here, so a claim can be
checked against a file rather than re-derived by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
OUT = ROOT / "results/processed"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # --- E1: the headline accounting gap ------------------------------------
    e1 = load("e1_ledger_vs_scale")
    e1 = e1[e1.oversubscription == 7.0].sort_values("n_accelerators")
    e1_inside = e1[e1.within_validity_envelope]
    tbl = pd.DataFrame(
        {
            "n_accelerators": e1.n_accelerators,
            "productive_pct": (e1.productive / e1.total * 100).round(1),
            "blocked_pct": (
                (
                    e1.blocked_comm
                    + e1.blocked_sync
                    + e1.blocked_bubble
                    + e1.blocked_checkpoint
                    + e1.blocked_restart
                )
                / e1.total
                * 100
            ).round(1),
            "discarded_pct": (e1.discarded / e1.total * 100).round(1),
            "unavailable_pct": (
                (e1.unavailable_down + e1.unavailable_spare_stranded) / e1.total * 100
            ).round(1),
            "ucf": e1.ucf.round(3),
            "ettr": e1.ettr.round(3),
            "mttf_hours": e1.job_mttf_hours.round(2),
            "within_envelope": e1.within_validity_envelope,
        }
    )
    tbl.to_csv(OUT / "table1_capacity_by_scale.csv", index=False)
    at16 = e1[e1.n_accelerators == 16384].iloc[0]
    summary["capacity_gap_16k"] = {
        "ucf": round(float(at16.ucf), 4),
        "ettr": round(float(at16.ettr), 4),
        "gap_percentage_points": round(float(at16.ettr - at16.ucf) * 100, 1),
    }
    summary["capacity_by_scale"] = {
        int(r.n_accelerators): round(float(r.ucf), 3) for _, r in e1_inside.iterrows()
    }

    # --- E3: rank reversals --------------------------------------------------
    e3 = load("e3_decision_map")
    e3i = e3[e3.within_validity_envelope]
    summary["decision_map"] = {
        "cells_inside_envelope": int(len(e3i)),
        "cells_total": int(len(e3)),
        "distinct_winners": sorted(e3i.winner.unique().tolist()),
        "winners_by_scale": {
            int(n): sub.winner.value_counts().to_dict()
            for n, sub in e3i.groupby("n_accelerators")
        },
    }
    e3i.groupby(["n_accelerators", "winner"]).size().rename("cells").reset_index().to_csv(
        OUT / "table2_decision_map_winners.csv", index=False
    )

    # --- E5: bias of the informal metric ------------------------------------
    e5 = load("e5_naive_bias")
    e5 = e5[e5.within_validity_envelope & np.isfinite(e5.ratio_naive_over_sea)]
    by_scale = e5.groupby("n_accelerators").ratio_naive_over_sea.median()
    summary["naive_metric_bias"] = {
        int(k): round(float(v), 3) for k, v in by_scale.items()
    }
    summary["naive_metric_understatement_factor"] = {
        int(k): round(float(1.0 / v), 2) for k, v in by_scale.items()
    }
    by_scale.rename("median_ratio_naive_over_sea").reset_index().to_csv(
        OUT / "table3_naive_bias.csv", index=False
    )

    # --- E6: additivity ------------------------------------------------------
    e6 = load("e6_attribution")
    clean = e6[(e6.capped == 0) & e6.within_validity_envelope]
    summary["additivity"] = {
        "rows_used": int(len(clean)),
        "rows_capped_or_outside": int(len(e6) - len(clean)),
        "by_bundle": {
            str(b): {
                "mean_error": round(float(g.additivity_error.mean()), 3),
                "min": round(float(g.additivity_error.min()), 3),
                "max": round(float(g.additivity_error.max()), 3),
            }
            for b, g in clean.groupby("bundle")
        },
        "superadditive_rows": int((clean.additivity_error < 0).sum()),
        "subadditive_rows": int((clean.additivity_error > 0).sum()),
    }
    clean.to_csv(OUT / "table4_attribution.csv", index=False)

    # --- E7: rank stability --------------------------------------------------
    e7 = load("e7_uncertainty")
    e7i = e7[e7.within_validity_envelope]
    share = (e7i.winner.value_counts(normalize=True) * 100).round(1)
    summary["rank_stability"] = {
        "draws_inside_envelope": int(len(e7i)),
        "draws_total": int(len(e7)),
        "share_ranked_first_pct": share.to_dict(),
    }
    share.rename("share_ranked_first_pct").reset_index().to_csv(
        OUT / "table5_rank_stability.csv", index=False
    )

    # --- E8: cross-fidelity --------------------------------------------------
    e8 = load("e8_cross_fidelity")
    inside = e8[e8.within_validity_envelope]
    summary["cross_fidelity"] = {
        "max_rel_error_ucf_inside": round(float(inside.rel_error_ucf.max()), 4),
        "max_rel_error_ucf_outside": round(float(e8[~e8.within_validity_envelope].rel_error_ucf.max()), 4),
        "n_inside": int(len(inside)),
        "n_outside": int(len(e8) - len(inside)),
    }

    # --- E4: counterexamples -------------------------------------------------
    e4 = load("e4_counterexamples")
    net = ["bandwidth_2x", "bandwidth_4x", "flat_fabric", "scaleup_2x"]
    cases = {}
    for case, grp in e4.groupby("case"):
        network_sea = grp[grp.intervention.isin(net)].sea
        cases[str(case)] = {
            "n_accelerators": int(grp.n_accelerators.iloc[0]),
            "baseline_ucf": round(float(grp.baseline_ucf.iloc[0]), 3),
            "max_network_sea": (
                None if not np.isfinite(network_sea).any() else round(float(network_sea[np.isfinite(network_sea)].max()), 1)
            ),
            "max_reliability_sea": (
                None
                if not np.isfinite(grp[grp.intervention == "reliability_2x"].sea).any()
                else round(float(grp[grp.intervention == "reliability_2x"].sea.iloc[0]), 1)
            ),
            "any_unattainable_by_scaling": bool((~np.isfinite(grp.sea)).any()),
            "within_validity_envelope": bool(grp.within_validity_envelope.iloc[0]),
        }
    summary["counterexamples"] = cases
    pd.DataFrame(cases).T.to_csv(OUT / "table6_counterexamples.csv")

    # --- validation ----------------------------------------------------------
    val = json.loads((ROOT / "results/validation/llama3_validation.json").read_text())
    summary["validation"] = {
        k: val[k]
        for k in (
            "calibrated_kernel_efficiency",
            "heldout_tflops_pred",
            "heldout_tflops_obs",
            "heldout_rel_error",
            "heldout_mfu_pred",
            "heldout_mfu_obs",
            "predicted_interruptions_54d",
            "observed_interruptions_54d",
            "interruption_rel_error",
            "predicted_ettr",
            "checks",
            "all_checks_pass",
        )
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
