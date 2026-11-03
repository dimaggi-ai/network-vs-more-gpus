"""Synthetic contract tests; no vendor performance or hardware qualification."""
from dataclasses import asdict, replace
import unittest

from netcap.capability_evidence import (
    CapabilityEvidence, ComparisonContext, Provenance, Quantity,
    UNMODELED_DIMENSIONS, assess, assess_pair,
)

DATE = "2026-11-03"


def control():
    context = ComparisonContext("synthetic-training", "fixed-quality-A", "protocol-A",
                                "dense_decoder_only", "netcap_four_tier", (8, 1, 1, 1), 3600.)
    return CapabilityEvidence(
        "synthetic-device-A", "BF16", "dense",
        Quantity(1., "PFLOP/s", "per_accelerator", "not_applicable"),
        Quantity(80., "GB", "per_accelerator", "not_applicable"),
        Quantity(100., "GB/s", "per_accelerator", "unidirectional"),
        Quantity(400., "Gbit/s", "per_accelerator", "unidirectional"),
        context, Provenance("synthetic:unit-test", "fixture-v1", DATE, "a" * 64, "synthetic"))


class ApplicabilityTests(unittest.TestCase):
    def assess(self, item, expected=None, **kw):
        return assess(item, expected or control().context, as_of=kw.get("as_of", DATE))

    def test_matched_control_normalizes_only_declared_dimensions(self):
        result = self.assess(control())
        self.assertEqual(result.status, "applicable_to_declared_model")
        self.assertEqual(asdict(result.inputs), dict(peak_flops_bf16=1e15, memory_gb=80.,
                                                    scaleup_bw_gbps=800., nic_bw_gbps=400.))
        self.assertEqual(result.unknown_dimensions, UNMODELED_DIMENSIONS)
        self.assertEqual(len(result.record_sha256), 64)
        self.assertTrue(any("full-stack" in s for s in result.limitations))

    def test_explicit_binary_memory_units(self):
        p = control()
        result = self.assess(replace(p, memory=replace(p.memory, value=80, unit="GiB")))
        self.assertEqual(result.inputs.memory_gb, 80 * 2**30 / 1e9)

    def test_explicit_decimal_bandwidth_units(self):
        p = control()
        for unit, value in [("Tbit/s", .8), ("TB/s", .1), ("Gbit/s", 800)]:
            with self.subTest(unit=unit):
                r = self.assess(replace(p, scaleup=replace(p.scaleup, unit=unit, value=value)))
                self.assertEqual(r.inputs.scaleup_bw_gbps, 800)

    def test_precision_refusal_and_corrected_recovery(self):
        p = control()
        for precision in ["FP4", "FP8", "BF16/FP4", "bf16"]:
            with self.subTest(precision=precision):
                refused = self.assess(replace(p, precision=precision))
                self.assertEqual(refused.status, "unsupported")
                self.assertIsNone(refused.inputs)
        self.assertEqual(self.assess(p), self.assess(replace(p, precision="BF16")))

    def test_no_sparsity_conversion(self):
        self.assertEqual(self.assess(replace(control(), sparsity="2:4_sparse")).status, "unsupported")

    def test_no_bidirectional_halving_or_aggregate_division(self):
        p = control()
        for field, value in [("direction", "bidirectional"), ("scope", "domain_aggregate")]:
            r = self.assess(replace(p, scaleup=replace(p.scaleup, **{field: value})))
            self.assertEqual(r.status, "unsupported")
            self.assertIsNone(r.inputs)

    def test_ambiguous_units_refused(self):
        p = control()
        for unit in ["TBps", "gbps", "Gb", "bytes", "TB/s bidirectional"]:
            with self.subTest(unit=unit):
                self.assertEqual(self.assess(replace(p, scaleup=replace(p.scaleup, unit=unit))).status,
                                 "unsupported")

    def test_missing_quantity_remains_incomplete(self):
        for field in ["compute", "memory", "scaleup", "scaleout"]:
            with self.subTest(field=field):
                r = self.assess(replace(control(), **{field: None}))
                self.assertEqual(r.status, "incomplete")
                self.assertIsNone(r.inputs)

    def test_numeric_type_and_range_refusals(self):
        p = control()
        for value in [True, False, 0, -1, float("nan"), float("inf"), -float("inf"),
                      10**10000, "1.0", None, [], 1e101]:
            for field in ["compute", "memory", "scaleup", "scaleout"]:
                with self.subTest(field=field, value_type=type(value).__name__):
                    r = self.assess(replace(p, **{field: replace(getattr(p, field), value=value)}))
                    self.assertEqual(r.status, "invalid")
                    self.assertIsNone(r.inputs)

    def test_missing_or_invalid_provenance(self):
        p = control()
        self.assertEqual(self.assess(replace(p, provenance=None)).status, "incomplete")
        for field, value in [("source", ""), ("revision", ""), ("sha256", "ABC"), ("sha256", None)]:
            with self.subTest(field=field):
                self.assertEqual(self.assess(replace(p, provenance=replace(p.provenance, **{field: value}))).status,
                                 "incomplete")
        self.assertEqual(self.assess(replace(p, provenance={})).status, "invalid")

    def test_dates_no_implicit_freshness(self):
        p = control()
        old = replace(p, provenance=replace(p.provenance, retrieved_on="2000-01-01"))
        r = self.assess(old)
        self.assertEqual(r.status, "applicable_to_declared_model")
        self.assertTrue(any("freshness" in s for s in r.limitations))
        for date in ["2026-02-30", "2026-1-3", "tomorrow", "2026-11-04", True]:
            with self.subTest(date=date):
                self.assertEqual(self.assess(replace(p, provenance=replace(p.provenance, retrieved_on=date))).status,
                                 "invalid")
        self.assertEqual(self.assess(p, as_of="").status, "invalid")

    def test_declared_expiry_only_and_recovery(self):
        p = control()
        expired = replace(p, provenance=replace(p.provenance, retrieved_on="2026-01-01", valid_until="2026-11-02"))
        self.assertEqual(self.assess(expired).status, "unsupported")
        corrected = replace(expired, provenance=replace(expired.provenance, valid_until=DATE))
        self.assertEqual(self.assess(corrected).status, "applicable_to_declared_model")
        invalid = replace(p, provenance=replace(p.provenance, valid_until="2020-01-01"))
        self.assertEqual(self.assess(invalid).status, "invalid")

    def test_evidence_class_not_mixed(self):
        p = control()
        for kind in ["vendor_specification", "measured"]:
            with self.subTest(kind=kind):
                other = replace(p, provenance=replace(p.provenance, evidence_class=kind))
                a, b = assess_pair(p, other, p.context, as_of=DATE)
                self.assertEqual((a.status, b.status), ("unsupported", "unsupported"))
                self.assertIsNone(a.inputs)
                self.assertIsNone(b.inputs)
        self.assertEqual(self.assess(replace(p, provenance=replace(p.provenance, evidence_class="target"))).status,
                         "unsupported")

    def test_matched_pair_is_declarations_only(self):
        p = control()
        pair = assess_pair(p, replace(p, hardware="synthetic-B"), p.context, as_of=DATE)
        self.assertTrue(all(x.status == "applicable_to_declared_model" for x in pair))
        self.assertTrue(all(any("not measured comparability" in s for s in x.limitations) for x in pair))

    def test_workload_quality_protocol_scale_window_mismatches(self):
        p = control()
        for field, value in [("workload", "other"), ("quality_target", "other"), ("protocol", "other"),
                             ("parallelism", (4, 1, 1, 1)), ("window_seconds", 7200)]:
            with self.subTest(field=field):
                r = self.assess(replace(p, context=replace(p.context, **{field: value})))
                self.assertEqual(r.status, "unsupported")

    def test_unsupported_model_or_topology(self):
        p = control()
        for field, value in [("model_family", "MoE"), ("topology_model", "Boardfly")]:
            changed = replace(p.context, **{field: value})
            r = self.assess(replace(p, context=changed), expected=changed)
            self.assertEqual(r.status, "unsupported")

    def test_invalid_context(self):
        p = control()
        for field, value in [("window_seconds", True), ("window_seconds", 10**10000),
                             ("parallelism", [8, 1, 1, 1]), ("parallelism", (True, 1, 1, 1)),
                             ("parallelism", (0, 1, 1, 1)), ("workload", "")]:
            with self.subTest(field=field):
                self.assertEqual(self.assess(replace(p, context=replace(p.context, **{field: value}))).status,
                                 "invalid")

    def test_unknown_dimensions_cannot_be_removed(self):
        p = control()
        for value in [(), [], ("host",), None]:
            self.assertEqual(self.assess(replace(p, unknown_dimensions=value)).status, "invalid")

    def test_bounded_text_and_typed_input(self):
        p = control()
        for value in ["", " x", "x\n", "x" * 2049, [], None]:
            self.assertEqual(self.assess(replace(p, hardware=value)).status, "invalid")
        self.assertEqual(assess({}, p.context, as_of=DATE).status, "invalid")
        self.assertEqual(self.assess(replace(p, scaleup={})).status, "invalid")
        with self.assertRaises(TypeError):
            CapabilityEvidence(**(asdict(p) | {"extra": "dropped?"}))

    def test_deterministic_without_mutation_and_record_binding(self):
        p = control()
        before = asdict(p)
        first = self.assess(p)
        self.assertEqual(first, self.assess(p))
        self.assertEqual(before, asdict(p))
        for changed in [replace(p, hardware="synthetic-B"),
                        replace(p, provenance=replace(p.provenance, revision="v2")),
                        replace(p, provenance=replace(p.provenance, sha256="b" * 64)),
                        replace(p, compute=replace(p.compute, value=2))]:
            self.assertNotEqual(first.record_sha256, self.assess(changed).record_sha256)


if __name__ == "__main__":
    unittest.main()
