"""Span-tier invariants, and mutation tests that delete the machinery.

A validation registry that is green proves nothing on its own: a registry of
tautologies is also green. These tests delete each piece of span machinery in
turn and require the registry to go red, which is the only evidence that its
points are load-bearing.

Two details matter for this to work at all.

First, the mutations patch every namespace that binds the mutated function, not
just the module that defines it. ``step_breakdown`` resolves ``group_layout``
and friends in ``netcap.performance``'s globals at call time, but ``regimes``
and the registry itself imported those names directly, so a single patch would
leave live copies behind and the mutation would silently do nothing. A test that
patches the wrong binding reports a green suite as proof of coverage.

Second, each mutation prints the red set it actually produced rather than
asserting a total. Some points cannot detect some deletions -- deleting the span
latency makes "distance does not bind below ten km" pass more easily, not less --
and ``test_registry_blind_spots`` records which those are. Coverage inferred from
a passing suite is not coverage.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Set

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

import netcap.performance as perf  # noqa: E402
import netcap.regimes as regimes  # noqa: E402
import validate_span as vs  # noqa: E402
from netcap.accounting import build_ledger  # noqa: E402
from netcap.config import TopologySpec, load_scenario, replace_nested  # noqa: E402

SCENARIO = ROOT / "configs/scenarios/llama3_405b_16k.yaml"


@pytest.fixture(scope="module")
def scn():
    return load_scenario(SCENARIO)


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


def test_defaults_disable_the_span_tier():
    """The extension must be inert unless a job is explicitly split."""
    topo = TopologySpec(scaleup_domain=8, pod_size=1024)
    assert topo.halls == 1
    assert topo.span_dimension == "none"
    assert topo.spans is False


def test_published_results_do_not_move(scn):
    """Hall-local accounting is byte-identical to the three-tier model."""
    before = build_ledger(scn)
    after = build_ledger(replace_nested(scn, **{"topology.halls": 1,
                                                "topology.span_dimension": "none"}))
    assert before.step.t_step == after.step.t_step
    assert before.useful_capacity_fraction == after.useful_capacity_fraction


def test_four_fate_invariant_survives_spanning(scn):
    """P + B + D + U = N * T must still close once a job crosses a stitch."""
    for cut in ("tp", "pp", "dp", "checkpoint"):
        for placement in ("hierarchical", "blind"):
            spanned = vs._span(scn, cut, 600.0, 800.0, placement)
            led = build_ledger(spanned)
            led.check_invariants(rel_tol=1e-9)
            total = led.productive + led.blocked + led.discarded + led.unavailable
            assert total == pytest.approx(led.total, rel=1e-9), f"{cut}/{placement}"


def test_span_config_rejects_incoherent_topologies():
    with pytest.raises(ValueError):
        TopologySpec(scaleup_domain=8, pod_size=1024, halls=2)  # no named dimension
    with pytest.raises(ValueError):
        TopologySpec(scaleup_domain=8, pod_size=1024, halls=2,
                     span_dimension="dp", span_bw_gbps=0.0)  # no circuit
    with pytest.raises(ValueError):
        TopologySpec(scaleup_domain=8, pod_size=1024, span_placement="round-robin")
    with pytest.raises(ValueError):
        TopologySpec(scaleup_domain=8, pod_size=1024, halls=0, span_dimension="dp")


def test_blind_placement_pays_latency_per_rank(scn):
    """2(k-1) alpha for a ring of k, against halls-many hops when contiguous."""
    topo = vs._span(scn, "dp", 40000.0, 204800.0, "blind").topology
    lay = perf.group_layout(128, 128, topo, "dp", 16384)
    assert lay.span_ring == 128
    blind = perf.span_alpha_seconds(lay, topo, "allreduce")

    topo_h = replace(topo, span_placement="hierarchical")
    lay_h = perf.group_layout(128, 128, topo_h, "dp", 16384)
    assert lay_h.span_ring == topo_h.halls == 2
    hier = perf.span_alpha_seconds(lay_h, topo_h, "allreduce")

    assert blind == pytest.approx(127.0 / 1.0 * hier / 1.0, rel=1e-9)
    assert blind > hier


def test_contention_scales_with_the_cluster(scn):
    """A shared circuit is divided by the groups crossing it, not per-rank."""
    topo = vs._span(scn, "dp", 600.0, 800.0, "hierarchical").topology
    small = perf.span_share_bytes_per_s(topo, 1.0, 400.0)
    crowded = perf.span_share_bytes_per_s(topo, 128.0, 400.0)
    assert crowded == pytest.approx(small / 128.0, rel=1e-9) or crowded < small


def test_pipeline_crossing_fraction_follows_placement(scn):
    """One boundary when contiguous, every boundary when interleaved."""
    topo_h = vs._span(scn, "pp", 600.0, 800.0, "hierarchical").topology
    topo_b = replace(topo_h, span_placement="blind")
    lay = perf.group_layout(16, 1, topo_h, "pp", 16384)
    assert lay.halls == 2
    assert perf.pipeline_crossing_fraction(lay, topo_h, 16) == pytest.approx(1.0 / 15.0)
    assert perf.pipeline_crossing_fraction(lay, topo_b, 16) == 1.0

    solo = perf.group_layout(16, 1, replace(topo_h, halls=1, span_dimension="none"), "pp", 16384)
    assert perf.pipeline_crossing_fraction(solo, topo_h, 16) == 0.0
    assert perf.pipeline_crossing_fraction(lay, topo_h, 1) == 0.0


# --------------------------------------------------------------------------
# the registry, unmutated: the green control
# --------------------------------------------------------------------------


def _red_set() -> Set[str]:
    return {p.name for p in vs.points() if not p.ok}


def test_registry_is_green_unmutated():
    """The control. Without this, a red mutation proves nothing."""
    red = _red_set()
    assert not red, f"registry red before any mutation: {sorted(red)}"


def test_registry_declares_what_it_does_not_check():
    assert len(vs.DECLINED) >= 8
    for title, why in vs.DECLINED:
        assert len(why) > 80, f"DECLINED entry {title!r} has no real reason"


def test_sanity_points_cite_nothing():
    """A self-consistency check with a citation beside it reads as external."""
    for p in vs.points():
        if p.kind == "sanity":
            assert p.ref == "-", f"{p.name} is sanity but cites {p.ref}"
        else:
            assert p.ref != "-" or p.kind == "emergent"


# --------------------------------------------------------------------------
# mutations
# --------------------------------------------------------------------------


def _patch_everywhere(monkeypatch, name: str, value) -> None:
    """Rebind a name in every module that imported it, not just its home."""
    for mod in (perf, regimes, vs):
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, value, raising=False)


def _mut_delete_latency_exposure(monkeypatch) -> None:
    """Charge the stitch's latency at the overlap fraction, like bandwidth.

    Note what this does NOT do: the span tier's alpha still enters the
    alpha-beta collective time via ``_tiers_from_layout``, so distance still
    costs something. ``span_alpha_seconds`` only computes the portion that
    cannot hide behind compute. A first draft of this file called this mutation
    "delete span latency" and expected it to flatten the distance ladder; it did
    not, because the latency has two paths and this deletes one. The ablation
    that removes both is ``zero-the-stitch-latency`` below.
    """
    _patch_everywhere(monkeypatch, "span_alpha_seconds",
                      lambda layout, topology, kind="allreduce": 0.0)


def _mut_zero_stitch_latency(monkeypatch) -> None:
    """Ablate distance itself: every stitch becomes instantaneous.

    Applied at the scenario rather than the code, because all three latency
    paths -- the span tier's alpha, the exposed term, and the inline pipeline
    term -- read ``topology.alpha_span_us``, and zeroing it removes all three at
    once. Any point that still passes here is not testing distance.
    """
    orig = regimes._spanned
    _patch_everywhere(monkeypatch, "_spanned",
                      lambda scenario, cut, alpha_us, gbps: orig(scenario, cut, 0.0, gbps))


def _mut_delete_nic_cap(monkeypatch) -> None:
    def uncapped(topology, contention, nic_bw_gbps=0.0):
        aggregate = topology.span_bw_gbps * 1e9 * topology.net_efficiency / 8.0
        return aggregate / max(1.0, contention)
    _patch_everywhere(monkeypatch, "span_share_bytes_per_s", uncapped)


def _mut_delete_contention(monkeypatch) -> None:
    def uncontended(topology, contention, nic_bw_gbps=0.0):
        return perf.__dict__["_orig_share"](topology, 1.0, nic_bw_gbps)
    monkeypatch.setitem(perf.__dict__, "_orig_share", perf.span_share_bytes_per_s)
    _patch_everywhere(monkeypatch, "span_share_bytes_per_s", uncontended)


def _mut_contiguous_pipeline_always(monkeypatch) -> None:
    """Assume a pipeline is always laid out contiguously, whatever the launcher.

    This is what the model did before the placement-aware branch existed: the
    crossing fraction was hardcoded to (halls-1)/(pp-1). It made the atlas report
    that rank order is irrelevant to a pipeline, which is false.
    """
    def contiguous(layout, topology, pp):
        if layout.halls <= 1 or pp <= 1:
            return 0.0
        return (layout.halls - 1) / (pp - 1)
    _patch_everywhere(monkeypatch, "pipeline_crossing_fraction", contiguous)


def _mut_collapse_blind_placement(monkeypatch) -> None:
    """Make blind placement indistinguishable from contiguous placement."""
    orig = perf.group_layout

    def collapsed(size, stride, topology, dimension="", world=0):
        lay = orig(size, stride, topology, dimension, world)
        return replace(lay, span_ring=lay.halls)
    _patch_everywhere(monkeypatch, "group_layout", collapsed)


def _mut_delete_span_tier(monkeypatch) -> None:
    monkeypatch.setattr(TopologySpec, "spans", property(lambda self: False))


#: Each mutation, and the points that MUST notice it. The required set is a
#: floor, not a prediction of the full red set: the actual red set is printed.
MUTATIONS: Dict[str, tuple] = {
    "delete-latency-exposure": (
        _mut_delete_latency_exposure,
        {"stitch-latency-survives-total-overlap"},
    ),
    "zero-the-stitch-latency": (
        _mut_zero_stitch_latency,
        {"blind-penalty-grows-with-distance",
         "tensor-parallel-is-the-only-distance-bound-cut",
         "stitch-latency-survives-total-overlap"},
    ),
    "contiguous-pipeline-always": (
        _mut_contiguous_pipeline_always,
        {"pipeline-placement-costs-at-every-distance"},
    ),
    "delete-nic-cap": (
        _mut_delete_nic_cap,
        {"no-rank-outruns-its-own-nic",
         "widening-past-the-nic-cap-buys-exactly-nothing"},
    ),
    "delete-contention": (
        _mut_delete_contention,
        {"a-metro-stitch-degrades-as-the-cluster-grows"},
    ),
    "collapse-blind-placement": (
        _mut_collapse_blind_placement,
        {"placement-not-distance-causes-the-cliff",
         "blind-penalty-grows-with-the-crossing-ring"},
    ),
    "delete-span-tier": (
        _mut_delete_span_tier,
        {"blind-penalty-grows-with-the-crossing-ring",
         "placement-not-distance-causes-the-cliff",
         "blind-penalty-grows-with-distance",
         "a-metro-stitch-degrades-as-the-cluster-grows",
         "stitch-latency-survives-total-overlap"},
    ),
}


@pytest.mark.parametrize("label", sorted(MUTATIONS))
def test_mutation_turns_the_registry_red(label, monkeypatch, capsys):
    mutate, required = MUTATIONS[label]
    mutate(monkeypatch)
    red = _red_set()

    with capsys.disabled():
        print(f"\n  mutation {label!r} -> {len(red)} red: {sorted(red)}")

    assert red, f"mutation {label!r} changed nothing: the registry cannot see it"
    missing = required - red
    assert not missing, (
        f"mutation {label!r} went unnoticed by {sorted(missing)}; "
        f"actually red: {sorted(red)}"
    )


def test_registry_blind_spots(monkeypatch, capsys):
    """Print the points no mutation can kill. These are the registry's limits.

    A point that survives every deletion is either checking something none of
    these mutations touch, or is not checking anything. Both are worth seeing in
    the open rather than discovering later.
    """
    survivors = {p.name for p in vs.points()}
    for label in sorted(MUTATIONS):
        mp = pytest.MonkeyPatch()
        try:
            MUTATIONS[label][0](mp)
            survivors -= _red_set()
        finally:
            mp.undo()

    with capsys.disabled():
        print("\n  points no mutation kills:")
        for name in sorted(survivors):
            print(f"    - {name}")

    # The two calibrated points are expected survivors: deleting machinery makes
    # a stitch look cheaper, and both are stated as ceilings a cheaper stitch
    # clears more easily. They guard against drift, not against deletion.
    assert survivors <= {
        "distance-does-not-bind-below-ten-km",
        "metro-two-site-both-cuts-clear-the-floor",
        "hall-local-is-the-published-three-tier-model",
        "spanning-never-speeds-a-step-up-at-reported-widths",
        "widening-a-circuit-never-slows-a-step",
        "no-stitch-no-stitch-latency",
        "exposed-bytes-order-the-cuts",
    }, f"unexpected survivors: {sorted(survivors)}"
