import dataclasses
import math
import json
import pytest
from netcap.wan import WANWorkload,evaluate,required_bandwidth


def test_closed_form_two_sites():
    w=WANWorkload(sites=2,update_bytes=1e9,aggregate_gbps=8,efficiency=1,rtt_ms=10,failure_rate_per_site_h=0)
    assert math.isclose(evaluate(w)['communication_s'],1.01)


def test_degradation_and_local_update_bounds():
    w=WANWorkload()
    assert evaluate(dataclasses.replace(w,bandwidth_health=0))['useful_compute_fraction']==0
    assert evaluate(dataclasses.replace(w,bandwidth_health=.5))['useful_compute_fraction']<evaluate(w)['useful_compute_fraction']
    assert evaluate(w,'periodic')['communication_per_step_s']==evaluate(w)['communication_s']/500
    assert evaluate(w,'decoupled')['useful_compute_fraction']>=evaluate(w,'periodic')['useful_compute_fraction']
    assert evaluate(dataclasses.replace(w,sites=1))['communication_s']==0


def test_required_bandwidth_round_trip_and_unreachable_target():
    w=WANWorkload(rtt_ms=0,failure_rate_per_site_h=0)
    bw=required_bandwidth(w,.8)
    assert math.isclose(evaluate(dataclasses.replace(w,aggregate_gbps=bw))['useful_compute_fraction'],.8)
    assert required_bandwidth(dataclasses.replace(w,rtt_ms=10000),.99) is None


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,True])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError): WANWorkload(rtt_ms=bad)


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_provenance_survives_disconnect_and_recovery(regime):
    w = WANWorkload()
    healthy = evaluate(w, regime)
    disconnected = evaluate(dataclasses.replace(w, bandwidth_health=0), regime)
    recovered = evaluate(w, regime)
    for result in (healthy, disconnected, recovered):
        assert result['evidence_class'] == 'derived'
        assert result['input_class'] == 'scenario'
        assert result['convergence_validated'] is False
        assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert disconnected == dict(
        regime=regime, evidence_class='derived', input_class='scenario',
        status='disconnected', useful_compute_fraction=0.,
        communication_s=None, cycle_s=None, convergence_validated=False)
    assert recovered == healthy


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_single_site_zero_bandwidth_remains_modeled(regime):
    w = WANWorkload(sites=1)
    local = evaluate(dataclasses.replace(w, bandwidth_health=0), regime)
    assert local == evaluate(w, regime)
    assert local['status'] == 'modeled'
    assert local['evidence_class'] == 'derived'
    assert local['input_class'] == 'scenario'
    assert local['communication_s'] == 0
    assert local['convergence_validated'] is False
    assert json.loads(json.dumps(local, allow_nan=False)) == local


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_unrepresentable_communication_refused_then_corrected(regime):
    w = WANWorkload()
    healthy = evaluate(w, regime)
    with pytest.raises(ValueError, match='finite numeric range'):
        evaluate(dataclasses.replace(w, bandwidth_health=math.nextafter(0., 1.)), regime)
    assert evaluate(w, regime) == healthy
    assert json.loads(json.dumps(healthy, allow_nan=False)) == healthy


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
@pytest.mark.parametrize('health', [1e-300, 1e-308])
def test_tiny_and_subnormal_health_with_finite_results_remain_modeled(regime, health):
    result = evaluate(WANWorkload(bandwidth_health=health), regime)
    assert result['status'] == 'modeled'
    assert result['communication_s'] > 1e290
    assert result['useful_compute_fraction'] > 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_single_site_minimum_positive_health_needs_no_exchange(regime):
    w = WANWorkload(sites=1)
    result = evaluate(dataclasses.replace(w, bandwidth_health=math.nextafter(0., 1.)), regime)
    assert result == evaluate(w, regime)
    assert result['communication_s'] == 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_subnormal_local_step_remains_finite(regime):
    result = evaluate(WANWorkload(sites=1, local_step_s=math.nextafter(0., 1.)), regime)
    assert result['status'] == 'modeled'
    assert result['cycle_s'] > 0
    assert result['useful_compute_fraction'] > 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_compute_overflow_refused_without_rejecting_finite_large_steps(regime):
    w = WANWorkload(local_step_s=1e308)
    if regime in ('periodic', 'decoupled'):
        with pytest.raises(ValueError, match='finite numeric range'):
            evaluate(w, regime)
    else:
        result = evaluate(w, regime)
        assert result['cycle_s'] == 1e308
        assert json.loads(json.dumps(result, allow_nan=False)) == result
    recovered = evaluate(dataclasses.replace(w, local_step_s=1.), regime)
    assert recovered == evaluate(WANWorkload(), regime)


@pytest.mark.parametrize('regime', ['synchronous', 'periodic', 'hierarchical', 'decoupled'])
def test_inverse_distinguishes_numeric_refusal_from_unreachable_then_recovers(regime):
    w = WANWorkload(rtt_ms=0, local_sync_s=0, inner_steps=1, failure_rate_per_site_h=0)
    with pytest.raises(ValueError, match='finite numeric range'):
        required_bandwidth(dataclasses.replace(w, bandwidth_health=math.nextafter(0., 1.)), .8, regime)
    assert required_bandwidth(dataclasses.replace(w, bandwidth_health=0), .8, regime) is None
    bw = required_bandwidth(w, .8, regime)
    result = evaluate(dataclasses.replace(w, aggregate_gbps=bw), regime)
    assert math.isclose(result['useful_compute_fraction'], .8)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
