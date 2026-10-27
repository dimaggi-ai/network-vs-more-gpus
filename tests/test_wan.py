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
