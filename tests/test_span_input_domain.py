"""Span-domain refusal and corrected recovery without changing the model."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from netcap.config import (TopologySpec, ParallelismSpec, load_scenario,
                           replace_nested, scenario_from_dict)
from netcap.cli import main
from netcap.accounting import build_ledger

ROOT = Path(__file__).resolve().parents[1]


def active(cut='dp', halls=2):
    return replace_nested(load_scenario(ROOT/'configs/scenarios/llama3_405b_16k.yaml'),
                          **{'topology.halls': halls, 'topology.span_dimension': cut,
                             'topology.span_bw_gbps': 800, 'topology.alpha_span_us': 600,
                             'parallelism.'+cut: 8, 'n_pool': 0})


BAD_SPAN = [('halls', x) for x in (True, False, 0, -1, 2.5, 2.0, '2', '2.5', 'nan', float('nan'), float('inf'), 10**400)] + [
    (name, x) for name in ('alpha_span_us', 'span_bw_gbps')
    for x in (True, False, -1, '2', '2.5', 'nan', float('nan'), float('inf'), 10**400)] + [('span_bw_gbps', 0)]


@pytest.mark.parametrize('field,value', BAD_SPAN)
def test_api_and_yaml_cli_refuse_then_correct(field, value, tmp_path, capsys):
    good = active()
    with pytest.raises(ValueError, match=field):
        replace(good.topology, **{field: value})
    data = asdict(good)
    data['topology'][field] = value
    with pytest.raises(ValueError, match=field): scenario_from_dict(data)
    path = tmp_path/'scenario.yaml'
    path.write_text(yaml.safe_dump(data))
    assert main([str(path)]) == 2
    captured = capsys.readouterr()
    assert field in captured.err
    assert captured.out == ''
    path.write_text(yaml.safe_dump(asdict(good)))
    assert main([str(path)]) == 0
    assert 'derived' in capsys.readouterr().out


@pytest.mark.parametrize('cut', ('tp','cp','pp','dp'))
@pytest.mark.parametrize('value', (True, False, 0, -1, 2.5, 2.0, '2', '2.5', 'nan', float('nan'), float('inf'), 10**400))
def test_named_degree_strict_api_yaml_cli(cut, value, tmp_path, capsys):
    good = active(cut)
    with pytest.raises(ValueError, match='parallelism.'+cut):
        replace(good, parallelism=replace(good.parallelism, **{cut: value}))
    data = asdict(good)
    data['parallelism'][cut] = value
    # Reverse field order proves topology validation does not depend on YAML ordering.
    data = dict(reversed(list(data.items())))
    path = tmp_path/'scenario.yaml'
    path.write_text(yaml.safe_dump(data))
    assert main([str(path)]) == 2
    assert 'parallelism.'+cut in capsys.readouterr().err
    path.write_text(yaml.safe_dump(asdict(good)))
    assert main([str(path)]) == 0
    capsys.readouterr()


@pytest.mark.parametrize('cut', ('tp','cp','pp','dp'))
def test_more_halls_than_degree_refuses_then_corrects(cut, tmp_path, capsys):
    good = active(cut)
    with pytest.raises(ValueError, match='halls must not exceed'):
        replace(good, topology=replace(good.topology, halls=9))
    data=asdict(good); data['topology']['halls']=9
    path=tmp_path/'scenario.yaml'; path.write_text(yaml.safe_dump(data))
    assert main([str(path)]) == 2
    assert 'halls must not exceed' in capsys.readouterr().err
    assert build_ledger(good).useful_capacity_fraction > 0


@pytest.mark.parametrize('cut', ('tp','cp','pp','dp'))
@pytest.mark.parametrize('halls', (2,3,8))
def test_valid_nondivisible_span_and_zero_latency(cut, halls):
    cfg=active(cut, halls)
    cfg=replace(cfg, topology=replace(cfg.topology, alpha_span_us=0))
    assert build_ledger(cfg).useful_capacity_fraction > 0
    assert scenario_from_dict(asdict(cfg)) == cfg


def test_hall_local_defaults_and_legacy_degree_coercion():
    cfg=load_scenario(ROOT/'configs/scenarios/llama3_405b_16k.yaml')
    assert cfg.topology.halls == 1
    assert cfg.topology.span_bw_gbps == cfg.topology.alpha_span_us == 0
    assert not cfg.topology.spans
    data=asdict(cfg); data['parallelism']['tp']='8'
    assert scenario_from_dict(data).parallelism.tp == 8
    assert build_ledger(scenario_from_dict(data)) == build_ledger(cfg)


@pytest.mark.parametrize('field', ('alpha_span_us','span_bw_gbps'))
def test_inactive_span_still_refuses_nonfinite(field):
    with pytest.raises(ValueError, match=field):
        TopologySpec(8,1024,**{field:float('nan')})
    assert TopologySpec(8,1024,**{field:0}).halls == 1
