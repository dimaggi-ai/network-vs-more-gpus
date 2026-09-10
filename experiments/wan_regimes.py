"""Reproducible communication-regime grid; all ranges are scenario assumptions."""
from dataclasses import asdict
import json
from pathlib import Path
from netcap.wan import WANWorkload,evaluate,required_bandwidth


def experiment():
    rows=[]
    for rtt in (.001,.1,1,10,40,100):
        for bandwidth in (10,100,400):
            for size in (2e9,140e9,810e9):
                for health in (.5,1):
                    w=WANWorkload(rtt_ms=rtt,aggregate_gbps=bandwidth,update_bytes=size,bandwidth_health=health)
                    for regime in ('synchronous','periodic','hierarchical','decoupled'):
                        rows.append(dict(inputs=asdict(w),result=evaluate(w,regime)))
    return dict(evidence_class='derived',input_class='scenario',runs=rows,
                reference_cases=[{'source':'https://arxiv.org/abs/2311.08105','parameter':'inner_steps',
                                  'value':500,'unit':'local steps/update','use':'Communication frequency reference only; no convergence transfer'}],
                baseline='Same sites, bytes, bandwidth, RTT, failure/recovery assumptions and local compute per step.',
                limitations=['Fewer updates are not assumed to give equal learning progress.',
                             'No UET, NCCL, RDMA or DiLoCo implementation is exercised.',
                             'Hierarchical case adds a declared local reduction cost; sites are WAN ring participants.',
                             'Decoupled is an ideal overlap bound, not measured serving/training performance.',
                             'Network requirement is conditional on model and target; no fabricated cost or margin result.'])


if __name__=='__main__':
    result=experiment();p=Path('results/wan-regimes.json');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'{len(result["runs"])} deterministic scenario comparisons written to {p}')
