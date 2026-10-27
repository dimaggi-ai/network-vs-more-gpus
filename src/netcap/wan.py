"""Communication-regime bounds for scale-across training, not an optimizer.

Reuses netcap's ring alpha-beta primitive. Local-update intervals describe
communication schedules only; identical token work need not have identical
model quality. No DiLoCo implementation or convergence claim is made.
"""
from dataclasses import dataclass, asdict
import math
from .collectives import ring_allreduce_time


@dataclass(frozen=True)
class WANWorkload:
    sites: int = 8
    update_bytes: float = 2e9
    local_step_s: float = 1.
    rtt_ms: float = 40.
    aggregate_gbps: float = 100.
    concurrent_groups: int = 1
    efficiency: float = .8
    bandwidth_health: float = 1.
    latency_inflation: float = 1.
    fail_slow_multiplier: float = 1.
    inner_steps: int = 500
    local_sync_s: float = .02
    failure_rate_per_site_h: float = .001
    recovery_s: float = 300.

    def __post_init__(self):
        for name,value in asdict(self).items():
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('sites','concurrent_groups','inner_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError(name)
        if self.local_step_s<=0 or self.aggregate_gbps<=0 or not 0<self.efficiency<=1:
            raise ValueError('positive compute time, bandwidth and efficiency required')
        if not 0<=self.bandwidth_health<=1 or min(self.latency_inflation,self.fail_slow_multiplier)<1:
            raise ValueError('invalid degradation input')


def evaluate(w: WANWorkload, regime='synchronous'):
    if regime not in ('synchronous','periodic','hierarchical','decoupled'):
        raise ValueError('unknown synchronization regime')
    interval=1 if regime in ('synchronous','hierarchical') else w.inner_steps
    compute=w.local_step_s*interval
    # Aggregate circuit shared by simultaneously communicating groups; a
    # fail-slow bottleneck member bounds collective throughput.
    bandwidth=w.aggregate_gbps*1e9/8*w.efficiency*w.bandwidth_health/w.concurrent_groups/w.fail_slow_multiplier
    if bandwidth==0 and w.sites>1:
        return dict(regime=regime,evidence_class='derived',input_class='scenario',status='disconnected',useful_compute_fraction=0.,
                    communication_s=None,cycle_s=None,convergence_validated=False)
    comm=ring_allreduce_time(w.update_bytes,w.sites,bandwidth or 1.,w.rtt_ms/2000*w.latency_inflation)
    local=w.local_sync_s if regime=='hierarchical' else 0.
    # Perfect communication/compute overlap is a lower bound for decoupled
    # operation; buffers, convergence/staleness and contention may invalidate it.
    cycle=max(compute,comm)+local if regime=='decoupled' else compute+comm+local
    # Renewal availability approximation; common gang-wide failure events and
    # recovery cost, not an implementation of fault-tolerant training.
    failure_rate=w.failure_rate_per_site_h*w.sites/3600
    availability=1/(1+failure_rate*w.recovery_s)
    useful=compute/cycle*availability
    return dict(regime=regime,evidence_class='derived',input_class='scenario',status='modeled',
                communication_s=comm,cycle_s=cycle,communication_per_step_s=comm/interval,
                useful_compute_fraction=useful,availability_assumption=availability,
                communication_overhead_fraction=(cycle-compute)/cycle,
                convergence_validated=False,
                limitation='Communication/service bound only; useful token progress and model quality are not established.')


def required_bandwidth(w, target_fraction, regime='synchronous', maximum_gbps=1e6):
    """Bisection over the SAME model; None means target exceeds its search bound."""
    from dataclasses import replace
    if not 0<target_fraction<1 or not math.isfinite(maximum_gbps) or maximum_gbps<=0:
        raise ValueError('invalid target/search bound')
    hi=maximum_gbps;lo=1e-9
    if evaluate(replace(w,aggregate_gbps=hi),regime)['useful_compute_fraction']<target_fraction: return None
    for _ in range(70):
        mid=(lo+hi)/2
        if evaluate(replace(w,aggregate_gbps=mid),regime)['useful_compute_fraction']>=target_fraction: hi=mid
        else: lo=mid
    return hi
