"""Four-fate accelerator-time accounting.

Every accelerator-second in the measurement window is assigned exactly one
terminal fate:

* **Productive**  computation whose result is present in the delivered model state.
* **Blocked**     allocated and powered, but not computing: exposed collective
  communication, synchronization wait, checkpoint stall, restart.
* **Discarded**   computation that was performed and then thrown away, namely
  everything between the last durable checkpoint and failure detection.
* **Unavailable** owned but not usable by the job: down awaiting repair, held as
  a spare, or stranded behind a domain, power, or fragmentation limit.

Classification is by *terminal fate*, not by activity. A second spent computing
inside a window that was later discarded is Discarded, not Productive. This is
what makes the four buckets mutually exclusive and collectively exhaustive, and
it is checked numerically by :meth:`CapacityLedger.check_invariants`.

The relationship to established metrics is deliberate and is stated in the
paper: Model FLOPs Utilization and Effective Training Time Ratio are ratios over
different denominators that overlap, so their product is not a valid capacity
fraction. The Useful Capacity Fraction defined here has a single denominator,
the accelerator-seconds paid for.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal

from .config import ScenarioConfig
from .performance import (
    StepBreakdown,
    model_flops_per_token,
    model_flops_utilization,
    step_breakdown,
)
from .reliability import (
    ReliabilityTiming,
    analytical_reliability,
    monte_carlo_reliability,
)

Fidelity = Literal["analytical", "monte_carlo"]


@dataclass(frozen=True)
class CapacityLedger:
    """Accelerator-seconds by terminal fate over the measurement window."""

    scenario_name: str
    n_pool: int
    world_size: int
    window_s: float

    productive: float
    blocked_comm: float
    blocked_sync: float
    blocked_bubble: float
    blocked_checkpoint: float
    blocked_restart: float
    discarded: float
    unavailable_down: float
    unavailable_spare_stranded: float

    step: StepBreakdown
    timing: ReliabilityTiming
    fidelity: str

    @property
    def total(self) -> float:
        return self.n_pool * self.window_s

    @property
    def blocked(self) -> float:
        return (
            self.blocked_comm
            + self.blocked_sync
            + self.blocked_bubble
            + self.blocked_checkpoint
            + self.blocked_restart
        )

    @property
    def unavailable(self) -> float:
        return self.unavailable_down + self.unavailable_spare_stranded

    @property
    def useful_capacity_fraction(self) -> float:
        """Productive accelerator-seconds divided by accelerator-seconds paid for."""
        return self.productive / self.total if self.total > 0 else 0.0

    @property
    def productive_accelerators(self) -> float:
        """Productive accelerator-seconds per second: the throughput currency.

        This is the quantity the substitution metric equalizes. It is
        proportional to useful tokens per second whenever kernel efficiency is
        held constant, which it is across every intervention studied here.
        """
        return self.productive / self.window_s if self.window_s > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        """Useful tokens per second implied by the productive time."""
        if self.step.t_compute <= 0:
            return 0.0
        return self.productive_accelerators / self.world_size * (
            self.step.tokens_per_step / self.step.t_compute
        )

    @property
    def effective_training_time_ratio(self) -> float:
        """ETTR in the sense of Meta HPCA 2025, for comparison with reported values.

        Their productive runtime includes normal stepping (compute plus exposed
        communication and synchronization) and excludes only failure-induced
        loss and checkpoint stalls; the denominator is time the job was
        available rather than everything owned.
        The denominator is the *job's* available wallclock, so spare and
        stranded pool capacity is excluded: ETTR is a job-level metric and does
        not see capacity the job never held. That difference from the pool-level
        Useful Capacity Fraction is deliberate, and the gap between the two is
        one of the results this project reports.
        """
        available = self.world_size * self.window_s - self.unavailable_down
        if available <= 0:
            return 0.0
        productive_runtime = (
            self.productive + self.blocked_comm + self.blocked_sync + self.blocked_bubble
        )
        return productive_runtime / available

    def check_invariants(self, rel_tol: float = 1e-9) -> None:
        """Assert exhaustiveness, exclusivity, and non-negativity."""
        parts = {
            "productive": self.productive,
            "blocked_comm": self.blocked_comm,
            "blocked_sync": self.blocked_sync,
            "blocked_bubble": self.blocked_bubble,
            "blocked_checkpoint": self.blocked_checkpoint,
            "blocked_restart": self.blocked_restart,
            "discarded": self.discarded,
            "unavailable_down": self.unavailable_down,
            "unavailable_spare_stranded": self.unavailable_spare_stranded,
        }
        for name, value in parts.items():
            if value < -1e-6:
                raise AssertionError(f"negative accelerator-seconds in {name}: {value}")
        total = sum(parts.values())
        if self.total <= 0:
            raise AssertionError("non-positive total accelerator-seconds")
        drift = abs(total - self.total) / self.total
        if drift > rel_tol:
            raise AssertionError(
                f"time accounting does not close: sum={total:.6e} total={self.total:.6e} "
                f"relative drift={drift:.3e}"
            )

    def as_dict(self) -> Dict[str, float]:
        """Flat record for tabular output."""
        return {
            "scenario": self.scenario_name,
            "fidelity": self.fidelity,
            "n_pool": self.n_pool,
            "world_size": self.world_size,
            "window_days": self.window_s / 86400.0,
            "productive": self.productive,
            "blocked_comm": self.blocked_comm,
            "blocked_sync": self.blocked_sync,
            "blocked_bubble": self.blocked_bubble,
            "blocked_checkpoint": self.blocked_checkpoint,
            "blocked_restart": self.blocked_restart,
            "discarded": self.discarded,
            "unavailable_down": self.unavailable_down,
            "unavailable_spare_stranded": self.unavailable_spare_stranded,
            "total": self.total,
            "ucf": self.useful_capacity_fraction,
            "productive_accelerators": self.productive_accelerators,
            "tokens_per_second": self.tokens_per_second,
            "ettr": self.effective_training_time_ratio,
            "step_efficiency": self.step.step_efficiency,
            "t_step_s": self.step.t_step,
            "job_mttf_hours": self.timing.job_mttf_s / 3600.0,
            "checkpoint_interval_s": self.timing.checkpoint_interval_s,
            "n_failures": self.timing.n_failures,
            "recovery_pressure": self.timing.recovery_pressure,
            "within_validity_envelope": self.timing.within_validity_envelope,
        }


def build_ledger(
    scenario: ScenarioConfig,
    fidelity: Fidelity = "analytical",
    seed: int = 0,
    n_replicates: int = 200,
) -> CapacityLedger:
    """Compute the four-fate ledger for one scenario."""
    step = step_breakdown(scenario)
    world = scenario.parallelism.world_size

    if fidelity == "monte_carlo":
        timing = monte_carlo_reliability(
            scenario.reliability,
            world,
            scenario.accelerator.accelerators_per_node,
            scenario.window_seconds,
            n_pool_accelerators=scenario.n_pool,
            seed=seed,
            n_replicates=n_replicates,
        )
    else:
        timing = analytical_reliability(
            scenario.reliability,
            world,
            scenario.accelerator.accelerators_per_node,
            scenario.window_seconds,
            n_pool_accelerators=scenario.n_pool,
        )

    # Surviving running time is split by the healthy-step composition.
    running = timing.running_s
    t_step = step.t_step
    if t_step <= 0:
        raise ValueError("non-positive step time")
    frac_compute = step.t_compute / t_step
    frac_comm = (
        step.t_tp_exposed + step.t_cp_exposed + step.t_pp_exposed + step.t_dp_exposed
    ) / t_step
    frac_bubble = step.t_bubble / t_step
    frac_sync = step.t_sync_wait / t_step

    productive = world * running * frac_compute
    blocked_comm = world * running * frac_comm
    blocked_bubble = world * running * frac_bubble
    blocked_sync = world * running * frac_sync
    blocked_ckpt = world * timing.checkpoint_blocked_s
    blocked_restart = world * timing.restart_blocked_s
    discarded = world * timing.discarded_s
    unavail_down = world * timing.unavailable_s
    unavail_pool = (scenario.n_pool - world) * scenario.window_seconds

    ledger = CapacityLedger(
        scenario_name=scenario.name,
        n_pool=scenario.n_pool,
        world_size=world,
        window_s=scenario.window_seconds,
        productive=productive,
        blocked_comm=blocked_comm,
        blocked_sync=blocked_sync,
        blocked_bubble=blocked_bubble,
        blocked_checkpoint=blocked_ckpt,
        blocked_restart=blocked_restart,
        discarded=discarded,
        unavailable_down=unavail_down,
        unavailable_spare_stranded=unavail_pool,
        step=step,
        timing=timing,
        fidelity=fidelity,
    )
    ledger.check_invariants(rel_tol=1e-6 if fidelity == "monte_carlo" else 1e-9)
    return ledger


def mfu(scenario: ScenarioConfig) -> float:
    """Convenience wrapper for the conventionally reported MFU of a scenario."""
    return model_flops_utilization(scenario, step_breakdown(scenario))
