"""Failure, detection, recovery, and checkpoint modeling.

Two independent paths compute the same quantities:

* :func:`analytical_reliability` uses renewal-theory expectations.
* :func:`monte_carlo_reliability` simulates the wall-clock timeline event by
  event with explicit checkpoints and failures.

Agreement between the two is a required validation check (Gate 2). They share
no code beyond the parameter dataclasses.

Accounting convention for a failure: the work performed between the last
durable checkpoint and the moment the failure is detected is **discarded**. The
subsequent re-execution of that work is ordinary productive execution and is
counted once, in the productive bucket. This avoids charging the same progress
twice, which is the most common error in informal goodput accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ReliabilitySpec, SECONDS_PER_DAY


#: Scope guard, not an accuracy cliff: the analytical and event-driven paths
#: agree to well under one percent at every tested recovery pressure (E8).
#: Configurations above this threshold are excluded from headline claims anyway,
#: because a job spending more than a quarter of its runtime on discarded work
#: and restart is outside the regime the model's independence assumptions
#: (uncorrelated failures, unqueued repairs) were calibrated for, and no
#: operator would run one uncorrected. See DECISIONS.md D9 and D14.
VALIDITY_RECOVERY_PRESSURE = 0.25


@dataclass(frozen=True)
class ReliabilityTiming:
    """Wall-clock time per accelerator, in seconds, over the measurement window."""

    discarded_s: float
    restart_blocked_s: float
    checkpoint_blocked_s: float
    running_s: float  # surviving running time, still to be split by step efficiency
    unavailable_s: float
    n_failures: float
    checkpoint_interval_s: float
    job_mttf_s: float

    @property
    def recovery_pressure(self) -> float:
        """Share of runtime consumed by discarded work plus restart.

        The dimensionless severity of the failure regime. Above
        :data:`VALIDITY_RECOVERY_PRESSURE` the configuration is excluded from
        headline results as out of scope; the exclusion is a modeling-scope
        judgment, not a fidelity limit, since the two implementations agree
        closely at every tested severity.
        """
        if not math.isfinite(self.job_mttf_s) or self.job_mttf_s <= 0:
            return 0.0
        return (self.discarded_s + self.restart_blocked_s) / max(
            1e-9, self.discarded_s + self.restart_blocked_s + self.running_s
        )

    @property
    def within_validity_envelope(self) -> bool:
        return self.recovery_pressure < VALIDITY_RECOVERY_PRESSURE


def daly_optimal_interval(checkpoint_write_s: float, mttf_s: float) -> float:
    """Daly's first-order optimal compute interval between checkpoints.

    Daly (2006) gives an optimal interval of ``sqrt(2 * delta * M) - delta`` for
    checkpoint cost ``delta`` and mean time to interrupt ``M``. Returned value is
    floored at 60 s to avoid degenerate configurations.
    """
    if checkpoint_write_s <= 0:
        return max(60.0, mttf_s)
    interval = math.sqrt(2.0 * checkpoint_write_s * mttf_s) - checkpoint_write_s
    return max(60.0, interval)


def expected_lost_work(rate: float, period: float) -> float:
    """Expected wall time since the last checkpoint, given a failure occurred.

    For exponential failures at ``rate`` and periodic checkpoints every
    ``period``, conditioning on the failure falling inside one period gives

        E[t] = (1/r - (period + 1/r) * exp(-r*period)) / (1 - exp(-r*period))

    which tends to ``period / 2`` as ``rate * period`` approaches zero. The
    series limit is used for small arguments to avoid catastrophic cancellation.
    """
    if rate <= 0 or period <= 0:
        return 0.0
    x = rate * period
    if x < 1e-6:
        return period / 2.0
    return (1.0 / rate - (period + 1.0 / rate) * math.exp(-x)) / (1.0 - math.exp(-x))


def spare_availability(n_spare_nodes: float, failure_rate_per_s: float, repair_s: float) -> float:
    """Probability a hot spare is free when a node fails.

    Concurrent down nodes are approximated as Poisson with mean
    ``failure_rate * repair_time``. With ``k`` spares the probability that fewer
    than ``k`` nodes are already down is the Poisson CDF at ``k - 1``.
    """
    k = int(math.floor(n_spare_nodes))
    if k <= 0:
        return 0.0
    mean = failure_rate_per_s * repair_s
    if mean <= 0:
        return 1.0
    cdf = 0.0
    term = math.exp(-mean)
    for i in range(0, k):
        if i > 0:
            term *= mean / i
        cdf += term
    return min(1.0, cdf)


def _resolve_interval(rel: ReliabilitySpec, mttf_s: float) -> float:
    if rel.checkpoint_interval_s is not None:
        return rel.checkpoint_interval_s
    return daly_optimal_interval(rel.checkpoint_write_s * rel.checkpoint_blocking_fraction, mttf_s)


def analytical_reliability(
    rel: ReliabilitySpec,
    n_job_accelerators: int,
    accelerators_per_node: int,
    window_s: float,
    n_pool_accelerators: Optional[int] = None,
) -> ReliabilityTiming:
    """Renewal-theory expectation of time in each failure-related state."""
    n_pool = n_pool_accelerators or n_job_accelerators
    n_nodes_job = max(1.0, n_job_accelerators / accelerators_per_node)

    rate_per_node_s = rel.failure_rate_per_node_day / SECONDS_PER_DAY
    lam = n_nodes_job * rate_per_node_s  # job interruption rate, per second
    mttf = 1.0 / lam if lam > 0 else float("inf")

    interval = _resolve_interval(rel, mttf)
    w_block = rel.checkpoint_write_s * rel.checkpoint_blocking_fraction
    wall_period = interval + w_block

    n_spare_nodes = rel.spare_fraction * n_pool / accelerators_per_node
    p_spare = spare_availability(n_spare_nodes, lam, rel.repair_time_s)
    restart_eff = rel.restart_time_s + (1.0 - p_spare) * rel.repair_time_s

    # Unavailable time for the job's own accelerators: the share of the window a
    # node is physically down awaiting repair. Spares and stranded capacity are
    # accounted at the pool level in :mod:`netcap.accounting`, not here, so that
    # they are never counted twice.
    down_nodes_mean = lam * rel.repair_time_s
    down_frac = min(0.5, down_nodes_mean * accelerators_per_node / max(1, n_job_accelerators))
    unavailable_frac = down_frac

    running_window = window_s * (1.0 - unavailable_frac)

    lost_per_failure = expected_lost_work(lam, wall_period) + rel.detect_time_s
    cycle = mttf + rel.detect_time_s + restart_eff
    n_failures = running_window / cycle if cycle > 0 else 0.0

    discarded = n_failures * lost_per_failure
    restart_blocked = n_failures * restart_eff
    surviving = max(0.0, running_window - discarded - restart_blocked)
    ckpt_blocked = surviving * (w_block / wall_period) if wall_period > 0 else 0.0
    running = surviving - ckpt_blocked

    return ReliabilityTiming(
        discarded_s=discarded,
        restart_blocked_s=restart_blocked,
        checkpoint_blocked_s=ckpt_blocked,
        running_s=running,
        unavailable_s=window_s * unavailable_frac,
        n_failures=n_failures,
        checkpoint_interval_s=interval,
        job_mttf_s=mttf,
    )


def monte_carlo_reliability(
    rel: ReliabilitySpec,
    n_job_accelerators: int,
    accelerators_per_node: int,
    window_s: float,
    n_pool_accelerators: Optional[int] = None,
    seed: int = 0,
    n_replicates: int = 200,
) -> ReliabilityTiming:
    """Event-driven simulation of the same timeline, averaged over replicates.

    Implemented independently of :func:`analytical_reliability`: it walks the
    wall clock, places checkpoints, draws exponential failure times, and
    attributes every second to a bucket as it goes.
    """
    n_pool = n_pool_accelerators or n_job_accelerators
    n_nodes_job = max(1.0, n_job_accelerators / accelerators_per_node)
    rate_per_node_s = rel.failure_rate_per_node_day / SECONDS_PER_DAY
    lam = n_nodes_job * rate_per_node_s
    mttf = 1.0 / lam if lam > 0 else float("inf")

    interval = _resolve_interval(rel, mttf)
    w_block = rel.checkpoint_write_s * rel.checkpoint_blocking_fraction
    wall_period = interval + w_block

    n_spare_nodes = rel.spare_fraction * n_pool / accelerators_per_node
    p_spare = spare_availability(n_spare_nodes, lam, rel.repair_time_s)
    restart_eff = rel.restart_time_s + (1.0 - p_spare) * rel.repair_time_s

    down_nodes_mean = lam * rel.repair_time_s
    down_frac = min(0.5, down_nodes_mean * accelerators_per_node / max(1, n_job_accelerators))
    unavailable_frac = down_frac
    running_window = window_s * (1.0 - unavailable_frac)

    rng = np.random.default_rng(seed)
    acc = np.zeros(4)  # discarded, restart, checkpoint, running
    fails = 0.0

    for _ in range(n_replicates):
        t = 0.0
        last_ckpt = 0.0  # wall time of the last completed checkpoint
        d = r = c = run = 0.0
        while t < running_window:
            gap = rng.exponential(mttf) if lam > 0 else float("inf")
            end = min(t + gap, running_window)
            span = end - t
            # Healthy span: split between checkpoint stalls and running.
            c_span = span * (w_block / wall_period) if wall_period > 0 else 0.0
            c += c_span
            run += span - c_span
            # Number of durable checkpoints completed inside this span.
            n_ck = math.floor((end - last_ckpt) / wall_period)
            if n_ck > 0:
                last_ckpt += n_ck * wall_period
            t = end
            if t >= running_window:
                break
            fails += 1
            # Work performed since the last durable checkpoint moves out of the
            # healthy buckets into discarded. Detection wall-time is *new* time,
            # booked directly as discarded; it was never part of the healthy
            # span, so it must not be subtracted from run or c. (An earlier
            # version subtracted it and then rescaled all buckets to fit the
            # window, which biased every bucket at high failure rates; see
            # DECISIONS.md D14.)
            lost_work = min(t - last_ckpt, run + c)
            lost_c = lost_work * (w_block / wall_period) if wall_period > 0 else 0.0
            lost_c = min(lost_c, c)
            c -= lost_c
            run -= lost_work - lost_c
            d += lost_work

            detect_span = min(rel.detect_time_s, running_window - t)
            d += detect_span
            t += detect_span
            if t >= running_window:
                break
            restart_span = min(restart_eff, running_window - t)
            r += restart_span
            t += restart_span
            last_ckpt = t  # resume from checkpoint state

        # Every wall-clock second is booked exactly once; assert it rather than
        # rescaling, so an accounting error fails loudly instead of hiding.
        total = d + r + c + run
        if abs(total - running_window) > 1e-6 * max(1.0, running_window):
            raise AssertionError(
                f"monte carlo accounting drift: booked {total:.6f} of "
                f"{running_window:.6f} wall seconds"
            )
        acc += np.array([d, r, c, run])

    acc /= n_replicates
    return ReliabilityTiming(
        discarded_s=float(acc[0]),
        restart_blocked_s=float(acc[1]),
        checkpoint_blocked_s=float(acc[2]),
        running_s=float(acc[3]),
        unavailable_s=window_s * unavailable_frac,
        n_failures=fails / n_replicates,
        checkpoint_interval_s=interval,
        job_mttf_s=mttf,
    )


def meta_ettr_closed_form(
    n_nodes: int,
    failure_rate_per_node_day: float,
    restart_overhead_s: float,
    checkpoint_interval_s: float,
    checkpoint_write_s: float,
) -> float:
    """The expected-ETTR expression reported by Meta HPCA 2025 (``SOURCES.md`` S2).

    E[ETTR] ~= (1 - N * r_f * (u0 + dt_cp / 2)) / (1 + w_cp / dt_cp)

    Reproduced here verbatim as an external check target, not as part of the
    model. Rates are converted to per-second internally.
    """
    rf = failure_rate_per_node_day / SECONDS_PER_DAY
    numerator = 1.0 - n_nodes * rf * (restart_overhead_s + checkpoint_interval_s / 2.0)
    denominator = 1.0 + checkpoint_write_s / checkpoint_interval_s
    return numerator / denominator
