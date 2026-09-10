# Communication schedules across sites

`python experiments/wan_regimes.py` produces 432 deterministic scenario
comparisons using `netcap.wan` and the existing ring alpha–beta primitive.
Inputs include RTT, aggregate circuit bandwidth, simultaneous groups, update
bytes, local compute time, bandwidth degradation, latency inflation, fail-slow
participants and declared failure/recovery assumptions.

The synchronous schedule exchanges every step. The periodic schedule exchanges
after a declared number of local steps. The hierarchical schedule adds a local
reduction cost before the inter-site exchange. The decoupled schedule is an
ideal communication-overlap bound, with no claim that staleness is tolerable.

The 500-step reference is informed by [DiLoCo](https://arxiv.org/abs/2311.08105),
not a reimplementation or a transfer of its convergence findings. Changing
communication frequency does not establish equal learning progress. The
model reports compute-service fraction, not convergence-adjusted training
goodput. No UET, NCCL or production WAN implementation is exercised.

`required_bandwidth` inverts this same model for a target service fraction;
it returns no answer when the target exceeds the declared search bound.
Latency or failure overhead can make a target unreachable even with more
bandwidth. Tests cover the two-site analytical case, degradation, zero-link
capacity, local operation and inversion consistency.

These scenario results do not yet price waiting for local capacity versus
remote expansion: no defensible site lead-time or equal-quality training-cost
calibration was added. Existing scale-up, campus and span models remain in
their original modules; this extension does not replace them.
