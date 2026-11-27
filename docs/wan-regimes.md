# Communication schedules across sites

`python experiments/wan_regimes.py` produces 432 deterministic scenario
comparisons using `netcap.wan` and the existing ring alpha–beta primitive.
Inputs include RTT, aggregate circuit bandwidth, simultaneous groups, update
bytes, local compute time, bandwidth degradation, latency inflation, fail-slow
participants and declared failure/recovery assumptions.

Both modeled and disconnected outputs use `evidence_class="derived"` and
`input_class="scenario"`: outputs are computed from declared assumptions,
not live-network measurements. Zero bandwidth health with multiple sites
reports `status="disconnected"`, zero useful compute fraction and null
communication/cycle times; convergence remains unvalidated. A single site
needs no WAN exchange and stays modeled at zero bandwidth health.

Finite inputs can still exceed floating-point result range, for example with
extremely small positive health or very large local compute intervals.
`evaluate` raises an actionable `ValueError` instead of returning infinite or
NaN modeled fields. Correct the workload scale or bandwidth assumptions before
retrying. No minimum-health threshold or clamping is applied: tiny and subnormal
values remain supported when the result is finite, including local single-site
operation. Existing finite results and disconnected null fields are unchanged.

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
it returns `None` when the target exceeds the declared search bound. The upper
search bound must have a finite evaluation; otherwise its numeric-range refusal
propagates as `ValueError`. With a finite feasible upper bracket, communication
or cycle overflow at a lower midpoint is treated as insufficient bandwidth,
so a representable solution is still found. Fixed workload inputs do not change
during this search; in every regime, less bandwidth only increases communication
and additive/max cycle time. Other errors propagate rather than becoming `None`.
Latency or failure overhead can make a target unreachable even with more
bandwidth. The inverse retains a nominal `1e-9` Gbps search floor, clamped
to `maximum_gbps` when the caller supplies a smaller maximum. It never returns
a successful answer above that maximum. A target already feasible at the floor
is floor-limited; this is not a claim about the mathematical minimum below it.
The search retains an evaluator-confirmed feasible upper endpoint, uses a safe
midpoint and stops when no representable interior midpoint remains, with a hard
2,048-iteration cap. This corrects the fixed-70-halving loss of accuracy on wide
intervals without changing the model. The returned floating-point endpoint is
conditional on the evaluator and bracket, not a production bandwidth guarantee.
Historical release notes describe the behavior of their original source. Tests cover the two-site analytical case, degradation, zero-link
capacity, local operation and inversion consistency.

These scenario results do not yet price waiting for local capacity versus
remote expansion: no defensible site lead-time or equal-quality training-cost
calibration was added. Existing scale-up, campus and span models remain in
their original modules; this extension does not replace them.
