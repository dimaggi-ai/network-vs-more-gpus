# The latency-regime atlas

Which cut of a training job survives being spread across two halls, at what
distance, over what circuit — and what actually decides the answer.

Everything below is computed on the study baseline
(`configs/scenarios/reference_405b_16k.yaml`, 16,384 H100, 405B dense,
tp 8 / pp 16 / dp 128) and is reproducible with
`python experiments/run_span_experiments.py`. Numbers quoted from the validation
registry are computed on the untuned Llama 3 scenario instead and will differ in
the last digits; each is labelled where it appears. That split is D8.

## The short version

The question everyone asks about scale-across is "how far apart can the halls
be?" On this model that is close to the wrong question. Stretching the stitch
from a campus (40 µs) to a coast-to-coast path (40 ms) — a thousandfold change
in round-trip time — costs the data-parallel cut 1.2 points of useful capacity
and the pipeline cut 0.5 points. Changing nothing about the fiber and only
reordering the ranks costs the pipeline cut **6.5x** and the data-parallel cut
**2.7x**.

Distance is not free, but at realistic circuit widths it is nowhere near the
binding constraint. Rank placement, circuit width, and how many groups share the
circuit are the binding constraints, and all three are software.

## The model addition

`TopologySpec` gains a fourth tier. It is unlike the other three in one
important way: scale-up, pod, and cross-pod bandwidth are per-accelerator
entitlements, while a stitch is an **aggregate shared circuit**. Which cut
crosses it therefore determines how many groups contend for it, and contention
scales with the cluster while the circuit does not.

Three details carry most of the result, all recorded in D16 and D17:

- **Latency is exposed, not overlapped.** A ring's bandwidth term streams
  alongside the backward pass, but its `2(k-1)` chained hops are serially
  dependent and drain at the step boundary. A model that overlaps them concludes
  distance is free.
- **A rank cannot outrun its own NIC.** Without that cap, an arbitrarily wide
  circuit lets the span tier beat the fabric the ranks are attached to.
- **A pipeline's crossing fraction is a placement question.** Contiguous stage
  blocks put `halls - 1` of the `pp - 1` boundaries on a hall edge; round-robin
  interleaving puts every one of them there.

Every span term is exactly zero when `halls == 1`, so no previously published
result in this repository moves.

## The atlas

Useful capacity retained, spanned over hall-local, at a 400 Gbit/s circuit.
Retention of 1.00 means the stitch cost nothing; 0.50 means half the pool's
useful capacity was spent on being split.

**Contiguous placement** (what a topology-aware launcher produces):

| cut | scale-up | campus | metro | region |
|---|---|---|---|---|
| data-parallel | 0.965 | 0.965 | 0.964 | 0.953 |
| checkpoint | 0.880 | 0.880 | 0.880 | 0.880 |
| pipeline | 0.607 | 0.607 | 0.607 | 0.602 |
| tensor-parallel | 0.004 | 0.004 | 0.004 | 0.004 |

**Interleaved placement** (what an unaware launcher produces):

| cut | scale-up | campus | metro | region |
|---|---|---|---|---|
| data-parallel | 0.362 | 0.362 | 0.360 | 0.230 |
| checkpoint | 0.880 | 0.880 | 0.880 | 0.880 |
| pipeline | 0.093 | 0.093 | 0.093 | 0.092 |
| tensor-parallel | 0.001 | 0.001 | 0.001 | 0.001 |

Read the rows, not the columns. The columns barely move; the tables differ from
each other everywhere.

Three rows are worth naming individually. **Tensor-parallel never survives** at
any distance or placement — it pays the stitch once per layer per micro-batch,
and it is the only cut whose retention responds materially to round-trip time.
**Checkpoint traffic is perfectly distance-flat**, because it is pure bandwidth
with nothing to overlap and nothing serially chained. **Pipeline** is the cut
where the launcher decides everything.

## What orders the atlas

Not the cut type, and not the raw bytes crossing the boundary. On this plan the
pipeline cut moves 586 GB across the stitch and the data-parallel cut moves 68
GB — roughly nine times less — and yet the pipeline cut retains far less. The ordering
follows **exposed** bytes: bytes weighted by what cannot be hidden behind
compute.

| cut | bytes across | hideable | exposed |
|---|---|---|---|
| tensor-parallel | 277,077 GB | 10% | 249,369 GB |
| checkpoint | 5,670 GB | 0% | 5,670 GB |
| pipeline | 586 GB | 70% | 176 GB |
| data-parallel | 68 GB | 85% | 10 GB |

That rule is checked as an ordering across four parallel plans in
`validation/validate_span.py`, with no threshold to tune.

## Width, and where it stops helping

Circuit width matters a great deal in the deployed range and then stops
mattering abruptly. The pipeline cut at metro distance:

| circuit | 100G | 400G | 800G | 1.6T | 3.2T | 12.8T | 204.8T |
|---|---|---|---|---|---|---|---|
| retention | 0.278 | 0.607 | 0.755 | 0.861 | 0.925 | 0.981 | 0.999 |

The curve flattens once each rank's share of the circuit reaches its own NIC
line rate. Past that point doubling the circuit changes nothing at all — which
is the mechanism behind Corning's bandwidth-insensitivity result, and also the
reason this repository declines to treat that result as a calibration (D18).

Retention is reported only at or below 3.2 Tbit/s. Above 12.8 Tbit/s the tier
model develops a genuine artifact — a split job comes out up to 0.36 percent
*faster* than a hall-local one, because each hall's residual group occupies
fewer pods — and the wider probe circuit is used only inside fixed-width
comparisons, where the effect cancels on both sides.

## A stitch does not scale with the cluster it serves

Holding a 120 km, 800 Gbit/s circuit fixed and growing the job:

| accelerators | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 |
|---|---|---|---|---|---|
| pipeline | 0.919 | 0.854 | 0.755 | 0.631 | 0.506 |
| data-parallel | 0.993 | 0.990 | 0.982 | 0.968 | 0.947 |

This is the model's sharpest warning against reading a successful two-site field
trial as a frontier-scale result. Contention grows with the cluster; the circuit
does not.

## Pricing a stitch

`stitch_equivalent_accelerators` answers the operator's question directly: a
circuit connects a second hall's worth of accelerators, so how many of them
actually arrive? At 400 Gbit/s connecting 8,438 accelerators, the data-parallel
cut delivers 74 percent of them at campus distance and 72 percent at region
distance; the pipeline cut delivers 9 percent. Those are ratios, not prices —
the research charter forbids currency figures, and the dollar comparison lives
in `compute-power-placement`.

## What this does not tell you

The validation registry prints its own DECLINED list; run
`python validation/validate_span.py` to see it in full. The substantive gaps:

- **Optical-layer behaviour.** A stitch here is a bandwidth and a latency.
  Insertion loss, bit-error rate, circuit flap, and retune are not modelled, and
  neither is the question of whether the declared circuit is the measured one.
- **Inference and interactive SLOs.** Useful capacity is a training quantity.
  `edge-continuum-placement` owns the latency-domain gates.
- **Asynchronous training.** The model is synchronous-only; DiLoCo-class methods
  change the communication structure the atlas assumes.
- **More than two halls.** Implemented and general, but checked against nothing
  published, so nothing is reported.
- **Corning's 26x cliff magnitude and its bandwidth-doubling claim.** Declined
  for the reasons in D18. The short-distance claim is reproduced.

## Reproducing

```
python experiments/run_span_experiments.py   # results/raw/s1..s7
python figures/make_span_figures.py          # fig8, fig9
python validation/validate_span.py           # the registry, and its DECLINED list
pytest tests/test_span.py -s                 # invariants, mutations, blind spots
```

The mutation tests are the reason to trust the registry: each deletes a piece of
span machinery and requires the registry to go red, printing the red set it
actually produced. `test_registry_blind_spots` prints the points that no
mutation kills, which are the registry's own limits.
