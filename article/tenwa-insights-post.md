Both numbers can be true at once, and the gap between them is where most infrastructure budget arguments actually live.

I built an open model to close that gap, and to answer a question that comes up constantly in capacity planning and almost never gets answered with numbers: when is it better to spend the next dollar on network and reliability than on more accelerators?

The short answer is that it depends on the regime, more than I expected, and often not in favor of the network. The longer answer is below. The code, the raw results, and the paper are linked at the end.

## The measurement problem

Two metrics dominate discussions of large training clusters.

**Model FLOPs Utilization** compares achieved model FLOPs against peak, over the wall-clock of a step that is running. It is a good measure of how well a step executes. It is blind to failures. A job that loses a third of its life to restarts can still post a respectable MFU, because MFU never looks at the time the job was not stepping.

**Effective Training Time Ratio**, introduced by Meta in their HPCA 2025 reliability paper, compares productive runtime against the job's available wall-clock. It is a good measure of availability. But it counts ordinary stepping as productive, so time a GPU spends blocked on a collective is credited as useful work. It also only sees capacity the job actually held, so spare and stranded hardware is invisible to it.

Both are useful. Neither answers "what fraction of the accelerators I paid for turned into model progress," and multiplying them does not either, because their denominators are different things.

## Four fates

The model I built starts from a deliberately boring accounting identity. Take a pool of N accelerators over a window of T seconds. That is N×T accelerator-seconds, and every one of them ends up in exactly one of four states:

- **Productive**: computation that is present in the model you finally ship.
- **Blocked**: powered and allocated but not computing. Exposed collective communication, synchronization wait behind a slow peer, pipeline bubble, checkpoint stall, restart.
- **Discarded**: computed and then thrown away. Everything between the last durable checkpoint and the moment a failure was detected.
- **Unavailable**: owned but not usable. Down awaiting repair, held as a spare, or stranded behind a domain or power limit.

The identity closes exactly, and the model asserts it on every evaluation.

Two conventions do the real work. First, a second is classified by what eventually happened to it, not by what the accelerator was doing. A second spent computing inside a window that was later thrown away is discarded, not productive. Second, re-execution is counted once: after a restart, the original attempt is discarded and the repeat is ordinary work. Charging both is the single most common error in informal goodput arithmetic, and it is why casual estimates of "time lost to failures" tend to be too large.

Applied to a 16,384-accelerator reference configuration training a 405B dense model, the split is 62.7% productive, 29.9% blocked, 4.5% discarded, 2.9% unavailable. The same configuration reports an ETTR of 0.889. Neither number is wrong. They measure different things, and only one of them is what you bought.

![Where paid-for accelerator time goes as job size grows, and what the blocked time consists of](https://raw.githubusercontent.com/dimaggi-ai/network-vs-more-gpus/main/figures/fig1_where_time_goes.png)

*The productive share falls from 78% at 1,024 accelerators to 38% at 65,536. Exposed communication stays roughly flat as a share of the total; what grows is pipeline bubble and failure-related time.*

## What "worth N GPUs" should mean

Infrastructure teams already speak in equivalent-GPU terms: "this recovers the equivalent of 200 GPUs." The usual arithmetic is to add up the idle and repeated accelerator-seconds you avoided and divide by the measurement period.

That arithmetic quietly assumes a purchased accelerator is fully productive. It is not. It inherits whatever inefficiency the fleet already has, and it makes that inefficiency slightly worse, because it adds communication volume and shortens the job's mean time to failure.

So I defined the comparison the other way around. **Substitution-Equivalent Accelerators** is the number of accelerators you would have to buy, at your current configuration, to get the same productive throughput the improvement gives you. You solve for the accelerator count that matches, rather than asserting an equivalence.

At the reference baseline, a marginal accelerator contributes 0.502 productive accelerator-seconds per accelerator-second purchased. So an improvement worth 141 productive accelerators is worth 281 purchased ones. The informal method would have told you 141.

That gap widens fast. The informal figure is 73% of the true substitution value at 2,048 accelerators and 22% at 65,536. It understates the case for infrastructure work by 1.4x at small scale and by more than 4.5x at large scale, which is exactly backwards from where you want your estimate to be reliable.

![The informal equivalent-GPU figure as a fraction of the substitution value, falling with scale](https://raw.githubusercontent.com/dimaggi-ai/network-vs-more-gpus/main/figures/fig5_naive_bias.png)

*The bias is structural: the informal metric implicitly divides by a marginal productivity of 1, and true marginal productivity falls with scale.*

There is a practical benefit to defining it this way. Substitution-Equivalent Accelerators **is** a break-even cost. If an intervention costs less than that many fully-loaded accelerators, fund it. The rule needs a ratio, not a price, which is why this project quotes no dollar figures anywhere. I do not know your capital costs, and inventing them would have added precision without accuracy.

## The answer changes with the regime

Here is the part I did not expect.

I swept node failure rate against cross-pod oversubscription at two cluster sizes and asked which single intervention has the highest substitution value in each cell. Across 96 cells that fall inside the model's scope, **three different interventions win**, and the winning mix is different at the two sizes.

![Decision map showing which intervention has the highest substitution value across failure rate and oversubscription, at two cluster sizes](https://raw.githubusercontent.com/dimaggi-ai/network-vs-more-gpus/main/figures/fig3_decision_map.png)

*Each cell is an operating regime. Blank cells fall outside the model's declared scope.*

At 16,384 accelerators, the map splits evenly between straggler control at lower failure rates and halving the node failure rate at higher ones. Additional bandwidth never wins anywhere on that map.

At 65,536 accelerators, reliability dominates most of the map, quadrupled bandwidth takes the low-failure, heavily-oversubscribed corner, and straggler control keeps a low-failure fringe. Faster restart never ranks first, yet it is still worth hundreds to thousands of accelerators in absolute terms. Ranking first and being worth funding are different questions.

Because point rankings are not decisions, I re-ran the ranking across 323 draws from documented parameter ranges, letting kernel efficiency, overlap fractions, detection time, restart time, checkpoint cost, and jitter all vary. At 16,384 accelerators: cutting the blocking checkpoint cost to 5 seconds ranks first in 45.8% of draws, halving the failure rate in 29.1%, straggler control in 15.8%, faster restart in 8.7%, and quadrupled bandwidth in 0.6%.

![Share of parameter draws in which each intervention ranks first](https://raw.githubusercontent.com/dimaggi-ai/network-vs-more-gpus/main/figures/fig6_rank_stability.png)

*Rank stability over 323 draws from documented parameter ranges at 16,384 accelerators.*

I started this project with a thesis that network capacity is compute capacity. At the most common operating point I examined, additional bandwidth is almost never the best marginal investment. That is a real finding and I am reporting it rather than reframing it.

## Where the thesis does hold

It holds in specific, identifiable places, and they are worth knowing.

**When communication is a large share of a small step.** Shorten the sequence length to 2,048 tokens or the global batch to 512 sequences, and doubling bandwidth becomes worth roughly 900 accelerators, because communication now dominates a step that has less compute in it.

**At the largest scales with an oversubscribed spine.** This is the corner where bandwidth wins outright at 65,536 accelerators.

**When nothing can be bought instead.** At roughly 131,000 accelerators under this configuration, buying is nearly worthless and soon harmful: productive throughput rises just 1.2% to a peak of 31,566 at 1.25x the pool, then falls to 29,628 at 2x. Meanwhile halving the failure rate takes it to 38,640, beyond anything the scaling curve reaches. There is no accelerator purchase that matches the infrastructure improvement. This is the strongest form of the thesis, and it is confined to a regime most operators are not in.

Equally worth knowing is where it clearly fails. Give the model an already-fast, already-flat network and a further quadrupling of bandwidth is worth 47 accelerators, three tenths of a percent of the pool, while halving the failure rate is worth 954. Run a 1,024-accelerator job and nothing is worth much, because 78% of that pool is already productive. Small jobs do not justify infrastructure investment on capacity grounds.

## One result that surprised me

I expected these interventions to be substitutes: fix one bottleneck and the others matter less. For network work that held. Bandwidth and topology together are worth about 7% less than the sum of their parts.

Recovery work behaves the opposite way. Faster detection, faster restart, and cheaper checkpoints together are worth about 9% **more** than the sum of their parts. The mechanism is straightforward once you see it: cheaper checkpoints shorten the optimal checkpoint interval, a shorter interval shrinks the window of work you throw away on failure, and a smaller discarded window makes fast detection and fast restart more valuable. They compound.

The practical version: recovery improvements should be funded as a program, not ranked against each other as competing line items.

## What this is not

Every number here is analytical or simulated. Nothing in this project was measured on hardware I operated. That is the central limitation and it bounds everything above.

What I could do is validate against published measurements. I calibrated one compute-side parameter on Meta's published 8,192-GPU Llama 3 configuration, then held it fixed and predicted their 16,384-GPU configuration, which doubles data parallelism and spreads the gradient all-reduce across twice as many pods. The prediction landed at 406.3 TFLOP/s per GPU against a published 400, a 1.6% error, with no network or reliability parameter touched. The model also reproduces their reported job mean-time-to-failure at 16,384 and 131,072 GPUs within a couple of percent, and independently recovers Daly's optimal checkpoint interval and Meta's own closed form for expected ETTR.

It also fails in a way worth stating. Meta's *measured* 1,024-GPU MTTF is 7.9 hours; inverse scaling from their per-node hardware failure rate predicts about 29 hours. Their own published figures are not mutually consistent under inverse scaling. Whatever dominates interruptions for smaller jobs is not per-node hardware failure, and this model does not represent it. Results below about 4,096 accelerators should be read as indicative only.

The first validation run also failed outright, predicting an ETTR of 0.60 against a published value above 0.90. The cause was two inputs I had specified wrongly rather than anything in the model: I had assumed no replacement capacity and 60-second fully blocking checkpoint writes, both of which the source paper contradicts. Worth noting because a useful piece of arithmetic fell out of the diagnosis: at the optimal checkpoint interval, combined checkpoint and lost-work overhead is roughly the square root of twice the checkpoint cost over the mean time to failure. With a 60-second blocking write and a 2.3-hour MTTF, that is about 12%, so an ETTR above 90% is simply unreachable at that scale with blocking checkpoints that slow, no matter what else you fix.

Two further corrections came out of an adversarial review of the code by a reviewer with no prior context, and both changed published numbers. One was an accounting bug in the event-driven reliability path that had made the fast path look inaccurate when it was not. The other was that interventions worded as ratios were implemented as absolute targets, which silently changed their strength across the decision map. Both are written up in the repository's decision log rather than quietly absorbed, and every number above was regenerated after the fixes.

Beyond that: the failure process assumes independent per-node failures, so correlated failures would look worse than modeled and would raise the value of fault-isolation work I do not represent. I assume dense, regular rank placement, which is the best case for the network and therefore conservative with respect to my own thesis. No published source reports a system measured before and after a network change, so the substitution metric's inputs are validated but its output is not. And the scope is dense transformer pre-training, synchronous, single campus. No inference, no mixture-of-experts, no cross-campus pooling.

## Three things to take away

**Measure capacity against what you purchased.** A fleet at 89% effective training time may be turning 63% of its purchased accelerator-seconds into progress. Both are real numbers and only one of them is the budget conversation.

**Ask for break-even costs, not recovered time.** "This is worth up to 971 accelerators in our current regime" is a fundable statement. "This recovers 480 GPU-equivalents" is usually an understatement by a factor that grows with your cluster.

**Distrust any single ranking.** The best marginal investment at 16,384 accelerators with reliable nodes is not the best at 65,536 with an oversubscribed spine. Anyone offering a universal ordering of infrastructure investments, including me at the start of this project, has not checked the regimes.

---

The paper, the model, the configurations, and every raw result are open source under an MIT license. The full experiment program runs in about ten seconds on a laptop and needs no accelerator access; `make reproduce` regenerates every number and figure in this article. Raw results are immutable and carry provenance records, and every validation threshold was fixed before the corresponding comparison was run.

**Repository:** [github.com/dimaggi-ai/network-vs-more-gpus](https://github.com/dimaggi-ai/network-vs-more-gpus)
**Paper (PDF):** [main.pdf](https://github.com/dimaggi-ai/network-vs-more-gpus/blob/main/paper/main.pdf)

*A preprint is being prepared for arXiv. This article will be updated with the link when it is posted.*
