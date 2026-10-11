# v1 unknown-fault-rate diagnoser — algorithm and the reasoning behind it

This document explains the **v1** unknown-fault-rate (ufr) diagnoser: not just *what* it does, but
*why each idea was reached* — the observation that motivated each step. It is written as the line of
reasoning that led from the brute-force diagnoser to v1.

Notation used throughout:
- **fault** `f`: a candidate fault mode (≈10 of them).
- **rate** `r`: a candidate fault-firing rate (≈10 of them). A **pair** is one `(f, r)`.
- **gap** `g`: a hidden segment of the trajectory — we observe its start state and end state but not
  what happened in between (partial observability).
- **p̂(f, r, g)**: Monte-Carlo estimate of the probability that simulating fault `f` at rate `r`
  across gap `g` reproduces `g`'s observed end state.
- **L(f, r)** = Σ_g log p̂(f, r, g): the log-likelihood score of a pair. A fault's score is its best
  rate's L; we rank faults by that; the metric is the true fault's rank.

---

## Part 1 — from a single-simulation CI to an L CI

### Where we started: the brute-force diagnoser

The full (brute-force) diagnoser works estimate-by-estimate. For **every** pair and **every** gap it
runs an adaptive Monte-Carlo loop that keeps drawing traces for that one `(f, r, g)` until the
estimate is precise:

```
keep sampling p̂(f, r, g) until  margin = 1.96 · sqrt( p̂(1-p̂) / n )  <  epsilon
```

So the object it is confident about is a **single simulation estimate** — one p̂, with its own
confidence interval `[p_lo, p_hi]`. Call this the **single-sim CI**. Only after every p̂ is pinned to
±epsilon does it sum the logs into `L(f, r)`, take each fault's best rate, and rank.

### The observation that started v1

The full diagnoser spends **all** its budget making each p̂ precise. But look at what those p̂ are
actually *used for*: nothing, except being summed into `L`, and the `L`s are only ever compared to
**rank** faults. The individual p̂ are intermediate values we never report and never decide on.

That is the whole tension in one sentence:

> **We were paying for precision on a quantity (each p̂) that is not the quantity we decide on (L,
> and ultimately the ranking).**

Two concrete ways that misallocates budget:

- **Wasted precision.** A gap where p̂ is 0.03 vs 0.05 — full drives *both* down to ±epsilon, yet
  for ranking it barely moves anything: the log terms are close and this fault may be far from its
  rivals anyway. Budget spent, decision unchanged.
- **Misplaced precision.** The real question is "*is fault A's L above fault B's L?*" Full never
  asks that. If A and B are already far apart, we needed **none** of their p̂ precise; if A and B are
  a genuine near-tie, **no** amount of per-p̂ precision will separate them. Either way, "make every
  p̂ precise" is not aimed at the decision.

### The reframe: put the CI on L, not on p̂

The decision quantity is `L(f, r) = Σ_g log p̂(f, r, g)`, and the ranking depends only on the
**differences** between pairs' `L`. So the thing we should be confident about is **L itself**.

That is the pivot of the whole method:

> **Move the confidence interval up one level — from each single simulation estimate (p̂) to the
> aggregate score we actually rank on (L).** Then don't sample to a precision target; sample only
> while competing `L`-intervals **overlap**, and stop the moment they **separate**.

Precision on p̂ stops being the *goal*; it becomes the *raw material*. The **stop criterion** moves
from "each p̂ within epsilon" to "the L-intervals that matter no longer overlap."

### How the L-CI is built from the single-sim CIs

Each gap still hands us a single-sim CI: `p̂(g) ∈ [p_lo(g), p_hi(g)]`, from the same Wald margin
`1.96·sqrt(p̂(1-p̂)/n)`. Because `L` is a **monotone increasing** sum of logs, the simplest way to
propagate those intervals up is endpoint-by-endpoint:

```
L_lo = Σ_g log p_lo(g)          # all gaps simultaneously at their low ends
L_hi = Σ_g log p_hi(g)          # all gaps simultaneously at their high ends
L    ∈ [L_lo, L_hi]
```

This is the **loose L-CI**: a valid, conservative bound that assumes the worst case of every gap
landing at the same end at once. Its width is the **sum of the per-gap half-widths**.

> A tighter alternative treats gaps as independent and adds **variances** instead of widths:
> `Var(log p̂) ≈ (1-p)/(n·p)` (delta method), so the half-width is
> `1.96·sqrt( Σ_g (1-p)/(n·p) )` — a *sqrt-of-sum-of-squares*, which is much narrower than the
> sum-of-widths above. **v1 deliberately uses the loose version:** it is simpler and *safe* (it will
> not separate two `L`s prematurely), at the cost of being over-wide — which makes the race slower to
> call than it strictly needs to be. (This is a known lever for a later version.)

### Why this reframe is the hinge of everything that follows

Once the object of confidence is `L` rather than each p̂, every later step in v1 is just an operation
on **L-intervals**:

- **Freezing rates** (Part 2): compare a fault's rates by their L-intervals; kill a rate whose whole
  interval sits below another's.
- **Ordering faults** (Part 3): compare faults by their L-intervals; stop when adjacent ones
  separate.
- **Where to spend the next trace** (Part 4): the gap whose single-sim CI contributes the most
  *width* to the L-interval — the widest-gap rule.

None of those are expressible in the brute-force framing, because there the unit of confidence was a
lone p̂. Lifting the CI to `L` is what makes racing, freezing, and targeted sampling possible.

---

## Part 2 — freezing rates inside a fault  *(to be written)*

## Part 3 — racing faults: the overlap/decided test  *(to be written)*

## Part 4 — where to sample: the widest-gap rule  *(to be written)*

## Part 5 — the per-estimate epsilon stop, and why it turned out redundant  *(to be written)*
