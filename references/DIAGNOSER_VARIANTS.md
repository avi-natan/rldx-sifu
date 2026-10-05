# Diagnoser variants — unknown-fault-rate sims-study

All variants solve the SAME problem and RANK faults identically; they differ **only in how they
allocate a fixed simulation budget**. This is what makes the rank-vs-budget comparison fair.

## Common setup (identical across every variant)

- **Candidates:** `F` = candidate fault modes (10 per instance incl. identity), `R` = fault-rate grid
  `{0.1, 0.2, …, 1.0}` (10 rates), `G` = **gaps** (the hidden segments of the trajectory, between two
  observed states; a gap's *length* = number of hidden steps).
- **Estimate / the thing we sample:** one `(fault, rate, gap)` triple. Sampling it = run the policy
  through that gap under that fault at that rate, and check whether the simulated end-state matches the
  observed one. `hits` = # matches in `n` sims. Per-gap **Jeffreys hit-rate** `p = (hits+0.5)/(n+1)`.
- **Scoring (ranking):** a fault's score at a rate = `L = Σ_gaps log p`. Fault score = `max over rate`
  of `L`. Faults sorted by score; we report the **rank of the true fault** (1 = best). Jeffreys point
  estimate everywhere, no 1e-12 floor. **Every variant uses this exact scorer.**
- **Budget:** brute-force fixed-N spends `N` sims on *every* estimate, so total `B = N · |F| · |R| · |G|`.
  Every adaptive variant is given the **same B** and spends it non-uniformly. Comparisons are at equal B.
- **Pruning interval (used by the adaptive variants):** 95% **Wilson** score interval per estimate
  (boundary-safe). A gap's "**widest-Wilson-CI**" = the estimate whose interval (in log-space) is widest,
  i.e. the one we are currently least certain about.

---

## 1. brute-force  (`jeffreys_fixedtries_N`)
- **Allocation:** uniform. Exactly `N` sims on **every** `(fault, rate, gap)`. No init/loop/adaptivity.
- **Role:** the reference baseline. Near-optimal at large N (nothing left to exploit); the bar to beat
  at small N.

## 2. UCB, arm = (fault, rate, gap)   (`mg_method=ucb`)
- **Init:** 30 sims on every `(fault, rate, gap)`.
- **Loop:** each step pick the single estimate with the largest
  `UCB(f,r,g) = p_jeffreys(f,r,g) + C·sqrt( ln(all) / n_{f,r,g} )`
  where `all` = total sims so far, `n_{f,r,g}` = sims on that estimate; add a batch to it.
- **Freeze:** drop rates that provably can't be their fault's best, and faults confidently outside
  top-K, so budget flows to contenders.
- **C = √2 ≈ 1.414.** Fine-grained bandit — one arm per estimate.

## 3. UCB, arm = (fault, rate), gap = widest-Wilson-CI   (`mg_method=ucbv`, `GAPMODE=ci`)
- **The combine-v2b-and-UCB variant.** The arm is coarse — a `(fault, rate)` pair, NOT per gap.
- **Init:** 30 sims on every `(fault, rate, gap)` (so all pairs score over the same gaps → comparable).
  Optional budget-scaled floor `MG_UCBV_INITFRAC=α` → init = `max(30, α·N)` (graceful degradation to
  brute at high N).
- **Loop:** UCB chooses **which pair** to invest in:
  `UCB(f,r) = geomean_g p_jeffreys(f,r,g) + C·sqrt( ln(all) / n_arm )`,
  `n_arm` = sims over all of that pair's gaps. Then **v2b's rule chooses which gap**: spend the batch on
  the pair's **widest-Wilson-CI gap** (its most uncertain estimate). Freeze as in #2.
- **C = √2.** Idea: UCB *focuses* budget on promising fault-rates; v2b-style gap pick keeps allocation
  variance-reducing.

## 4. v2b   (`mg_method=v2b`)
- **Init:** 30 sims on every `(fault, rate, gap)`.
- **Loop:** each round, for **every still-contended pair** `(f,r)`, add a batch to that pair's
  widest-Wilson-CI gap. Between rounds: **freeze** rates that can't be a fault's best, and **decide**
  faults whose rank vs both neighbours is already separated. Stop at budget, or early if the full
  ranking is settled (then it lands at *fewer* sims than brute — an even stronger result).
- **Vs #3:** v2b spreads **evenly over all contended pairs**; ucbv-ci (#3) uses a bandit to *focus* on
  the promising ones. Same within-pair (widest-CI) gap rule.

## 5. UCB, arm = (fault, rate), budget split across gaps   (`mg_method=ucbv`, `GAPMODE=even`)
- Same as #3 (coarse `(fault, rate)` arm, UCB pair selection, same init/freeze), **except** within the
  chosen pair the batch goes to the **least-sampled gap** (round-robin) → the pair's budget is divided
  **evenly** across its gaps, instead of chasing the most-uncertain one.
- A **biased** split (e.g. proportional to gap length or to Wilson width) is a future `GAPMODE` option.

---

### One-line contrast of the adaptive trio
| variant | picks which PAIR by | picks which GAP by |
|---|---|---|
| v2b (#4) | round-robin over all contended pairs | widest Wilson CI |
| ucbv-ci (#3) | **UCB** (focus on promising) | widest Wilson CI |
| ucbv-even (#5) | **UCB** (focus on promising) | even round-robin |

Env knobs: `MG_UCB_*` (#2), `MG_UCBV_*` (#3,#5: `_C _N _INIT _INITFRAC _BATCH _FREEZE _TOPK _GAPMODE`),
`MG_V2B_*` (#4). Budget `N` ∈ {50,100,200,400,800}; results root `bruteforce_unknown_eps_experiments`.
