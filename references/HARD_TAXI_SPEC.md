# Hard Taxi-v4 diagnosis benchmark — design spec

A new script that builds **deliberately hard, curated** Taxi-v4 diagnosis instances: for each
seed we pick a realistic execution fault that fires enough to be diagnosable, then a
difficulty-graded set of candidate faults the diagnoser must rank it against.

Domain facts (see `references/gym_envs.md`): Taxi-v4, 6 actions
`0=South/DOWN, 1=North/UP, 2=East/RIGHT, 3=West/LEFT, 4=Pickup, 5=Dropoff`, 500 states, rainy 0.7,
fickle_passenger off. Fixed map → **variation source is the seed** (start/passenger/destination).

## 1. Per-seed action profile
- **100 seeds.**
- For each seed, roll out the **healthy policy 50 times** (rainy env → trajectories vary across reps).
- Record a **`{action: count}`** profile, **summed over the 50 reps**.
- This is a **proxy** (healthy rollout; the real trajectory is faulty) — good for "will a fault trigger"
  and for ranking actions by how heavily they're used.

## 2. Fault pools
- **Execution-fault pool = 45** realistic single-cause faults:
  - **30 single-redirects** — one action misfires (`a→b`, all else identity; Hamming-1, non-bijective).
  - **15 single-swaps** — one crossed pair (`a↔b`; Hamming-2, bijective).
- **Distractor universe** = the broader simple-map space (redirects / swaps / small combinations),
  mined per seed. Distractors need only be *confusable*, not realistic.

## 3. Execution-fault selection (per seed)
1. Take the seed's **top-3 most-used actions** (by count).
2. Keep faults from the 45-pool whose faulty action `f` is one of those top-3 (so the fault fires a lot
   → well-posed, evidence-rich).
3. Pick **one at random**, RNG **seeded by the seed** (reproducible).
4. **One execution fault per seed → 100 instances.**

Rationale: high evidence on `f` makes the instance **well-posed, not easier** — the difficulty lives in
the candidate set, because near-twins *agree* with E on `f`, so f-evidence can't separate them.

## 4. Candidate set (10 = E + 9 distractors)
**Difficulty metric** (relative to execution fault `E`, on this seed):
```
score(C) = Σ count[a]   for every action a where C(a) ≠ E(a)
```
Lower score = harder (less evidence separates C from E).

The 9 distractors:
- **2 ALTERNATIVE faults** — *disagree* with E on `f`, and corrupt a **heavily-used** action ≠ `f`
  → high score → **easy**. (Test the diagnoser's ability to **reject** a wrong fault.)
- **7 NEAR-TWINS** — *agree* with E on `f` (reproduce the fault signature), differ on a **used non-`f`**
  action → graded by score. (Test **discrimination**.) Tiering:
  - Rank the seed's **used (count > 0)** non-`f` actions **ascending** by count.
  - **lowest nonzero count → HARD (4)**, middle → **MEDIUM (2)**, highest → **EASY (1)**.
  - Fill the 4/2/1 counts by **varying the redirect target** when there aren't enough distinct actions.
- Difficulty totals: **4 hard / 2 medium / 3 easy.**

**Guards:**
- **Never** build a distractor that differs from E only on a **zero-count** action — it'd be behaviorally
  identical to E (degenerate tie), not hard.
- All 9 distinct, none equal to E.
- **Fallback:** if a seed uses too few actions to fill the tiers, vary targets on the same low action;
  if *still* too few, **skip the seed (logged).**

## 5. Instance generation
- The candidate set is built **once per seed** and **reused across** visibility × fault_rate.
- Inherited sweep params: **visibility {20,40,60,80,100}**, **fault_rate {0.5, 0.8}** (a parameter,
  extensible later), **min trajectory length = 25** (Taxi-specific).
- 100 seeds × 5 visibility × 2 fault_rate = **1000 diagnosis instances**.
- Counts for ranking use the **summed-over-50** profile (per-trajectory gives identical ordering).

## 6. Deferred (out of scope of this spec)
- **Masking-survival guarantee:** with random hiding, a hard distractor's rare differing action can be
  fully masked at low visibility → that instance becomes *impossible* (forced tie). Accepted for now;
  no special masking logic.
- **Rank metric refinements** (e.g. handling tied/impossible instances) — handled separately, not here.
