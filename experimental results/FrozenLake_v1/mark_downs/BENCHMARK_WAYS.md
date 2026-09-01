# FrozenLake fault-diagnosis benchmark — two construction ways

**Decision (2026-09-01): we use WAY 2 only. WAY 1 is NOT used** — it is kept in the code and
in these results purely as a documented baseline that shows why the Taxi-style recipe fails on a
slippery environment.

Both ways build, per map, an execution fault `E` and 10 candidate fault modes for the diagnoser to
rank. A fault mode is a 4-slot map indexed by the **commanded** action whose value is the action
that **actually executes**: `[LEFT->?, DOWN->?, RIGHT->?, UP->?]`, `0=LEFT 1=DOWN 2=RIGHT 3=UP`,
healthy `[0,1,2,3]`. Code: `frozen_lake_fault_modes.py` (`build_way1` / `build_way2`), driver
`multiple_experiment_FrozenLake_fault_benchmark` (`main.py --fl_way {1,2}`).

---

## WAY 2 — distinguishable (CHOSEN ✅)

Per map: run the healthy policy once, count how often each action is commanded.
- **a\*** = the **most-used** commanded action.
- **E** = corrupt a\* to a random other target (seeded).
- **10 candidates** = `healthy` + all 3 redirects of each of the **3 most-used** actions.

Every candidate corrupts a **frequently-commanded** action, so each produces a visibly different
trajectory → the candidates are separable at full observability. **Hardness comes only from partial
observability (hidden states), not from structural ambiguity.**

## WAY 1 — Taxi-style, ambiguous (NOT USED ❌)

- **a\*** = the **rarest** commanded action (count >= 2).
- **10 candidates** = the 4 maps of a\* + **near-twins** on the 2 rarest *other* actions.

Mirrors `hard_taxi_benchmark_v2`. On a slippery env this recipe fails: corrupting rare actions and
adding near-twins makes candidate outcomes overlap (slip smears movement), so the true fault stays
ambiguous **even at full observability**. Kept only to document this negative result.

---

## Evidence for the decision (known fault rate, 4 epsilons done: 0.10/0.07/0.05/0.04)

Avg rank (lower = better), top-1 = % true fault ranked #1.

| slice | way1 rank / top-1 | way2 rank / top-1 |
|---|---|---|
| overall (pooled) | 2.83 / ~20% | **1.93 / ~66%** |
| full obs (100%) | 2.54 / 25.8% | **1.46 / 81.2%** |
| fault rate 0.8, full obs | 2.37 / 29.1% | **1.13 / 91.8%** |

Key points:
- **way2 >> way1** everywhere (~1.9 vs ~2.8 rank; ~66% vs ~20% top-1).
- **way2 clears full observability**: at fault rate 0.8 + 100% vis it hits rank **1.13 / top-1
  91.8% / top-3 95%** — a clean headline.
- **way1 is un-rescuable**: more firings (fr 0.8) and full observability barely move it
  (top-1 ~29% at best) — the structural ambiguity is real. This is exactly why we drop it.
- Epsilon is inert for accuracy (only trades compute); visibility and fault rate are the real axes.

Full data: `frozenlake_way{1,2}_known_fr_epsweep/*.xlsx` (each row tagged `benchmark_way`,
`a_star`, `execution_fault`). way1 files are retained for the baseline comparison only.
