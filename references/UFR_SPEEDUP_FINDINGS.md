# Speeding up the unknown-fault-rate (ufr) diagnoser — offline findings

**Question.** The ufr variant costs ~9× the compute of known-fault-rate (it searches ~10 fault-rate
candidates per gap×fault instead of being told the rate) while giving essentially the same avg
real-fault rank. Where can we cut that compute **without hurting rank**?

**Method.** Every ufr result xlsx stores `log_prob_total_per_fault_and_rate` — the full grid of every
candidate fault's log-likelihood at every rate {0.1..1.0}. So we can *re-rank* each instance under any
rate subset and compare to full ufr, entirely offline (no cluster, no re-simulation). Reproduces the
recorded ufr rank exactly on FrozenLake (validation). Script: `experiments_scripts/analyze_ufr_speedup.py`.

> Caveat: these are **optimistic ceilings** — pruning here uses *converged* scores. A real online
> algorithm decides from *noisy partial* estimates, so real savings are lower and rank cost > 0.
> The numbers say where the headroom is; a prototype + timed rerun is needed to bank it.

## Why "just use a high fixed rate (e.g. 1.0)" does NOT work
Re-ranking at each **fixed** rate (FrozenLake, n=1480; avg true-fault rank):

| fixed rate | 0.1 | 0.4 | 0.5 | 0.8 | **1.0** | ufr(adaptive) |
|---|---|---|---|---|---|---|
| avg rank | 2.75 | 2.33 | 2.33 | 2.52 | **3.61 (worst)** | **2.18 (best)** |

`fr=1.0` = "the fault fired on every step"; the moment the true fault left any observed step
*un*-corrupted (constant at rate 0.3–0.5) that history becomes near-impossible → the truth gets buried.
And the best *fixed* rate **slides with the injected rate** (fr 0.3→best 0.2, 0.5→0.4, 0.8→0.6), which
you don't know — and even the best fixed rate loses to ufr, which adapts *per instance*.

## Finding 1 — a coarse *fixed* rate grid is nearly free (the cheapest win)
Δ avg rank vs full 10-rate ufr (≤ ~0.07 everywhere):

| domain | five {0.1,0.3,0.5,0.7,0.9} | four {0.2,0.4,0.6,0.8} | coarse-to-fine (~6) |
|---|---|---|---|
| FrozenLake | +0.029 | −0.003 | +0.003 |
| Taxi | −0.006 | +0.067 | +0.042 |
| Empty 0.5 | −0.009 | −0.021 | −0.015 |
| Cross 0.5 | −0.010 | −0.028 | +0.038 |
| Cross 0.7 | +0.036 | +0.003 | +0.021 |

**Takeaway:** 4–5 fixed rates → **2–2.5× fewer simulations at ~0 rank cost**, a *one-line* change
(shorter candidate list). Coarse-to-fine (~1.7×) is actually *less* speedup than plain 4-rate and more
complex — the data says we don't need it. The rate curve is smooth, so ranking is preserved even when
the exact best rate (often 0.1 in FrozenLake/Taxi) is skipped.

## Finding 2 — candidate elimination: big where separable, impossible on Taxi
Contenders = candidates within 5 logL of the top (of 10):

| domain | median contenders | % instances ≤3 contenders |
|---|---|---|
| FrozenLake | 3 | 75% |
| Cross 0.7 | 3 | 61% |
| Empty 0.5 | 4 | 46% |
| Cross 0.5 | 4 | 42% |
| **Taxi** | **10** | **0%** |

On FrozenLake/MiniGrid ~6–7 candidates are hopeless → confidence-bounded elimination could add ~2–3×.
**On Taxi every candidate stays in contention** — nothing is safely eliminable.

## Finding 3 — difficulty axis explains the Taxi floor
Top1–top2 log-likelihood margin: **Taxi is 92.6% near-ties** (median margin 0.08) — genuine
photo-finishes no cheap trick resolves. FrozenLake/Empty are ~50% near-ties + ~10% "easy" (margin>10,
massively overspent today). Cross-0.7 is the most separable (31% near-ties).

## Direction
Build a **racing / successive-elimination ufr with a coarse rate grid**, all moves confidence-bounded:
1. **Coarse rate grid** (4–5 fixed rates, or coarse-to-fine) — ~2× everywhere, ~0 rank. *Ship first.*
2. **Drop a fault when its upper confidence bound < the leader's lower bound** — prunes to the 3–4 real
   contenders on separable instances; does nothing on genuine near-ties (safe by construction).
3. **Early-stop the instance** when top-1 is confidently settled (the overspent easy ~10%); **cap**
   detected near-ties, where extra compute cannot help.

**Expected ceiling:** ~3–5× on FrozenLake/MiniGrid, ~1.7–2× on Taxi. Taxi reveals the hard floor:
when every candidate is a near-tie, *only* rate-pruning helps — that's the information limit, not a
method failure.

**Orthogonal high-risk/high-reward bet:** importance-sampling re-weighting across rates (simulate once
per gap×fault, re-score all rates) could remove most of the rate factor at ~1× cost — but it can't be
validated offline (needs per-trace records); wants a small variance prototype before committing.
