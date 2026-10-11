===============================================================================
 UCB VARIANT 1  --  ARM = (fault, rate, gap)        [experiment folder: ucb_frg_arm]
 mg_method = ucb_frg
 diagnoser  = fault_identification_non_deterministic_PO_unknown_fault_rate_UCB_FRG
              (p_diagnosers.py)
===============================================================================

WHAT THIS METHOD IS
-------------------
Unknown-fault-rate diagnosis by a UCB1 bandit. One ARM = one (fault, rate, gap)
estimate. We are given the SAME total simulation budget as brute-force fixed-N
and we spend it non-uniformly, steering sims toward the estimates that matter.

  Total budget  B = N * |faults| * |rates| * |gaps|      (== brute-force fixed-N)

So every comparison against brute-force / v2b is at EQUAL budget: the only thing
that differs between the methods is HOW the budget is allocated, never the final
scoring.


THE UCB SCORE  (the formula)
----------------------------
For each live arm (f, r, g):

  UCB(f,r,g) = p_jeffreys(f,r,g)  +  C * sqrt( ln(all) / n(f,r,g) )
               \___ exploit ___/      \________ explore ________/

  exploit  p_jeffreys = (hits + 0.5) / (n + 1)        in [0,1]
           - the Jeffreys hit-rate of this estimate (sims whose simulated gap
             end-state matched the observed one). This is the SAME estimate used
             to rank faults at the end (boundary-safe, no 1e-12 floor).

  explore  C * sqrt( ln(all) / n )
           - all = total sims spent so far over EVERY arm
           - n   = sims spent on THIS arm
           - an arm with few samples gets a big bonus -> gets explored; a
             heavily-sampled arm's bonus shrinks.

  C is the exploration constant (see PARAMETERS).


ALGORITHM STRUCTURE
-------------------
  0. INIT
       Pull every (fault, rate, gap) arm INIT_BATCH times (uniform warm-up).
       This gives each arm a first estimate and makes all faults score over the
       SAME gaps, so fault scores are comparable. Then run FREEZE once.

  1. LOOP   (while budget remains)
       a. Pull the single LIVE arm with the largest UCB; add BATCH sims to it.
       b. Re-run FREEZE  -- AFTER EVERY PULL (there is no recheck cadence knob;
          the freeze check is cheap next to the batch of policy rollouts).
       Stops when the budget is exhausted (stop=budget), or early if every fault
       has been decided (stop=all_decided) -- then it lands at FEWER sims than
       brute, which is an even stronger result.

  2. RANK
       Each fault is scored by its BEST rate's Jeffreys log-likelihood summed
       over all gaps:  L(f) = max_r  sum_g log p_jeffreys(f,r,g).
       Faults sorted by L descending. This scoring is IDENTICAL to brute-force
       and v2b -- only the allocation differs. We report the rank of the true
       (injected) fault; lower is better (1 = perfect).


FREEZE  (identical rule to v2b)
-------------------------------
FREEZE prunes hopeless work so budget flows to the real contenders. Two parts:

  (a) RATE freeze
      Within a fault, a rate (f,r) is frozen when its L-interval upper end is
      below another live rate's lower end -> it provably cannot be that fault's
      best rate, so we stop refining it.

  (b) FAULT freeze  -- NEIGHBOUR SEPARATION  (this replaced an earlier top-K rule)
      Rank the faults by score. A fault is "decided" (frozen, gets no more sims)
      once it is confidently separated from BOTH its ranking neighbours:
      its lower bound is above the next fault's upper bound, AND the previous
      fault's lower bound is above its upper bound. In other words, a fault stops
      getting sims once its position relative to the faults directly above and
      below it is locked. (No "top-K": there is no fixed number of contenders.)

Confidence bounds for freeze decisions use the 95% Wilson score interval per
estimate (boundary-safe); ranking itself uses the Jeffreys point estimate.


PARAMETERS
----------
  TUNABLE (the two study variables, read from the environment):
    MG_UCB_FRG_C   C, the exploration constant.  SWEPT: {0.25, 0.5, 1.0, 1.414, 2.0, 3.0, 4.0}
    MG_UCB_FRG_N   N, per-estimate budget.        SWEPT: {50, 100, 200, 400, 800}

  HARDCODED (fixed for the whole comparison; NOT env-tunable):
    INIT_BATCH = 30                     uniform warm-up sims per arm
    FREEZE     = on                     rate freeze + fault (neighbour) freeze, every pull
    BATCH      = 100 if N >= 200 else 20  pull size (matches the brute/v2b batch convention)


WHAT IS RECORDED PER DIAGNOSIS (xlsx columns of interest)
---------------------------------------------------------
    real_fault_rank        rank of the true fault (1 = best)          <- the metric
    total_simulations      sims actually spent (== budget unless it stopped early)
    ucbfrg_rate_freezes    # of (fault,rate) RATE freezes this diagnosis
    ucbfrg_fault_freezes   # of ENTIRE-FAULT freezes this diagnosis
    ucb_picks              # of UCB pulls
    ucb_C, ucb_budget_N    the C and N used
    ucb_stop_reason        "budget" | "all_decided"


IMPLEMENTATION NOTES
--------------------
  - Per-gap Common-Random-Numbers seeding: trace i of a gap uses seed
    base + i*MAX_STATES, so the same random draws are reused across faults/rates
    for a given gap (low-variance comparison).
  - Self-contained: shares no logic with the coarse-arm ucbv variants; only the
    shared infrastructure (execute_one_trace / load_trained_model /
    make_wrapped_env / comparators) is common.

RESULTS LAYOUT (filled by the sweep)
------------------------------------
  ucb_frg_arm/<domain>/c<C>_N<N>/xlsx/...     one xlsx per instance
    domains: FrozenLake_v1, Taxi_v4, MiniGrid_Empty_16x16_v0,
             MiniGrid_SimpleCrossing_S11N2_v0
===============================================================================
