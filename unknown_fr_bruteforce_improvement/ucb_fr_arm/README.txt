===============================================================================
 UCB VARIANT 2  --  ARM = (fault, rate), gap chosen by WIDEST-Wilson-CI   [folder: ucb_fr_arm]
 mg_method = ucb_fr
 diagnoser  = fault_identification_non_deterministic_PO_unknown_fault_rate_UCB_FR
              (p_diagnosers.py, right after UCB_FRG)
===============================================================================

WHAT THIS METHOD IS
-------------------
The combine-v2b-and-UCB idea. A bandit over the COARSE (fault, rate) arm -- there are
len(faults)*len(rates) arms, NOT one per gap. UCB decides WHICH (fault, rate) pair to invest the
next batch in; v2b's own rule decides WHICH GAP of that pair the batch is spent on. So UCB *focuses*
budget on the promising fault-rates, and v2b *allocates* within the chosen pair.

Total budget  B = N * |faults| * |rates| * |gaps|   (== brute-force fixed-N), so every comparison
against brute-force / v2b / ucb_frg is at EQUAL budget: only the ALLOCATION differs, never the scoring.


THE UCB SCORE  -- and exactly what we compute inside it
-------------------------------------------------------
For each live arm (f, r):

    UCB(f, r) = arm_reward(f, r)  +  C * sqrt( ln(all) / n_arm )
                \___ exploit ___/     \______ explore ______/

--- the pieces ---

  A (fault, rate) pair spans several GAPS (the hidden trajectory segments). For each gap g we keep:
      hits_g = number of simulated traces through gap g (under fault f, rate r) whose END state
               matched the observed end state of that gap,
      n_g    = number of sims spent on gap g for this pair,
      p_g    = (hits_g + 0.5) / (n_g + 1)      <- the per-gap JEFFREYS hit-rate (in (0,1),
                                                   boundary-safe; the SAME estimate used to rank).

  EXPLOIT = arm_reward(f, r) = GEOMETRIC MEAN of the per-gap rates over the pair's G gaps:
        arm_reward = ( p_g1 * p_g2 * ... * p_gG ) ^ (1/G)
                   = exp( (1/G) * sum_g log p_g )
                   = exp( L / G ),   where L = sum_g log p_g is the pair's log-likelihood.
     Why the geometric mean (not the plain average): the fault is finally RANKED by
     L = sum_g log p_g, so exp(L/G) is just the per-gap-normalised version of the exact quantity we
     rank by -- it keeps exploit aligned with the ranking AND lands in [0,1], which is what UCB1's
     exploration term is calibrated for. (One gap => it reduces to the single p_g.)

  EXPLORE = C * sqrt( ln(all) / n_arm ):
      all    = total number of sims spent so far across EVERY arm (the running total).
      n_arm  = total sims spent on THIS pair, summed over ALL its gaps:  n_arm = sum_g n_g.
      C      = the exploration constant (the knob we sweep).
     A pair that has had few sims (small n_arm) gets a big bonus -> it gets explored; a
     heavily-sampled pair's bonus shrinks. Both `all` and `n_arm` are counted in SIMULATIONS (not in
     "number of pulls") so the two terms stay on a consistent scale and match the equal-budget axis.

--- worked example ---
  Pair (F, R) with 2 gaps:
     gap A: 72 hits / 80 sims  -> p_A = 72.5/81 = 0.895
     gap B:  6 hits / 20 sims  -> p_B =  6.5/21 = 0.310
     arm_reward = sqrt(0.895 * 0.310) = 0.527        n_arm = 80 + 20 = 100
  Suppose all = 50,000 sims so far and C = 1.4142:
     explore = 1.4142 * sqrt( ln(50000) / 100 ) = 1.4142 * sqrt(10.82/100) = 1.4142 * 0.329 = 0.465
     UCB(F,R) = 0.527 + 0.465 = 0.992
  The arm with the largest UCB across all live pairs is pulled next.


ALGORITHM STRUCTURE
-------------------
  0. INIT
       Seed EVERY (fault, rate, gap) with INIT_BATCH sims (uniform warm-up). This gives every arm a
       first estimate and makes all faults score L over the SAME gaps -> fault scores are comparable.
       Then run FREEZE once.

  1. LOOP   (while budget remains)
       a. Pull the max-UCB LIVE arm (fault, rate).
       b. Spend the BATCH on that pair's WIDEST-Wilson-CI gap -- the gap whose current estimate is
          the MOST UNCERTAIN (widest 95% Wilson interval, in log-space). This is exactly v2b's
          within-pair rule: put sims where the estimate most limits the ranking.
       c. Every RECHECK pulls, re-run FREEZE.
       Stops at the budget (stop=budget), or early if every fault is decided (stop=all_decided) ->
       then it lands at FEWER sims than brute, an even stronger result.

  2. RANK
       Each fault scored by its BEST rate's Jeffreys log-likelihood summed over all gaps:
       L(f) = max_r sum_g log p_jeffreys(f,r,g). Faults sorted descending. The true fault's rank is
       reported (1 = best). IDENTICAL scoring to brute-force / v2b / ucb_frg.


FREEZE  (identical rule to v2b and ucb_frg -- NEIGHBOUR SEPARATION)
------------------------------------------------------------------
  (a) RATE freeze: a rate (f,r) whose L-interval upper end is below another live rate's lower end
      provably cannot be that fault's best rate -> stop refining it.
  (b) FAULT freeze: rank the faults by score; a fault is "decided" (frozen, no more sims) once it is
      confidently separated from BOTH its ranking neighbours (its lower bound above the next fault's
      upper bound, AND the previous fault's lower bound above its upper bound). No top-K.
  Confidence bounds use the 95% Wilson score interval; ranking itself uses the Jeffreys point estimate.


PARAMETERS
----------
  TUNABLE (the two study variables, read from the environment):
    MG_UCB_FR_C   C, the exploration constant.  SWEPT: {1.4142 (=sqrt 2), 2.0}
    MG_UCB_FR_N   N, per-estimate budget.        SWEPT: {50, 100, 200, 400, 800}

  HARDCODED (fixed for the whole comparison; NOT env-tunable):
    INIT_BATCH = 30                     uniform warm-up sims per (fault,rate,gap)
    FREEZE     = on                     rate freeze + fault neighbour-separation freeze
    RECHECK    = 20                     re-run FREEZE every 20 pulls
    BATCH      = 100 if N >= 200 else 20  pull size (matches the brute/v2b batch convention)
    gap pick   = widest Wilson CI       v2b within-pair rule


WHAT IS RECORDED PER DIAGNOSIS (xlsx columns of interest)
---------------------------------------------------------
    real_fault_rank        rank of the true fault (1 = best)          <- the metric
    total_simulations      sims actually spent (== budget unless it stopped early)
    ucbfr_rate_freezes     # of (fault,rate) RATE freezes this diagnosis
    ucbfr_fault_freezes    # of ENTIRE-FAULT freezes this diagnosis
    ucbfr_picks            # of UCB pulls
    ucbfr_C, ucbfr_budget_N  the C and N used
    ucbfr_stop_reason      "budget" | "all_decided"


HOW IT DIFFERS FROM ucb_frg (variant 1)
---------------------------------------
  ucb_frg: arm = (fault, rate, GAP); UCB directly picks a single gap-arm to sample.
  ucb_fr : arm = (fault, rate);      UCB picks the PAIR, v2b's widest-Wilson-CI rule picks the gap.
  Both: same Jeffreys ranking, same v2b neighbour-separation freeze, same equal budget.


IMPLEMENTATION NOTES
--------------------
  - Per-gap Common-Random-Numbers seeding: trace i of a gap uses seed base + i*MAX_STATES, so the
    same random draws are reused across faults/rates for a given gap (low-variance comparison).
  - Self-contained: shares no logic with the other variants; only the shared infrastructure
    (execute_one_trace / load_trained_model / make_wrapped_env / comparators) is common.

RESULTS LAYOUT (filled by the sweep)
------------------------------------
  ucb_fr_arm/<domain>/c<Ctag>_N<N>/xlsx/...     batched (~10 instances per file)
    Ctag in {1_414, 2_0};  N in {50,100,200,400,800}
    domains: FrozenLake_v1, Taxi_v4, MiniGrid_Empty_16x16_v0, MiniGrid_SimpleCrossing_S11N2_v0
===============================================================================
