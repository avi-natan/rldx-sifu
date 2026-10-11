===============================================================================
 WWMG  --  Widest-Wilson-Margin-Gap allocation   [folder: wwmg]   (formerly "v2b")
 mg_method = wwmg
 diagnoser  = fault_identification_non_deterministic_PO_unknown_fault_rate_WWMG  (p_diagnosers.py)
 NOTE: this method used to be called "v2b" -- renamed because that label was not meaningful.
       The algorithm is unchanged; old v2b_N* result folders are the same method under the old name.
===============================================================================

WHAT THIS METHOD IS
-------------------
The smart-allocation baseline. Given the SAME total budget as brute-force fixed-N
(B = N * |faults| * |rates| * |gaps|), it spends those sims NON-uniformly to beat brute-force's
rank at equal budget. It does NOT use a bandit; it allocates by a simple rule:

  pour each batch into the WIDEST-Wilson-margin gap of every still-CONTENDED (fault, rate) pair.

That is the one idea in the name: "widest Wilson margin gap". The ranking is identical to
brute-force (Jeffreys log-likelihood), so the ONLY thing that differs from brute is the ALLOCATION.

ALGORITHM STRUCTURE
-------------------
  0. INIT  -- seed every (fault, rate, gap) with a small uniform batch (init 30), so all pairs score
              L over the same gaps (comparable), and each estimate has a first Wilson interval.
  1. LOOP (rounds) -- each round, for EVERY still-contended pair (f,r), add a batch to that pair's
              WIDEST-Wilson-CI gap (log-space) -- the estimate that most limits the ranking. Between
              rounds, FREEZE/DECIDE (same neighbour-separation rule as the UCB variants):
                * freeze a rate whose L-interval sits below another live rate's low end, and
                * decide (fix) a fault once it is separated from both its ranking neighbours.
              Stop at the budget, or early if the full ranking is settled (then it lands at FEWER
              sims than brute -- an even stronger result).
  2. RANK  -- each fault by its best rate's Jeffreys log-likelihood over all gaps, sorted.

  Difference vs the UCB variants: WWMG spreads budget UNIFORMLY over all contended pairs each round;
  ucb_fr uses a bandit to FOCUS budget on the promising pairs (same within-pair widest-CI gap rule).

ESTIMATORS
----------
  RANKING  : Jeffreys point p = (hits+0.5)/(n+1)   (matches brute-force; no 1e-12 floor).
  PRUNING  : 95% Wilson score interval (boundary-safe), used for freeze/decide AND the widest-CI gap.

PARAMETERS
----------
  TUNABLE (study variable): MG_WWMG_N  -- budget N, SWEPT {50,100,200,400,800}.  (No C -- not a bandit.)
  HARDCODED (fixed constants, NOT env-tunable; the proven config):
     init_batch = 30, per-round batch_size = 20, CI = tight (quadrature), Jeffreys ranking = on.

WHAT IS RECORDED (xlsx columns of interest)
-------------------------------------------
  real_fault_rank, total_simulations, wwmg_num_rounds, wwmg_frozen_rate_pairs, wwmg_decided_faults,
  wwmg_budget_N, wwmg_stop_reason ("budget" | "settled" | "no_live_pairs").

RESULTS LAYOUT (filled by the sweep)
------------------------------------
  wwmg/<domain>/N<N>/xlsx/...     batched (~10 instances per file)
    N in {50,100,200,400,800};  domains: FrozenLake_v1, Taxi_v4,
    MiniGrid_Empty_16x16_v0, MiniGrid_SimpleCrossing_S11N2_v0
===============================================================================
