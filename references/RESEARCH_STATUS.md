# Research status & forward plan (context from 24.5.2026 meeting + Ahmad's background)

Living doc. The findings below are now **grounded against the post-24.5 result batch**
(`~/Downloads/experiments_outputs/`, 53 result `.xlsx`, all `FrozenLake_v1` PO). Result-file
column schema is documented in memory [[excel-results-schema]].

**Setup note:** injected **fault rate is a run parameter (0.5 and 0.8)**, NOT a data column — do
not confuse it with the unknown-rate method's *estimated* per-fault rate (`Best Rate:` /
`*_estimated_rate`, searched over candidates `[0.1,0.3,0.5,0.7,0.8]`). The axes that vary inside the
data are **`percent_visible_states` ∈ {20,40,60,80,100}** (within a file) and **`epsilon`** (across files).

## Findings (grounded in the post-24.5 batch)

### Epsilon / adaptive Monte-Carlo (estimating gap probabilities)
A "gap" = hidden span in a trajectory `s1,a1,…,(hidden×k),…,sn,an`. We estimate
`P(gap | candidate fault)` by simulation.
- Moved from a **constant number of simulations** → **confidence-interval adaptive sampling**:
  get `p_hat` + an interval, and do a **custom #sims per gap** (`simulate_m_traces_adaptive_monte_carlo`).
- ✅ Confirmed: **larger ε → fewer sims → less time.** `try2` (known rate, 49 maps): mean
  `diagnosis_time_sec` drops **174s (ε=0.015) → ~7s (ε≥0.1)**; `adaptive_avg_real_tries`
  **2773 → ~156**.
- ❗ Confirmed surprise: **smaller ε did NOT improve average real-fault rank** — mean rank stays
  **flat ~1.6–1.8 across all ε** in `try2`. Smaller ε buys ~18× more sims and ~25× more time for
  **no rank gain**. Still all on FrozenLake → a key reason to add more domains.
- ⚠️ **try2 vs try3 floor discrepancy (open question):** the two sweeps are *byte-identical for
  ε ≤ 0.04*, then diverge — `try3`'s higher-ε runs use a **much lower simulation floor (~20 sims vs
  ~156)**, and there mean rank **degrades to ~2.3–2.4** (top1 ~0.65→~0.50). So rank is insensitive
  to #sims **above ~156**, but **collapses near ~20 sims**. Sweet spot ≈ **ε 0.1** (near-best rank at
  ~7s vs 174s at ε 0.015). **Need to confirm which min-tries floor is the intended/"production" one.**

### Profiling — confirmed in the data
- ❗ Counterintuitive but **confirmed**: within `epsilon_0_03`, mean `diagnosis_time_sec` *rises*
  with visibility — **20%→100% : 12.3s → 53.6s** — even though `largest_gap` shrinks (18.5 → 0).
  Cause: **`num_gaps` explodes (22 → 108)**; cost tracks *number of gaps × sims*, not gap size.
  Accuracy improves meanwhile (mean rank 2.24 → 1.40; top1 0.61 → 0.72).
- cProfile showed **per-single-simulation overhead is large**; plotting is fine → **number of
  simulations is the dominant cost**. Future: optimize #sims / per-sim overhead.
- Mentor asked to **visualize cProfile output** with a tool (later). cProfile `.docx`/`.txt`/`.out`
  in `profle_results/` not yet deep-read.

## Forward plan

### A. (Likely first) Make the code testable
- Unsure the current code is testable. Investigate, then add a test harness / smoke tests so we can
  validate after every big change. See [[code-testability-first]].

### B. Unknown fault rate — more experiments
- Dynamic unknown-fault-rate method is **implemented** (`fault_identification_non_deterministic_PO_unknown_fault_rate`).
- **First grounded comparison (ε 0.03, `UNknow_vs_known/`, n=290 each):**

  | | mean_rank | top1 | top3 | time_sec |
  |---|---|---|---|---|
  | KNOWN | 1.707 | 0.645 | 0.921 | 34.6 |
  | UNKNOWN | **1.628** | **0.700** | 0.917 | **148.6** |

  → Hypothesis **partially holds**: unknown is **~4.3× slower** (more time ✅) but **not worse on
  rank** — marginally *better* here (worse-rank hypothesis ✗). Estimates mean rate ~0.60 vs true
  0.50 (slight over-estimate; candidate spread also picks up 0.7/0.8). Good story for the method.
- Still need **many more experiments** (more maps/seeds, both fault rates 0.5 & 0.8) to confirm.

### C. New domains — Taxi-v4 (DELAYED; plan kept here for later)
1. **Check the Taxi-v4 implementation** for correctness.
2. **Write a single simple test** for it.
3. **Test the newly-trained policy**; if not good, train a better one.
   - Open question: *is a very-good policy even desirable?* (a near-optimal policy may rarely
     deviate, giving fewer diagnosis signals.)
4. If all good: **discussion with Claude to understand the domain** better.
5. **Add a `multiple_experiment` method for Taxi** (mirroring the FrozenLake one).
6. Reference: Ahmad to give **gymnasium doc links** for FrozenLake + Taxi-v4 (Claude can WebFetch).
7. Ties in the known bug: the freshly-trained rainy Taxi model isn't wired into the loader (see `CODE_SCAN.md`).
- Next domain after Taxi-v4: **CliffWalking**.

### D. Domains catalog
- Create a **list of all gym domains + properties** (discrete/continuous, static/dynamic, stochastic,
  #actions, avg trajectory length) — like Table 1 in the SIF/SIFU paper.

## Pending inputs from Ahmad
- ✅ **Post-24.5 result batch scanned** (53 xlsx) — findings above are now grounded.
- **Profiling outputs**: the cProfile `.docx`/`.txt`/`.out` in `profle_results/` still need a deep read
  (per-function breakdown behind the visibility paradox).
- **Gymnasium doc links** (FrozenLake + Taxi-v4).
- The `pong model/` outputs in the batch look **EOM/old-project-related, not rldx** — treated as
  out of scope unless told otherwise.
