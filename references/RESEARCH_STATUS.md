# Research status & forward plan (context from 24.5.2026 meeting + Ahmad's background)

Living doc. Findings below are Ahmad's account; to be confirmed/updated once the post-24.5
result files and profiling outputs are scanned.

## Findings so far

### Epsilon / adaptive Monte-Carlo (estimating gap probabilities)
A "gap" = hidden span in a trajectory `s1,a1,…,(hidden×k),…,sn,an`. We estimate
`P(gap | candidate fault)` by simulation.
- Moved from a **constant number of simulations** → **confidence-interval adaptive sampling**:
  get `p_hat` + an interval, and do a **custom #sims per gap** (`simulate_m_traces_adaptive_monte_carlo`).
- ✅ Confirmed: **larger ε → fewer iterations + less time** (one large single experiment).
- ❗ Surprise: **smaller ε did NOT improve "average real-fault rank."** All experiments were on
  **FrozenLake**, so the inputs may be too easy / too hard / too similar — unknown. **A key reason we
  want more domains.**
- TODO: scan real results + plots to characterize this.

### Profiling
- ❗ Counterintuitive: **average runtime increases as visibility rate increases** (expected the
  opposite — more visibility → smaller gaps → less time).
- cProfile showed **per-single-simulation overhead is large**; plotting is fine → **number of
  simulations is the dominant cost**. Future: optimize #sims / per-sim overhead.
- Mentor asked to **visualize cProfile output** with a tool (later).
- Profiling outputs to be attached + scanned.

## Forward plan

### A. (Likely first) Make the code testable
- Unsure the current code is testable. Investigate, then add a test harness / smoke tests so we can
  validate after every big change. See [[code-testability-first]].

### B. Unknown fault rate — more experiments
- Dynamic unknown-fault-rate method is **implemented** (`fault_identification_non_deterministic_PO_unknown_fault_rate`).
- Need **many more experiments comparing vs known fault rate**, measuring **time** and
  **average fault rank**. Hypothesis: unknown rate → **more time + worse avg rank**.

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
- A batch of **post-24.5 result files** to scan → then update "where we are now" here.
- **Profiling outputs**.
- **Gymnasium doc links** (FrozenLake + Taxi-v4).
