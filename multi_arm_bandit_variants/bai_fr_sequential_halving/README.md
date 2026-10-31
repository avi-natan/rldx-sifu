# bai_fr — Sequential-Halving best-arm identification (MAB variant #1)

A total-budget diagnoser that treats fault diagnosis as **pure exploration / best-arm
identification** instead of regret minimisation. Motivation + literature mapping:
Combinatorial Pure Exploration (Chen & Lin, NeurIPS 2014) and the successive-rejects /
sequential-halving family (Audibert-Bubeck-Munos; Karnin-Koren-Somekh).

## Method
- **Arm roles:** FAULT = decision target (argmax); RATE = nuisance, profiled out by max-over-rate;
  GAP = conditionally-independent evidence, combined by geomean of Jeffreys p-hat (same scoring as
  brute / ucb_fr_b, so ranks are comparable).
- **Scheduler:** Sequential Halving over the F faults — `ceil(log2 F)=4` phases; each phase spends an
  equal slice of B uniformly over surviving faults (and their rate×gap estimates), then drops the
  worst half, reallocating freed budget to survivors. **No init / warm-up.**
- Code: `p_diagnosers.py::fault_identification_non_deterministic_PO_unknown_fault_rate_BAI_FR`,
  variant name `bai_fr`, budget env `MG_BAI_B`.

## Cluster run (2026-10-27)
7 budgets (50k–3.2M) × 4 domains, HARD benchmark, eps 0.04, -ufr; same batched config as
`ucb_fr_b` / `brute_even` (job IDs 23627013–23627040). Results in `results/`, comparison in
`comparison_plots/`. See `results/PULL_MANIFEST.txt`.

## RESULT — negative (bai_fr is NOT competitive as-is)
Best-so-far in **2 / 28 cells, worse in 26** (the 2 "wins" are within ±1 SEM = noise).
- **Low budget:** badly worse (FL 50k: bai 3.51 vs ucb 2.49; MG-Empty 100k: 3.45 vs 2.84).
- **High budget:** converges up to the pack, never leads.

**Cause:** Sequential Halving optimises TOP-1 by **permanently eliminating** faults, but the metric
is the **rank of the true fault** in the full ordering. A fault halved out in phase 1 (on near-noise
low-budget estimates) is forced below every survivor and cannot recover → rank ≥ 6. Hard elimination
is the wrong mechanism for a rank objective; the non-eliminating methods (ucb_fr_b, brute) keep every
fault rankable by accumulated likelihood and win.

**Fix direction (if revisited):** a ranking-aware, **non-eliminating** allocation (LUCB / UGapE /
Track-and-Stop style) that steers budget toward the contested top while keeping every fault sampled
enough to stay rankable — closer to what ucb_fr_b already does. Or bank this as a documented negative
result and try a different MAB variant in this folder.
