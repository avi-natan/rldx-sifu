# Mentor feedback — 2026-10 meeting (2-day work plan)

Source: Ahmad's meeting with mentor (Roni) after showing the unknown-fault-rate
brute-vs-adaptive tables/plots (`unknown_fr_bruteforce_improvement/`).
Current best adaptive method = **`ucb_fr`** (arm = (fault, rate); budget spent on the
widest-Wilson-CI gap). Beats brute in ~15/20 cells, big low-N wins.

Metric throughout: mean `real_fault_rank` (position of the true fault in the ranked
list, 1 = best), averaged over the instances in a cell (~390–500). 10 candidate faults.

---

## 1 + 2. Uncertainty / significance  (THE core ask — same underlying issue)

Motivation (Ahmad's own observation in the meeting): in several cells brute beats us by
a *tiny* margin. Is that real or just noise? Where is the line between a "big" and a
"small" difference? Notes 1 and 2 are the two answers to that one question.

**Note 1 — shaded std err on plots (maybe tables too).**
Each plotted mean has uncertainty: `SE = std(per-instance ranks) / sqrt(n_instances)`.
Shade a band of ±1 SE (or ±1.96 SE ≈ 95%) around each line. Reading: close lines whose
bands overlap heavily => difference is noise; bands clearly separated => difference is real.

**Note 2 — mark statistical significance in tables; which test.**
Comparison is **PAIRED** — CRN means brute and ucb_fr run on the SAME instances with the
SAME seeds. So test the per-instance difference `d_i = rank_brute(i) - rank_ours(i)` for
mean != 0, NOT two independent column means.
- Test: **Wilcoxon signed-rank** on d_i (paired, non-parametric; ranks are bounded ints
  1..10, not normal), or a **paired bootstrap** on mean(d). Paired t-test = crude cousin.
- Mark cells: `*` p<0.05, `**` p<0.01, `***` p<0.001; unmarked = too close to call.
- Pairing cancels instance-to-instance variance → far more powerful than comparing means.
- **PREREQUISITE:** confirm brute per-instance rows align to ucb_fr rows by a stable key
  (map / fault / seed) in the xlsx. If rows pair up, both notes are straightforward.

## 3. Total budget B (not N × arms)

Today budget = N sims **per arm**, so total = N × faults × rates × gaps(G) — depends on
trajectory length (G varies per instance). Mentor wants a **fixed total budget B**,
independent of G. Plan: brute splits B evenly over all arms; **sweep B**; pick a few B
values to compare against the adaptive methods (which already think in a total budget).
Mostly re-parametrizes brute; keep the equal-budget framing for the adaptive methods.

## 4. Candidate-fault selection for Taxi & MiniGrids — maybe make it easier

Currently adversarial: per-instance static hardest-10 (MiniGrid), class-2 hard (Taxi).
Review each domain's candidate picker; consider a less brutal / more separable selection
so the problem is easier. (Identity fault always included.)

## 5. Plot #frozen as a function of budget N

Already logged: `ucbfrg_rate_freezes` / `ucbfrg_fault_freezes`,
`wwmg_frozen_rate_pairs` / `wwmg_decided_faults`. Plot mean frozen vs N.

## 6. Fault rating: max-over-rate vs sum

Today: fault score L = **max over rate** of Σ_gaps log p  (profile likelihood, "best-case
rate"). Roni suggests **sum** across rates = marginalize the rate (sum / log-sum-exp,
Bayesian average). NB: we already SUM over gaps within a rate — the max-vs-sum question is
only about combining ACROSS rates. One-line scorer swap; A/B both. Affects every method
(shared scorer).

## 7. MAB variant for a 3-element arm where we choose 1 (literature task)

Maps to best-arm identification with a **factored / structured arm (fault, rate, gap)**
where only ONE factor (the fault) is the answer and the others are nuisance dimensions —
exactly the ucb_frg (arm=f,r,g) vs ucb_fr (arm=f,r) tension. Research the formal name:
structured / combinatorial pure-exploration, nuisance-parameter bandits, best-arm-ID over
groups. Bring back what fits and whether it suggests a better allocation rule.

---

### Suggested order of attack (2 days)
Quick wins first: 5 (freeze plot, data already there) and 1 (SE shading). Then 2 (paired
significance — needs row alignment, highest value). Then 6 (scorer A/B — cheap, affects
all). Then 3 (total-budget brute sweep — new runs). 4 and 7 in parallel (4 = code review of
candidate pickers; 7 = literature). Re-run affected experiments as needed.
