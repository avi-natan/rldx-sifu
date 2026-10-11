# Total-budget brute-force experiments

Sibling of `unknown_fr_bruteforce_improvement/`, but the brute baseline is driven by a **fixed
TOTAL simulation budget B per instance** instead of a fixed number of sims per estimate (N).
Same structure: `<variant>/<domain>/B<budget>/{xlsx,logs}`.

Metric unchanged: mean `real_fault_rank` over the instances in a cell (lower = better), 10 candidates.
Everything else matches the existing brute runs exactly so results are comparable:
epsilon 0.04, unknown fault rate, Jeffreys floor on, max-over-rate scoring, same instances /
candidate sets, same injected fault rates (**FrozenLake 0.5, Taxi 0.3, MiniGrid both 0.5 @ noise 0.5**),
same CRN seeding.

Budget sweep (anchored to the AVG total the old N-sweep consumed across the 4 domains):
**50k, 100k, 200k, 400k, 800k, 1.6M, 3.2M** (200k≈N50, 400k≈N100, …, 3.2M≈N800).

---

## Method 1 — `brute_even` (built; this run)

`fault_identification_non_deterministic_PO_unknown_fault_rate_BRUTE_EVEN` (dedicated, hardcoded).
Reads only the swept budget `B` from env `MG_BRUTE_B`.

**Diff vs the old brute** (shared full diagnoser, `MG_FIXED_TRIES=N`):

| | old brute (N per estimate) | brute_even (total budget B) |
|---|---|---|
| control knob | N = sims **per** (fault,rate,gap) estimate | B = **total** sims for the whole instance |
| per-estimate sims | exactly N, for every estimate | `n = round(B / (F·R·G))`, same for every estimate |
| total sims / instance | `N · F · R · G` → **scales with #gaps G** (trajectory length / visibility) | `≈ B` → **independent of G** |
| even split across gaps | yes (each estimate = N) | yes (each estimate = n) |
| scoring / seeding / Jeffreys | — unchanged — | — unchanged — |

F = #candidate faults (10), R = #candidate rates (10), G = #gaps in that trajectory. Every estimate
runs exactly `n` sims (min=max=n, epsilon inert). So two instances with very different gap-counts
now get the *same* total budget B — the whole point (mentor note #3).

Recorded per row: `budget_total_requested` (B), `budget_split` ("even"), `budget_per_estimate_n` (n),
`budget_num_estimates` (F·R·G).

## Method 2 — `brute_length` (built; ready to launch, not yet launched)

`fault_identification_non_deterministic_PO_unknown_fault_rate_BRUTE_LENGTH` (dedicated, hardcoded).
Same total budget B (env `MG_BRUTE_B`), but split across gaps **proportional to gap length** `ℓ`:
`n_g = max(1, round(B · ℓ_g / (F·R·L)))`, `L = Σℓ_g`, so longer (harder, smaller-p) gaps get more
sims; total ≈ B. Recorded `budget_split="length"`, `budget_per_estimate_n` = realized mean n.
Rationale: the fault score is `Σ_gaps log p`, and long gaps have small, noisy p that can dominate
that sum — length-weighting pours sims where they matter most. Variant name `brute_length`.

**Exactness note (applies to both brute_even and brute_length):** the per-estimate sim count is run
as a SINGLE Monte-Carlo batch of exactly `n` (epsilon inert), so the total lands on B. The earlier
batch size of `min(50,n)` overshot `max_tries` to the next multiple of 50 whenever `n` was not a
multiple of 50 (the old N-sweep never hit this because N∈{50,100,200,400,800}); fixed `2026-10-12`.
