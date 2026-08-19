# Epsilon / #-simulations experiment — persistent handoff

**Written:** 2026-06-21. **Author:** Claude Code session (Opus 4.8), with Ahmad.
**Why this file exists:** Ahmad is stepping away for ~2–3 weeks. This is the complete,
self-contained record of the epsilon experiment: how inputs are built, what was run, the
results, and the exact path + meaning of every data/log file we will need when we come back.

**Branch:** `new-master-ai`. **Everything below is currently UNCOMMITTED** (Ahmad said "we
will commit, wait for now"). See the "Uncommitted state" section at the bottom before any
`git` cleanup.

> **⚠️ Repository reorg (2026-08-19).** All experiment scripts named below now live under
> **`experiments_scripts/`** — prefix every `python <script>.py` command accordingly
> (e.g. `python experiments_scripts/hard_taxi_benchmark_v2.py`). Run them **from the repo
> root** (each has a `sys.path` bootstrap so its `p_*`/`h_*` imports resolve).
> These one-off scripts were **removed** in the cleanup (their outputs remain under `runs/`):
> `run_class2_lowfr.py`, `run_class2_more.py`, `finish_phaseB.py`, `analyze_sweep.py`,
> `deepdive.py`, and the v1 runner `run_hard_taxi_benchmark.py`. The served rainy-0.7 policy
> and the good-policy library are now committed under `environments/Taxi_v4/models/PPO/`.

---

## 0. One-paragraph summary

We measured how the adaptive Monte-Carlo confidence threshold **epsilon (ε)** trades off
against **diagnosis accuracy** on the hardest Taxi-v4 instances (near-twin fault candidates,
very low fault rates). Smaller ε = tighter MC stopping = more simulated traces = (hypothesis)
better fault ranking. **Result: confirmed, monotone, and significant** at n=549 paired cells
per ε — avg true-fault rank improves **3.55 → 3.13** and detection@1 **22% → 29%** as ε goes
**0.1 → 0.02**, with the effect strongest at the lowest fault rate (fr=0.0025) and compute
exploding (33k → 369k traces). Practical sweet spot **ε ≈ 0.04–0.03**.

---

## 1. How inputs are prepared (files + methods)

> **Two generators exist.** The **first experiments** used **v1**
> (`hard_taxi_benchmark.py`, frozen to `hard_taxi_data.py`) which corrupts the **most-used**
> actions — documented in §1.4. The **ε experiment** (this handoff's focus) uses **v2**
> (`hard_taxi_benchmark_v2.py`) which corrupts the **least-used** action (class 2) — §1.1–1.3.

### 1.1 The benchmark generator — `hard_taxi_benchmark_v2.py`
Builds the instance pool. Key entry point:

```python
build_benchmark(seeds_per_class=50, seed_start=1, max_steps=200,
                verbose=True, max_scan=8000, classes=(2,3,4))
```
Returns a list of tuples, one per instance:
```
(main_seed, actual_seed, class_id, chosen_action a*, E_str, [10 candidate map strings])
```
- `actual_seed = main_seed * SEED_BLOCK` (= the instance's per-block base seed; `SEED_BLOCK=1_000_000`).
- **Policy:** the hardcoded **0.7-rainy** Taxi table policy (`load_trained_model("Taxi_v4","PPO")`
  → `TaxiHardcodedPolicy`, tabulated from the promoted `Taxi_v4__PPO.zip`).
- **Profiling** (`profile_seed`): rolls the **healthy** policy from `reset(seed=base)` and counts
  how often each of the 6 actions is commanded. Seeds where the policy doesn't solve are skipped.
- **Difficulty classes** (by how often the corrupted action `a*` is commanded → how often the
  fault fires → hardness), via `class_picks(counts, seed)`:
  - **class 2** = least-used action *with count ≥ 2* (rarest real movement) → **hardest**. *This is the class the whole ε study uses.*
  - **class 3** = most-used action → easiest.
  - **class 4** = random from the remainder (excludes classes 1/2/3 picks).
  - **class 1** (least-used overall = almost always Pickup/Dropoff) was **dropped**: corrupting
    it re-converges → impossible, not hard. Class ids kept as 2/3/4 so each id keeps its meaning.
- **Candidate set** (`build_candidates(a_star, b, seed)`), 10 maps, true fault `E` first:
  - `E` corrupts `a*` → a random *other* action.
  - + the 5 other maps of `a*` (a*→each other action) = all 6 maps of `a*`.
  - + 4 "tier-2 twins": keep E's a*, instead corrupt the **2nd-rarest** action `b`
    (`select_b`) to 4 distinct targets. These are the near-twins that make ranking hard.
- `classes=(2,)` argument lets a worker build **class 2 only** (used by the runners below).

### 1.2 The diagnosis call — `p_pipeline.py :: run_NON_DETERMINSTIC_single_experiment_PO`
Every cell calls this with (see runner args in §2). It:
1. Prepares inputs (`single_experiment_prepare_inputs_non_determinstic`), which includes the
   **firing-gate sweep**: execute the trajectory with `fault_seed_offset` starting at
   `FAULT_OFFSET` and incrementing until the fault actually fires (≥1 faulty action) and the
   trajectory is long enough; if it reaches `SIMULATION_OFFSET` without firing, the **seed is
   dropped** (returns empty → cell recorded as `dropped`). This is a **runtime** drop, not a
   generation-time filter.
2. Runs the stochastic PO diagnoser (`fault_identification_non_deterministic_PO` in
   `p_diagnosers.py`) — adaptive Monte-Carlo + logL ranking of the 10 candidates.
3. Returns a dict; the keys we record: `real_fault_rank`, `adaptive_total_real_tries`
   (total MC traces simulated), `adaptive_total_calls`, `adaptive_avg_p_hat`, `sorted_faults`.

### 1.3 Seeding (so the runs are reproducible & decorrelated) — `h_consts.py`
```
SEED_BLOCK = 1_000_000   # each instance owns a block of 1e6 seeds: base = main_seed*SEED_BLOCK
WINDOW_SIZE = 1000
TRAJECTORY_OFFSET = 0     # env reset / trajectory stream
CANDIDATE_OFFSET  = 1     # candidate-shuffle stream
MASK_OFFSET       = 2     # visibility-mask stream
FAULT_OFFSET      = 3     # fault-firing stream; firing sweep runs [3, SIMULATION_OFFSET)
SIMULATION_OFFSET = 5000  # MC simulation stream base (raised 1000→5000 for firing-retry room)
MAX_STATES = 200          # MC per-gap seed stride (gap decorrelation: residue class mod 200)
```
MC seed per gap/trace = `base + SIMULATION_OFFSET + last_observed_index + t*MAX_STATES`
(candidates share it → Common Random Numbers; gaps decorrelated). Inlined in
`simulate_m_traces_adaptive_monte_carlo` (`p_diagnosers.py`).

### 1.4 The FIRST experiments' input prep — `hard_taxi_benchmark.py` (v1, most-used action)
The earlier benchmark. Built once and **frozen** to `hard_taxi_data.py` (100 instances), so
those experiments read static data instead of rebuilding. Spec: `references/HARD_TAXI_SPEC.md`.
Action ids: 0=DOWN 1=UP 2=RIGHT 3=LEFT 4=Pickup 5=Dropoff; healthy map `[0,1,2,3,4,5]`.

Five steps (run `python experiments_scripts/hard_taxi_benchmark.py 2` then `5` to regenerate):
1. **Execution-fault pool — 45 faults** (`execution_fault_pool`):
   - `single_redirects()` — one action misfires a→b, rest identity = 6×5 = **30** (Hamming-1).
   - `single_swaps()` — one crossed pair a↔b = C(6,2) = **15** (Hamming-2, bijective).
2. **Per-seed action profile** (`profile_seed` / `build_counts_table`): scan seeds up from
   `SEED_START=42` until `TARGET_SOLVED=105` solve; one healthy rollout per seed counts how
   often each action is **commanded**; stuck seeds skipped; frozen as `COUNTS_TABLE`.
3. **Per-seed execution fault E** (`select_execution_fault`) — *the "most-used action" choice:*
   take the **top-3 most-used** actions (`top_used_actions(k=3)`), keep those commanded
   `EXEC_MIN_COUNT=2`+ times (so the fault fires enough to be well-posed), restrict the 45-pool
   to faults whose corrupted action is in that set, then pick **one uniformly at random**,
   `random.Random(seed)`. → E corrupts a **heavily-commanded** action.
4. **Graded 10-candidate set** (`build_candidate_set`), scored by frequency-weighted
   disagreement (`candidate_score`, low = hard): **7 near-twins** that agree with E on the
   faulty action f but differ on ONE other used action — **4 hard** (cheapest used action) /
   **2 medium** (middle) / **1 easy** (priciest) — plus **2 alternatives** (healthy except a
   heavily-used action corrupted, disagree on f → easy rejection). Seeds with <2 other used
   actions are dropped as too sparse.
5. **Assemble + freeze** (`build_benchmark`): 100 tuples `(seed, E_str, [10 candidates; E first])`
   → `hard_taxi_data.py` as `BENCHMARK` + `EXECUTION_FAULT_POOL` + `DISTRACTORS` +
   `COUNTS_TABLE` + `ROLLOUT_META`.

**v1 → v2 contrast** (why we changed it):
| | v1 (first experiments) | v2 (ε study) |
|--|--|--|
| seed → fault on | **top-3 MOST-used** action | class 2 = **LEAST-used (count ≥ 2)** |
| E pool/target | random from 45 single-cause faults hitting a top action | a* → a random other action |
| 10 candidates | E + 7 freq-graded near-twins + 2 alternatives | E + 5 other a*-maps + 4 tier-2 twins on `b` |
| difficulty | **graded** (hard/med/easy inside one instance) | **uniformly hard** near-twins |
| storage | **frozen** → `hard_taxi_data.py` (100 instances) | rebuilt each run from seeds |
| seeding | RNG `random.Random(seed)` | per-instance seed blocks (§1.3) |

v1 was for measuring diagnoser accuracy on *well-posed, graded* instances; v2 deliberately moves
to the *least-fired, near-twin* regime where the ε / #-simulations tradeoff is observable.

---

## 2. What we ran (params & arguments)

**Two experiments.** The headline ε result is the **second** (class-2 low-fault-rate study).

### 2.1 Multi-class sweep (earlier, vis{50,80}) — context only
- Driver scripts: `run_epsilon_sweep.py` (Phase A 10/class + Phase B +10/class),
  `finish_phaseB.py` (finished class 4 to 20/class), `analyze_sweep.py`.
- Classes **2,3,4**, **20 seeds/class**, **vis {50, 80}**, **fault rates {0.3,0.2,0.1,0.05,0.02}**,
  **ε {0.035, 0.07, 0.1}**. 1800 cells total.
- Purpose: confirm class 2 is hardest and that ε matters mainly at low fault rate. It did, but
  at 20/class the ε effect was within noise — motivating experiment 2.2.

### 2.2 Class-2 low-fault-rate ε study (THE headline experiment)
Fixed config for **every** cell:
- **Class:** 2 only (hardest near-twins).
- **Visibility:** `percent_visible_states = 100` (fully observed).
- **Fault rates:** `{0.02, 0.01, 0.005, 0.0025}` (very low — the sim-hungry regime).
- **Epsilons:** `{0.1, 0.07, 0.05, 0.04, 0.03, 0.02}` (run as 6 parallel processes, one per ε).
- **Candidates:** 10 (`num_candidate_fault_modes=10`, `fixed_candidate_fault_modes=cands`).
- **`max_exec_len=200`, `unknown_fault_rate=False`, `ml_model_name="PPO"`, `domain="Taxi_v4"`.**
- **Seeds:** 140 class-2 instances total =
  - first **40** seeds → produced by `run_class2_lowfr.py <eps>` → `cells_eps<eps>.csv`
  - next **100** seeds (indices 40–139) → produced by `run_class2_more.py <eps> 40 100`
    → `cells2_eps<eps>.csv`
- Per ε: 140 seeds × 4 fault rates = 560 cells; **549 valid** after 11 firing-drops (the 11
  dropped cells are identical across ε because firing depends only on seed/fault-rate, not ε).

Run commands (each line a separate process; 6 ran concurrently):
```
python run_class2_lowfr.py 0.1 ;  python run_class2_lowfr.py 0.07 ; ... (first 40 seeds)
python run_class2_more.py 0.1 40 100 ; python run_class2_more.py 0.07 40 100 ; ... (next 100)
```

---

## 3. Results

### 3.1 Combined 140-seed overall (per ε; n=549 valid each, 11 dropped each)
| ε    | avg true-fault rank | detect@1     | avg MC traces |
|------|---------------------|--------------|---------------|
| 0.10 | 3.55                | 123/549 (22%)| 32,885        |
| 0.07 | 3.57                | 124/549 (23%)| 39,219        |
| 0.05 | 3.44                | 131/549 (24%)| 65,319        |
| 0.04 | 3.33                | 145/549 (26%)| 98,165        |
| 0.03 | 3.21                | 154/549 (28%)| 168,451       |
| 0.02 | **3.13**            | 158/549 (29%)| 369,406       |

(SEs on rank ≈ 0.09–0.10; rank is over 10 candidates, lower=better. Because cells are
**paired** across ε — same seeds — the 0.1→0.02 improvement of 0.42 is well beyond noise.)

**Read:** monotone improvement as ε shrinks (the 0.1↔0.07 pair is a flat "dead zone" — both too
coarse). Diminishing returns past ε≈0.04–0.03 while compute explodes (~11× from 0.1 to 0.02).

### 3.2 avg rank by (fault_rate × ε) — effect grows as fault rate drops
```
fr        0.10   0.07   0.05   0.04   0.03   0.02     improvement
0.02      2.37   2.39   2.36   2.26   2.21   2.26     ~ -0.11  (least hard)
0.01      3.31   3.30   3.11   3.01   3.03   2.97     ~ -0.34
0.005     3.97   3.99   3.84   3.66   3.53   3.38     ~ -0.59
0.0025    4.56   4.60   4.45   4.39   4.09   3.91     ~ -0.65  (hardest; keeps falling to 0.02)
```

### 3.3 detect@1 % by (fault_rate × ε)
```
fr        0.10   0.07   0.05   0.04   0.03   0.02
0.02      34%    34%    36%    39%    40%    41%
0.01      26%    27%    28%    30%    32%    31%
0.005     20%    20%    19%    23%    23%    27%
0.0025     9%     9%    12%    13%    17%    16%
```
At the hardest fault rate (0.0025) detection roughly **doubles** (9% → ~16–17%) as ε shrinks.

### 3.4 Worked examples (smaller ε detects, larger ε misses) — `deepdive.py`
Class-2 seeds **6** (fr 0.02) and **7** (fr 0.05): at ε=0.035 the true fault is **rank 1**
(~65k–92k traces); at ε=0.07/0.1 it falls to **rank 3** (~16–28k traces). The logL gap to the
winner is tiny (~0.003–0.014) — i.e. difficulty is **near-twin similarity** (p_E/p_C → 1), and
only more simulations resolve it. (Note: `deepdive.py` rebuilds with `seeds_per_class=20` and
uses the older ε triple {0.035,0.07,0.1}; rerun it to reproduce those two cases.)

### 3.5 Bottom line for the thesis
On class-2 near-twins at very low fault rates, **more Monte-Carlo simulations (smaller ε) buy
real, monotone diagnostic accuracy**, strongest where firing is rarest (fr ≤ 0.005), with
**sharply diminishing returns past ε ≈ 0.03**. Quantified statement of the ε / #-sims tradeoff,
backed by 140 paired seeds.

---

## 4. Exact data/log files (paths + what each contains)

All under repo root `C:\Users\ahmad\Desktop\rldx_repo\rldx-sifu\`.

### 4.1 THE headline data — class-2 low-fault-rate study (`runs/class2_lowfr/`)
Each CSV: columns `seed, a_star, eps, vis, fr, rank, dropped, total_tries`
(`rank` blank + `dropped=1` when the fault never fired; `total_tries` = MC traces simulated).
| File | Seeds | ε | Made by |
|------|-------|----|---------|
| `runs/class2_lowfr/cells_eps0.1.csv`  | first 40  | 0.10 | `run_class2_lowfr.py 0.1` |
| `runs/class2_lowfr/cells_eps0.07.csv` | first 40  | 0.07 | `run_class2_lowfr.py 0.07` |
| `runs/class2_lowfr/cells_eps0.05.csv` | first 40  | 0.05 | `run_class2_lowfr.py 0.05` |
| `runs/class2_lowfr/cells_eps0.04.csv` | first 40  | 0.04 | `run_class2_lowfr.py 0.04` |
| `runs/class2_lowfr/cells_eps0.03.csv` | first 40  | 0.03 | `run_class2_lowfr.py 0.03` |
| `runs/class2_lowfr/cells_eps0.02.csv` | first 40  | 0.02 | `run_class2_lowfr.py 0.02` |
| `runs/class2_lowfr/cells2_eps0.1.csv`  | next 100 | 0.10 | `run_class2_more.py 0.1 40 100` |
| `runs/class2_lowfr/cells2_eps0.07.csv` | next 100 | 0.07 | `run_class2_more.py 0.07 40 100` |
| `runs/class2_lowfr/cells2_eps0.05.csv` | next 100 | 0.05 | `run_class2_more.py 0.05 40 100` |
| `runs/class2_lowfr/cells2_eps0.04.csv` | next 100 | 0.04 | `run_class2_more.py 0.04 40 100` |
| `runs/class2_lowfr/cells2_eps0.03.csv` | next 100 | 0.03 | `run_class2_more.py 0.03 40 100` |
| `runs/class2_lowfr/cells2_eps0.02.csv` | next 100 | 0.02 | `run_class2_more.py 0.02 40 100` |
- `log_<eps>.log` / `log2_<eps>.log` — one-line "DONE" markers per process (no analysis).
- **To reproduce §3 tables:** combine `cells_eps<e>.csv` + `cells2_eps<e>.csv` for each ε,
  keep rows with `dropped=0` and non-blank `rank`, then group by `fr`. (The exact PowerShell
  used is in the session transcript; trivial to redo with pandas.)

### 4.2 Earlier multi-class sweep (`runs/epsilon_sweep/`) — context, vis{50,80}
- `runs/epsilon_sweep/cells.csv` — 1800 rows, columns `class, seed, a_star, eps, vis, fr, rank, dropped`.
  Classes 2/3/4 × 20 seeds × vis{50,80} × fr{0.3,0.2,0.1,0.05,0.02} × ε{0.035,0.07,0.1}.
- `runs/epsilon_sweep/report.txt` — timestamped progress + the per-(class,ε) avg-rank tables.
  Key line (20/class, low fr only): class 2 ε0.035=3.27 / ε0.07=3.44 / ε0.1=3.49 → ε helps,
  hardest class; class 4 worst overall.
- `runs/epsilon_sweep_console.log`, `runs/epsilon_sweep_finishB.log` — raw stdout of those runs.

### 4.3 Policy training (`runs/taxi_v4/`) — provenance of the served policy
- `runs/taxi_v4/shaped_rainy_s2/final_model.zip` — the **0.7-rainy** policy that was promoted to
  `environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip` (scores 20/20 @ rainy 0.7).
- `runs/taxi_v4/shaped_rainy_s05_warm/best_model.zip` — the 0.5-rainy policy (promotion gated).
- `runs/taxi_v4/shaped_rainy_s03_warm/best_model.zip` — the 0.3-rainy policy.
- `runs/taxi_v4/*_train_*.log` — training logs. Other subdirs (`det`, `shaped_det*`, `smoke`,
  `rainy_*`) are earlier/worse attempts.
- Full policy provenance + eval numbers: `references/TAXI_POLICIES_AND_TRAJ_ANALYSIS.md`.

> Note: `runs/` is intended to be cleaned up later (see memory `taxi-training-cleanup-obligations`),
> keeping the winner. **Do not delete `runs/class2_lowfr/` or `runs/epsilon_sweep/`** — they are
> the experiment results, not training scratch.

---

## 5. Scripts (the code that produced everything)
| File | Role |
|------|------|
| `hard_taxi_benchmark.py` | **v1** generator (first experiments; most-used action, 5 steps → frozen). |
| `hard_taxi_data.py` | **v1** frozen output: 100 instances (`BENCHMARK`) + pool/distractors/counts. |
| `references/HARD_TAXI_SPEC.md` | v1 design spec. |
| `hard_taxi_benchmark_v2.py` | **v2** instance generator (`build_benchmark`, classes, candidates). |
| `run_class2_lowfr.py` | Runs first 40 class-2 seeds at one ε → `cells_eps<e>.csv`. |
| `run_class2_more.py` | Runs next 100 class-2 seeds at one ε → `cells2_eps<e>.csv`. |
| `run_epsilon_sweep.py` | Earlier multi-class vis{50,80} sweep (Phase A/B). |
| `finish_phaseB.py` | Finished class-4 to 20/class for the sweep. |
| `analyze_sweep.py` | Aggregates `runs/epsilon_sweep/cells.csv`. |
| `deepdive.py` | Per-example MC effort + logL scores for the "small ε wins" cases. |
| `p_pipeline.py` | `run_NON_DETERMINSTIC_single_experiment_PO` + firing-gate sweep. |
| `p_diagnosers.py` | Stochastic PO diagnoser; adaptive MC; gap-decorrelated seeding. |
| `p_executor.py` | `execute(..., fault_seed_offset=...)`. |
| `h_consts.py` | Seed-block offset layout. |

---

## 6. Uncommitted state (READ before any git cleanup)
Ahmad said **"we will commit, wait for now."** As of writing, uncommitted/untracked:
- Modified: `h_consts.py` (SIMULATION_OFFSET=5000), `p_pipeline.py` (firing-gate sweep),
  `environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip` (now the promoted 0.7 policy).
- Untracked: `hard_taxi_benchmark_v2.py`, `run_class2_lowfr.py`, `run_class2_more.py`,
  `run_epsilon_sweep.py`, `finish_phaseB.py`, `analyze_sweep.py`, `deepdive.py`,
  `environments/Taxi_v4/models/PPO/Taxi_v4__PPO_deter.zip` (deterministic backup),
  `references/TAXI_POLICIES_AND_TRAJ_ANALYSIS.md`, this file, and `runs/`.
- `p_diagnosers.py` MC seeding changes are committed (`2dbd4e7d`) + a small later inline edit.

**Suggested first action on return:** commit the scripts + `h_consts.py` + `p_pipeline.py` +
the promoted policy as one checkpoint so the experiment is reproducible from a clean tree.

## 7. Open / deferred items (from memory notes)
- Commit the above (gated on Ahmad).
- `r>0` retry-window seed plumbing; reset-per-trace MC optimization (CRN robustness).
- Fold the firing gate into benchmark regen.
- Possibly extend ε below 0.02 only if a fr<0.0025 regime is wanted (compute already ~370k traces/ε).
