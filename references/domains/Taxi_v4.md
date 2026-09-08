# Domain Card — Taxi-v4 (rainy)

One-stop reference for the `Taxi_v4` fault-diagnosis domain. Same 12-section template as the MiniGrid
and FrozenLake cards. Taxi is the most curated domain (hand-built "hard class-2" near-twin instances
for the epsilon study).

> Some numbers below come from ~3-week-old memories (marked "≈" / "from memory"); verify against the
> live xlsx if exact figures are needed.

---

## 1. Identity
| | |
|---|---|
| **domain_name** (code key) | `Taxi_v4` |
| **gym id** | `Taxi-v4` — a **custom "rainy" Taxi variant** (stochastic movement), built via `gymnasium.make` in `make_wrapped_env` |
| **observability** | **PARTIALLY OBSERVED via HIDDEN STATES** (visibility sweep → gaps). The state is the exact discrete Taxi state (0…499), fully known WHEN observed; no aliasing. (Same PO style as FrozenLake; contrast MiniGrid.) |
| **status** | Added; **the hard class-2 epsilon experiment is the headline result**. Per CLAUDE.md still flagged "not yet fully tested". |
| **branch** | `new-master-ai` |

---

## 2. Environment stochasticity
- **Mechanism:** `DOMAIN_KWARGS["Taxi_v4"] = {is_rainy: True, rainy_probability: 0.7, fickle_passenger: False}`.
  "Rainy" is the movement-stochasticity analog of FrozenLake "slippery".
- **Parameter:** **`rainy_probability = 0.7`** for the benchmark (higher = less noisy). This is the
  fixed benchmark stochasticity (wrapper default). Lower = more stochastic.
- **Note:** the hard-instance *analysis* tool sweeps `rainy_probability` down to **0.3** (harder
  regime) by building the env directly — but the committed benchmark runs at **0.7**.

---

## 3. Policy
- **Type:** a **trained PPO** policy (trained under rainy 0.7), served **TABULATED**: at load,
  `build_taxi_hardcoded_policy` precomputes the net's deterministic action for **all 500 states** →
  a dict (`TaxiHardcodedPolicy`), so diagnosis is an O(1) lookup (same pattern MiniGrid later reused).
  Deterministic.
- **Model:** `environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip` (the served policy; ≈**94–98%**
  success — the earlier "55%" was a measurement artifact, see [[taxi-v4-trained-policy]]).
- **Gotcha (CODE_SCAN):** the loader keys on `Taxi_v4__PPO.zip`; a freshly-trained
  `Taxi_v4_PPO_rainy_0.7_steps_..._seed_42` had **no `.zip`** and was never loaded — likely why Taxi
  was tagged "untested". A `rainy-0.5` candidate lives at `runs/taxi_v4/shaped_rainy_s05_warm`
  (promotion gated).

---

## 4. State, observation, comparison
- **Raw state:** the integer Taxi state (0…499). Carried directly.
- **Refiner:** `taxi_refiner` = `int(raw_state)`.
- **Comparator:** `taxi_compare` = integer equality. No aliasing.
- **`set_state`:** `TaxiV4SetStepWrapper.set_state` = exact restore (`unwrapped.s = int(raw_state)`).
  No belief/sampling.

---

## 5. Diagnosis settings (the standardized knobs)
| Knob | Value(s) | Notes |
|---|---|---|
| **epsilon** | **SWEEP** {0.1, 0.07, 0.05, 0.04, 0.03, 0.02} | ONE epsilon per run. In the actual experiments (committed driver, fr 0.3) ε is **FLAT** for accuracy (rank ~2.88–2.96, live) — it only trades compute. (A pathological low-fr side-study where ε mattered barely matters; see §11 aside.) |
| **fault rate** | **depends on the experiment** | The committed `main` driver (class-2) FIXES **0.3** (only visibility swept). The deeper ε study used very low {0.02, 0.01, 0.005, 0.0025}; the easy v1 experiment used {0.5, 0.8}. |
| **visibilities** | **20, 40, 60, 80, 100 %** | swept INSIDE each run. |
| **candidate faults** | **10 per instance** | class-2: E + 5 other a*-maps + 4 tier-2 twins (see §6). Frozen per seed. |
| **known/unknown fault rate** | **both** | known (default) / unknown (`-ufr`, searches 0.1..1.0, ~10× slower). |
| **max_exec_len** | **200** | |
| **stochasticity** | rainy_probability **0.7** | benchmark default. |

---

## 6. Fault space & candidate construction (hard class-2, v2)
- **Actions:** 6 — `0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff` (movement = 0–3).
- **Instances** come from `hard_taxi_benchmark_v2.build_benchmark` (class-based, rainy 0.7). We use
  **CLASS 2** = `a*` is the **LEAST-used** commanded action with count ≥ 2 (rarest *movement*) →
  hardest (a fault on a rarely-commanded action fires rarely → thin likelihood margins). Class 1
  (absolute least-used = Pickup/Dropoff, re-converges → impossible) was dropped.
- **Why class-2 is hard** (near-twin theory, [[taxi-hard-instance-analysis]]): a twin differing from
  E on an action commanded exactly once, at low fault rate + high stochasticity + re-convergent
  geometry, makes `p_E/p_C→1` so the MC diagnoser needs many sims to separate them — the regime
  where epsilon matters.

---

### Per instance (per seed) — exactly how the execution fault & candidates are chosen
An instance is `(main_seed, base = main_seed·SEED_BLOCK, class_id=2, a*, E, [10 candidate maps])`,
built by `build_benchmark` (`experiments_scripts/hard_taxi_benchmark_v2.py`):
1. **Profile:** roll the healthy policy in the rainy-0.7 env at the seed → per-action command
   `counts`.
2. **`a*`** = `class_picks(counts, seed)[2]` = the **least-used movement action with count ≥ 2**
   (seeded tie-break).
3. **`b`** = `select_b(counts, a*)` = the **2nd-rarest** commanded action ≠ a* (the tier-2 twin axis).
4. **Execution (injected) fault `E`** = redirect `a*` to a **SEEDED-RANDOM** other action:
   `e_target = rng.choice(t ≠ a*)`; `E = healthy [0..5] with a*→e_target` (`build_candidates`).
5. **Candidates (10)** = `[E]` + the **5 other maps of a\*** (all 6 `a*→t` redirects, incl. healthy-at-a*
   and E) + **4 tier-2 twins** (keep E's a*, redirect `b` to 4 other actions). E is first; all distinct.
   **Frozen per seed** → identical across every visibility/epsilon cell of that instance.
- **Functions/classes:** `build_benchmark`, `class_picks`, `select_b`, `build_candidates`
  (`hard_taxi_benchmark_v2.py`); the driver passes `execution_fault_mode_name` /
  `fixed_candidate_fault_modes` into `run_NON_DETERMINSTIC_single_experiment_PO`.

---

## 7. Seeds benchmark — human explanation
The Taxi benchmark is **curated hard instances**, not a fixed map/fault grid. We scan seeds and keep
the first `num_seeds` (=100) that yield a valid **class-2** instance (a least-used movement action
with count ≥ 2 that also has a 2nd action for tier-2 twins). **Each seed = one instance**
(`base = main_seed·SEED_BLOCK`, a per-instance seed block, see [[seeding-namespace-redesign]]), with
its own `a*`, injected fault `E`, and frozen 10-candidate set. A run fixes ONE epsilon (+known/unknown),
FIXES fault_rate = 0.3, and sweeps **visibility** over the 100 instances. Epsilon is swept ACROSS runs.
Hardness is engineered (rare-action fault + low rate + stochasticity + re-convergence) so that the
epsilon / #-simulations tradeoff is actually observable (unlike easy FrozenLake/v1 instances).

---

## 8. Experiment entry point (called from main)
- **Function:** `multiple_experiment_Taxi_v4_hard_class2_PO(epsilon=0.03, num_seeds=100, run_folder,
  unknown_fault_rate=False)` in `p_single_experiments.py` — this is the **DEFAULT** (the `else`
  branch in `main.py`; no `--taxi` flag).
- **Run:**
  ```
  python main.py --epsilon 0.04 [-ufr] -o <run_folder>      # Taxi hard class-2 (default path)
  ```
  Sweep epsilon by submitting once per epsilon value (one xlsx per run). (The deeper ε study used the
  standalone `experiments_scripts/run_class2_*.py` with very low fault rates.)

---

## 9. Files & functions (this domain's implementation)
| File | Taxi pieces |
|---|---|
| `h_wrappers.py` | `TaxiV4SetStepWrapper`, `DOMAIN_KWARGS["Taxi_v4"]={is_rainy, rainy_probability:0.7, fickle_passenger}` |
| `h_raw_state_comparators.py` | `taxi_compare` (int equality) |
| `h_state_refiners.py` | `taxi_refiner` (int) |
| `h_rl_models.py` | `TaxiHardcodedPolicy`, `build_taxi_hardcoded_policy` (tabulate 500 states), `load_trained_model` branch |
| `p_single_experiments.py` | `multiple_experiment_Taxi_v4_hard_class2_PO` (+ `multiple_experiment_Taxi_v4_NON_DETERMINSTIC_PO`) |
| `experiments_scripts/hard_taxi_benchmark_v2.py` | `build_benchmark`, `class_picks`, `select_b`, `build_candidates` (class-2 v2) |
| `experiments_scripts/hard_taxi_benchmark.py`, `hard_taxi_data.py` | v1 (most-used) generator + frozen data |
| `experiments_scripts/analyze_hard_instance.py` | hardness analysis tool (search/report; sweeps rainy_probability) |
| `experiments_scripts/` | `train_taxi_v4_ppo.py`, `eval_taxi_policy.py`, `run_epsilon_sweep.py`, `run_class2_lowfr.py`, `run_class2_more.py` |
| `main.py` | default (`else`) dispatch → Taxi class-2 |
| `environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip` | served policy |

---

## 10. Cluster & outputs
- **Cluster method:** no dedicated committed sbatch; the epsilon sweep runs one process/task per
  epsilon (job array or parallel procs). See [[cluster-run-workflow]].
- **Output filename:** `hard_taxi_PO_{known,unknown}_fr_epsilon_<eps>_SEEDS_<n>.xlsx` under
  `experimental results/Taxi_v4/<run_folder>/` (e.g. `eps_full/`, `hard_class2_epsweep/`,
  `hard_class2_unknown_fr_epsweep/`).
- **Records tag:** `a_star`, `real_fault_prob` (= injected fault rate), `real_fault_rank`, + the
  shared 47-col schema ([[excel-results-schema]]).

---

## 11. Results (the experimental results that matter) — avg real-fault rank of 10; lower better; random ≈ 5.5
**Committed class-2 benchmark, known-fr, fr 0.3** — live from the xlsx (N=3000: 100 seeds × 5 vis × 6 epsilons):
- **Overall:** mean rank **2.94**, top-1 **0.32**, top-3 **0.68**.
- **By visibility** (the real difficulty axis, monotone): 20% → **4.16 / 0.17** · 40% → 3.42 / 0.23 ·
  60% → 2.91 / 0.32 · 80% → 2.36 / 0.39 · 100% → **1.83 / 0.51**.
- **Epsilon:** FLAT (rank ~2.88–2.96 across {0.02…0.1}) — in the actual experiments, more sims don't
  change accuracy; epsilon only trades compute.

_(Aside, low priority: a one-off very-low-fault-rate side-study — standalone `run_class2_*.py`,
fr 0.0025–0.02, `runs/class2_lowfr/*.csv` — showed epsilon CAN become monotone (rank 3.55→3.13) if
the problem is made pathologically hard. It barely matters and is not part of the experimental
results above.)_

---

## 12. Gotchas / caveats
- **Fault rate is experiment-dependent** — the committed `main` driver uses **0.3**; the ε study used
  {0.02,0.01,0.005,0.0025}; v1 used {0.5,0.8}. Don't assume one value.
- **`rainy_probability`:** benchmark **0.7**; the analysis tool sweeps it (down to 0.3) by building
  the env directly (`make_wrapped_env` hardcodes 0.7).
- **Model-name gotcha:** loader wants `Taxi_v4__PPO.zip`; freshly-trained runs used a different name
  and were silently not loaded (the "untested" flag).
- **Instances are curated by seed-scan** (not a fixed file for v2) — reproducible from the seed, but
  regenerated each run via `build_benchmark`; v1 froze its data in `hard_taxi_data.py`.
- **Some experiment code was historically uncommitted** on `new-master-ai` (Ahmad said "wait");
  verify what's committed before relying on it.
- **Epsilon matters here (unlike FrozenLake)** — but only in the low-rate / partial-visibility /
  high-stochasticity regime; at fr 0.5 + full visibility near-twins are trivially separable.
