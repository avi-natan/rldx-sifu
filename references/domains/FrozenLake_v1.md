# Domain Card — FrozenLake (8×8, slippery)

One-stop reference for the `FrozenLake_v1` fault-diagnosis domain. Same template as the MiniGrid and
Taxi cards; note how many fields differ (this is the point of having per-domain cards).

---

## 1. Identity
| | |
|---|---|
| **domain_name** (code key) | `FrozenLake_v1` |
| **gym id** | `FrozenLake-v1` (legacy `gym`, built via `make_wrapped_env`) |
| **observability** | **PARTIALLY OBSERVED, but via HIDDEN STATES, not aliasing.** The state is the exact tile index and is fully known WHEN observed; partial observability comes from HIDING a fraction of the trajectory (the visibility sweep → gaps). (Contrast MiniGrid, where the observation itself is ambiguous.) |
| **status** | Benchmarked (way2, known + unknown fault rate, epsilon sweeps done). |
| **branch** | `new-master-ai` (Ahmad's main AI branch; MiniGrid is a later child branch). |

---

## 2. Environment stochasticity
- **Mechanism:** the env's built-in **`is_slippery=True`** (set in `DOMAIN_KWARGS["FrozenLake_v1"]`,
  alongside the per-map `desc`). No custom wrapper (unlike MiniGrid).
- **Slip model:** the standard FrozenLake slip — the agent moves in the **intended** direction with
  prob **1/3**, and in **each perpendicular** direction with prob **1/3**. This smears movement
  outcomes (which is why near-twin faults are ambiguous here — see §6).
- **Parameter:** there is **no tunable numeric knob** — slipperiness is the fixed 1/3–2/3 model
  (either on or off). (Contrast MiniGrid's `MINIGRID_ACTION_PROB ∈ {0.3,0.5,0.7}`.)

---

## 3. Policy
- **Type:** a **per-map OPTIMAL policy**, computed (not trained): the optimal **discounted (γ=0.99)
  policy value-iterated over the SLIPPERY model** (`value_iteration_policy_slippery` in
  `frozen_lake_benchmark.py`) — slip-aware AND goal-directed. Deterministic.
- **Where it lives:** one policy per map, stored in `frozenlake_maps_8x8_slippery.json`. The driver
  sets `h_rl_models.HARD_CODED_POLICY = <this map's policy>` before each map, and
  `load_trained_model` returns a `FrozenLakeHardcodedPolicy` wrapping it (asserts it's set).
- **Note:** an earlier `risk_averse_reachability_policy` was a BUG (safe but not goal-directed →
  dawdled, ~0% success on ~19% of maps) and was replaced by the slippery-VI optimal policy.

---

## 4. State, observation, comparison
- **Raw state:** the integer tile index (0…63 for an 8×8 map). Carried directly.
- **Refiner:** `frozen_lake_refiner` = `int(raw_state)`.
- **Comparator:** `frozen_lake_compare` = integer equality (`int(s1) == int(s2)`). No aliasing —
  two states are equal iff the same tile.
- **`set_state`:** `FrozenLakeSetStepWrapper.set_state` does an EXACT restore (`unwrapped.s =
  raw_state`). No belief/sampling (the state is exactly known when observed).

---

## 5. Diagnosis settings (the standardized knobs)
| Knob | Value(s) | Notes |
|---|---|---|
| **epsilon** | **SWEEP** {0.1, 0.07, 0.05, 0.04, 0.03} (known-fr sweep, live) | ONE epsilon per run (cluster-array friendly). FrozenLake DOES sweep epsilon. **Finding (confirmed live): epsilon is FLAT for accuracy — rank ~2.31–2.34 across all 5 epsilons; it only trades compute.** |
| **fault rates** | **0.3, 0.5, 0.8** | `fault_rate_list` (default `(0.5, 0.8)`; pass `[0.3]` for the 0.3 runs). Swept INSIDE a run. |
| **visibilities** | **20, 40, 60, 80, 100 %** | swept INSIDE each run; the primary difficulty axis. |
| **candidate faults** | **10 per MAP** | way2: healthy + all 3 redirects of the 3 most-used actions (see §6). Per-MAP, from that map's policy profile. |
| **known/unknown fault rate** | **both** | known (default, told the rate) and unknown (`-ufr`, searches `fault_rate_candidates=0.1..1.0`, ~10× slower). |
| **max_exec_len** | **200** | trajectory cap (executor ignores gym's 100-step TimeLimit; checks only `terminated`). |

---

## 6. Fault space & candidate construction (way2)
- **Actions:** 4 — `0=LEFT, 1=DOWN, 2=RIGHT, 3=UP`.
- **way 2 (the ONLY way we use), `build_way2` in `frozen_lake_fault_modes.py`:** profile the healthy
  policy on the map → command counts → `a*` = **most-USED** action; **10 candidates** = healthy
  (identity) + all **3 redirects of the 3 most-used actions**. Every candidate corrupts a
  frequently-commanded (visible) action → **distinguishable**. The candidate set is **per-MAP**
  (from that map's profile) and includes the true execution fault `E` + healthy.
- **way 1 exists but is NOT used** — a Taxi-style near-twin recipe kept only as a documented
  baseline of why it fails here: the slip smears movement so corrupted directions overlap, and only
  ~22–31% of way-1 instances are solvable even at full observability. See
  `experimental results/FrozenLake_v1/BENCHMARK_WAYS.md`.

### Per instance (per map) — exactly how the execution fault & candidates are chosen
An instance is one MAP `i` (`instance_seed = 10 + i`). Both are produced by ONE call:
`build_way2(counts, seed=instance_seed)` (`frozen_lake_fault_modes.py`), where `counts` = the
healthy policy's action-usage Counter on that map (from a healthy `p_executor.execute(...)` rollout
at `instance_seed·SEED_BLOCK`).
- **`a*` = the MOST-used action** (`ranked[0]`, tie-break by action index).
- **Execution (injected) fault `E` = redirect `a*` to a SEEDED-RANDOM other action:**
  `e_target = random.Random(instance_seed·911 + 3).choice(actions ≠ a*)`; then
  `E = HEALTHY [0,1,2,3] with a*→e_target`. So E is per-map and seeded by that map's `instance_seed`
  (deterministic + reproducible — the SAME E every run of that map).
- **Candidates (10) = healthy(identity) + all 3 redirects of each of the 3 most-used actions**
  (3 actions × 3 targets = 9, + healthy = 10; deduped/truncated to 10). Always contains `E` and
  healthy. Per-map (depends on that map's usage profile), NOT global.
- **Functions/classes:** `build_way2` / `BUILDERS` (`frozen_lake_fault_modes.py`); the driver
  `multiple_experiment_FrozenLake_fault_benchmark` computes `counts` via `Counter(healthy_trajectory[1::2])`
  then calls `build_way2(counts, seed=instance_seed)`, passing `execution_fault_mode_name` /
  `possible_fault_mode_names` into `run_NON_DETERMINSTIC_single_experiment_PO`.

---

## 7. Seeds benchmark — human explanation
**The substrate.** The FrozenLake benchmark is **100 distinct 8×8 slippery maps**, each with its own
optimal (slippery-VI) policy, in `frozenlake_maps_8x8_slippery.json` — generated by
`generate_good_maps_and_policies` (`frozen_lake_benchmark.py`) from a fixed **seed 42** (fully
reproducible, maps AND policies). A difficulty gradient is baked in via `p_safe` tiers
(0.92×40, 0.85×30, 0.78×30 maps), with an exact finite-horizon (200-step) **solvability gate ≥ 0.2**
so every map is hard-but-solvable.

**The instances.** Here **each MAP is one instance** (its own geometry, optimal policy, `a*`, and 10
candidates), `instance_seed = 10 + map_index`. So the 100 maps ARE the 100 instances — variance
comes from the distinct maps, not from seeds × faults (contrast MiniGrid). Benchmark hardness comes
mainly from the FAULTS (way2) + partial observability (visibility) + the slip; the hole gradient is
secondary.

**A run.** One call fixes ONE epsilon (+ known/unknown) and sweeps `fault_rate_list × visibility`
over the 100 maps (or a map window). Epsilon is swept ACROSS runs (a job-array task per epsilon).

---

## 8. Experiment entry point (called from main)
- **Function:** `multiple_experiment_FrozenLake_fault_benchmark(epsilon=0.03, unknown_fault_rate=False,
  fault_rate_list=(0.5,0.8), maps_num=100, run_folder, map_start=0, map_end=None)` in
  `p_single_experiments.py` (way2 only).
- **Run:**
  ```
  python main.py --frozenlake [-ufr] --epsilon 0.04 --fl_fault_rates 0.5 0.8 \
                 --fl_group $SLURM_ARRAY_TASK_ID --fl_num_groups 10 -o <run_folder>
  ```
  `--fl_group/--fl_num_groups` split the 100 maps for a job array; omit to run all. `-ufr` = unknown
  fault rate. Sweep epsilon by submitting once per epsilon value.

---

## 9. Files & functions (this domain's implementation)
| File | FrozenLake pieces |
|---|---|
| `h_wrappers.py` | `FrozenLakeSetStepWrapper` (exact `set_state`), `DOMAIN_KWARGS["FrozenLake_v1"]={is_slippery, desc}`, `FROZENLAKE_DESC`, `FROZENLAKE_SLIPPERY` |
| `h_raw_state_comparators.py` | `frozen_lake_compare` (int equality) |
| `h_state_refiners.py` | `frozen_lake_refiner` (int) |
| `h_rl_models.py` | `FrozenLakeHardcodedPolicy`, `HARD_CODED_POLICY` (mutated per-map), `load_trained_model` branch |
| `p_single_experiments.py` | `multiple_experiment_FrozenLake_fault_benchmark` (way2 driver) |
| `frozen_lake_fault_modes.py` | `build_way1` / `build_way2` / `BUILDERS` (a* + 10 candidates) |
| `frozen_lake_benchmark.py` | `generate_good_maps_and_policies`, `value_iteration_policy_slippery`, `load_pairs_from_json`, `print_map_and_policy` |
| `frozenlake_maps_8x8_slippery.json` | the 100-map + policy substrate (seed 42) |
| `main.py` | flags `--frozenlake --fl_fault_rates --fl_group --fl_num_groups` (+ `-ufr --epsilon -o`) + dispatch |
| `experimental plot scripts/` | `plot_epsilon_sweep.py`, `plot_fault_rate_view.py`, `plot_known_vs_unknown.py`, `plot_frozenlake_unknown*.py`, `plot_frozenlake_known_vs_unknown.py` |

---

## 10. Cluster & outputs
- **Cluster method:** no dedicated committed sbatch — runs use an in-place sbatch as a **job array
  over the 6 epsilons** (e.g. arrays `20797314/20797315`), with `--fl_group/--fl_num_groups` for map
  splitting. See [[cluster-run-workflow]]. (Recommend a committed `sb_frozenlake.sbatch` like MiniGrid.)
- **Output filename:** `frozenlake_way2_PO_{known,unknown}_fr_epsilon_<E>_INJFR_<rate>_MAPS_<start>-<end>.xlsx`,
  under `experimental results/FrozenLake_v1/<run_folder>/` (organized into `known_fr_experiments/`,
  `UNknown_fr_experiments/`, `known_vs_unknown_comparison/`). Merge per-group xlsx afterward.
- **Records tag:** `benchmark_way`, `a_star`, `execution_fault`, plus the shared schema (incl.
  `real_fault_prob` = injected fault rate — see [[excel-results-schema]]).

---

## 11. Results so far — LIVE (avg real-fault rank of 10; lower better; random ≈ 5.5)
Computed from the committed known-fr way2 xlsx (N=7400: 100 maps × 5 vis × 3 fault rates × 5 epsilons).
- **Overall:** mean rank **2.33**, top-1 **0.56**, top-3 **0.83**.
- **By fault rate** (monotone): fr 0.3 → 3.11 / 0.36 · fr 0.5 → 2.40 / 0.55 · fr 0.8 → **1.45 / 0.78**.
- **By visibility** (monotone): 20% → 3.12 / 0.43 · 40% → 2.49 · 60% → 2.23 · 80% → 2.00 · 100% → **1.80 / 0.69**.
- **Best cell** (fr 0.8, 100% vis): rank **1.13**, top-1 **0.913**.
- **Epsilon:** FLAT for accuracy (rank ~2.31–2.34 across the 5 epsilons) — only trades compute. The
  real axes are **fault rate** and **visibility**. (way1 baseline, for contrast: ~2.83 / ~20%.)

---

## 12. Gotchas / caveats
- **`HARD_CODED_POLICY` is a mutated global** — the driver sets it per map before diagnosing; don't
  assume a single fixed policy.
- **Epsilon is inert for accuracy** — a sweep only shows the compute/precision tradeoff, not an
  accuracy change.
- **Unknown-fr is ~10× slower** than known-fr (searches the 0.1..1.0 rate grid).
- **way1 is a documented dead-end** — kept only as the honest Taxi-parallel baseline; never use it.
- **Regenerating the substrate:** `python frozen_lake_benchmark.py` (seed 42) rebuilds the 100-map
  json; the old deterministic benchmark (`frozen_lake_random_envs.py`, `frozenlake_100_pairs.json`)
  was deleted.
