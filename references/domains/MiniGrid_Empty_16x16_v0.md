# Domain Card — MiniGrid Empty 16×16

One-stop reference for the `MiniGrid_Empty_16x16_v0` fault-diagnosis domain: what settings we use,
how it's wired, and how to run it. (Companion cards exist for FrozenLake and Taxi-v4.)

---

## 1. Identity
| | |
|---|---|
| **domain_name** (code key) | `MiniGrid_Empty_16x16_v0` |
| **gym id** | `MiniGrid-Empty-16x16-v0` (built via `gymnasium.make`, `disable_env_checker=True`) |
| **observability** | **PARTIALLY OBSERVED** — the diagnoser sees only the agent's egocentric observation, never the true position. This is the domain's defining feature. |
| **status** | Integrated + benchmarked. Noise 0.7 & 0.3 benchmarks DONE; 0.5 running (2026-09-08). |
| **branch** | `minigrid-integration` |

---

## 2. Environment stochasticity (the "slippery" analog)
- **Parameter:** `MINIGRID_ACTION_PROB` (in `h_wrappers.py`) = probability the intended action
  executes; otherwise a random action is drawn. MiniGrid is deterministic by default, so we inject
  this ourselves.
- **Mechanism:** our **`SeededStochasticActionWrapper`** (NOT MiniGrid's own `StochasticActionWrapper`,
  which draws its coin from the GLOBAL numpy RNG and is not seed-controllable). Both the slip-coin
  and the replacement action come from `self.np_random` (seeded by `reset(seed)`).
- **Slip semantics:** with prob `p` the intended action fires; with prob `1-p` it is **re-rolled
  uniformly over the 3 real actions `{0=left,1=right,2=forward}`** (can coincidentally repeat the
  intended one). So at p=0.7 the effective breakdown is ~0.8 intended / ~0.2 genuinely-different.
- **Values we work with:** **0.3, 0.5, 0.7** ("0.7 success" = 70% intended). A trained policy
  exists for each; each is one benchmark noise level (difficulty: 0.7 easy → 0.5 mid → 0.3 hard).
- **Control:** `main.py --mg_noise {0.3,0.5,0.7}` calls `h_wrappers.set_minigrid_action_prob()`,
  which sets BOTH the env noise AND selects the matching trained policy (they always agree).
  MiniGrid-Empty ONLY — the constant is read only in MiniGrid code paths.

---

## 3. Policy
- **Type:** a **trained OBSERVATION-input PPO** policy (deterministic / greedy). One per noise level,
  trained UNDER that noise so the greedy policy is robust to it.
- **Observation fed to the net:** `flatten(image)/255 ++ one_hot(direction)` = 151-dim vector
  (`h_rl_models.minigrid_obs_vector`; must match `experiments_scripts/train_minigrid_ppo.py`
  `ImgDirFlatWrapper`). Mission text dropped.
- **Models:** `environments/MiniGrid_Empty_16x16_v0/models/PPO/MiniGrid_Empty_16x16_v0__PPO__noise{0.3,0.5,0.7}.zip`.
  All three: **100% greedy success** (≈40 steps @0.7, ≈58 @0.5, ≈99 @0.3 under their noise).
- **Diagnosis-time form:** loaded and **TABULATED** over all 784 states into a dict
  (`MiniGridTabulatedPolicy`, like the Taxi policy) → O(1) lookup instead of a NN forward pass on
  every Monte-Carlo step (~900× faster; verified identical to `model.predict`). Because the policy
  is obs-based and aliased states share an observation, they share an action — the table is an
  observation policy in disguise.
- **Training:** `experiments_scripts/train_minigrid_ppo.py` (+ `sb_minigrid_train*.sbatch`); the old
  `MiniGridEmptyHardcodedPolicy` greedy navigator is kept for reference but no longer used.

---

## 4. State, observation, comparison (PO specifics)
- **Raw state:** `(col, row, dir)` — `dir ∈ {0=E,1=S,2=W,3=N}`. This is what the trajectory/wrapper
  carry and what the (identity) refiner passes to the policy.
- **Refiner:** `minigrid_refiner` = **identity** (the tabulated policy takes the raw state directly).
- **Observation (what the diagnoser is allowed to use):** the DEFAULT MiniGrid obs = egocentric
  7×7×3 image + `direction`. The **view key** is `(image.tobytes(), direction)`
  (`build_minigrid_view_maps`). Only the object channel of the image varies in Empty.
- **Comparator:** `make_minigrid_view_comparator` — two states are "equal" iff they produce the same
  `(image, direction)`. Value-based (`bytes`/`int`), not object identity.
- **`set_state`:** does NOT restore the true state. It looks up the observed view, finds all states
  consistent with it (`view2states`), and **samples one** (seeded `_belief_rng`); direction is
  observed so only POSITION is sampled. Aliasing: with direction, up to **63** states share the
  open-interior view (was 252 image-only). The MC diagnoser averages over "where am I?".

---

## 5. Diagnosis settings (the standardized knobs)
| Knob | Value(s) | Notes |
|---|---|---|
| **epsilon** | **0.04** only | Adaptive-MC confidence threshold. MiniGrid does NOT sweep epsilon (unlike Taxi/FrozenLake); 0.04 fixed. (Smoke test uses 0.05.) |
| **fault rates** | **0.3, 0.5, 0.8** | Injected fault-firing probability; ONE per run (not swept inside a run). |
| **visibilities** | **20, 40, 60, 80, 100 %** | `MINIGRID_VISIBILITIES`; swept INSIDE each run. |
| **candidate faults** | **10 per instance** | true fault + identity + 8 hardest confusers (per-instance, static; see §6). |
| **known/unknown fault rate** | **known** (default) | `unknown_fault_rate=False`. |
| **max_exec_len** | **80** | trajectory cap (< MAX_STATES=200). |

---

## 6. Fault space & candidate-set construction
- **Fault space:** a fault corrupts action EXECUTION for the 3 nav actions and maps ONLY within
  `{0,1,2}` → every function `{0,1,2}→{0,1,2}` = 3³=27; minus identity = **26 injected faults**
  (`MINIGRID_FAULTS`). Identity (`{0:0,1:1,2:2}`) = "no fault", **candidate-only, never injected**.
- **Per-instance candidate set (STATIC, size 10):** for true fault T,
  `distance(T,C)` = # of the 3 actions mapped differently. Candidate set = **always {T, identity}**,
  then filled with the hardest others (all distance-1, then SEEDED-RANDOM distance-2). Built once by
  `build_minigrid_candidate_sets(seed=42)` → `MINIGRID_CANDIDATE_SETS`. Policy/noise-INDEPENDENT →
  fixed across all runs. (This is the key difference from FrozenLake/Taxi, which use a shared set.)

### Per instance (per seed) — exactly how the execution fault & candidates are chosen
An instance is `(fault_index fi ∈ 0..25, seed_index si ∈ 0..2)`.
- **Execution (injected) fault = `MINIGRID_FAULTS[fi]`** — SYSTEMATIC, not random: we inject the
  `fi`-th of the 26 faults (enumeration order of all `{0,1,2}→{0,1,2}` minus identity), each with 3
  seeds. `si` (the seed) changes ONLY the trajectory's noise realization, NOT which fault is
  injected. So in a run every one of the 26 faults is injected exactly 3× (× 5 visibilities).
- **Candidates (10) = `MINIGRID_CANDIDATE_SETS[_mg_spec(execution_fault)]`** — a function of the
  INJECTED fault only, so all 3 seeds of a fault share the SAME 10 candidates. Frozen at import by
  `build_minigrid_candidate_sets(seed=42)`: always `{T, identity}` + the 8 hardest others (all 6
  distance-1, then 2 seeded-random distance-2). Static → identical across every noise/fault-rate run.
- **Functions/classes:** `MINIGRID_FAULTS`, `_mg_spec`, `build_minigrid_candidate_sets`,
  `MINIGRID_CANDIDATE_SETS` (all `p_single_experiments.py`); the driver reads `MINIGRID_FAULTS[fi]`
  + `MINIGRID_CANDIDATE_SETS[spec]` per work-unit and passes them as `execution_fault_mode_name` /
  `fixed_candidate_fault_modes` to `run_NON_DETERMINSTIC_single_experiment_PO`.

---

## 7. Seeds benchmark — human explanation
**The idea.** Empty-16×16 has a FIXED start, FIXED goal, and a FIXED deterministic policy, so the
only things that vary per instance are the noise realization and the injected fault. We therefore
build **one fixed benchmark** and evaluate it under different conditions:

- **The fixed benchmark = 26 faults × 3 seeds = 78 instances.** Each instance injects one real fault
  under a given seed; it carries its own static 10-candidate set (§6, always incl. the true fault
  and identity).
- **A "run" fixes ONE `(noise, fault_rate)`** and diagnoses every instance across the **full
  visibility sweep 20→100%**. So the SAME 78 instances are compared across conditions — a clean
  controlled comparison (only the condition changes, never the instances).
- **Work-units** = fault × seed × visibility = **26·3·5 = 390 diagnoses per run**. A full noise level
  = 3 fault-rate runs = **1170 diagnoses**.

**Seeding.** Instance index `= fault_index·num_seeds + seed_index`; `instance_seed = 10 + index`;
the block base passed to the diagnoser is `instance_seed · SEED_BLOCK` (per-instance seed namespace,
see [[seeding-namespace-redesign]]). The index depends only on (fault, seed), so the same instance is
reproduced identically across every `(noise, fault_rate)` run.

---

## 8. Experiment entry point (called from main)
- **Function:** `multiple_experiment_MiniGrid_fault_benchmark(epsilon=0.04, unknown_fault_rate=False,
  fault_rate, num_seeds=3, run_folder, unit_start, unit_end, domain_name)` in `p_single_experiments.py`.
- **Run locally (one slice):**
  ```
  python main.py --minigrid --mg_noise 0.7 --mg_fault_rate 0.5 --epsilon 0.04 -o <run_folder>
  ```
  (`--mg_group`/`--mg_num_groups` split the 390 units for a job array; omit to run all.)
- **num_seeds** is fixed at 3 in the `main.py --minigrid` dispatch.

---

## 9. Files & functions (this domain's implementation = the per-domain contract)
| File | MiniGrid pieces |
|---|---|
| `h_wrappers.py` | `MiniGridSetStepWrapper` (localize+sample `set_state`), `SeededStochasticActionWrapper`, `build_minigrid_view_maps`, `make_wrapped_env` branch, `MINIGRID_ACTION_PROB` + `set_minigrid_action_prob`, `MINIGRID_SUPPORTED_NOISE` |
| `h_raw_state_comparators.py` | `make_minigrid_view_comparator` (+ lazy registration in `comparators`) |
| `h_state_refiners.py` | `minigrid_refiner` (identity) in `refiners` |
| `h_rl_models.py` | `minigrid_obs_vector`, `MiniGridTabulatedPolicy`, `build_minigrid_tabulated_policy`, `load_trained_model` branch |
| `p_single_experiments.py` | `MINIGRID_FAULTS` (26), `build_minigrid_candidate_sets` / `MINIGRID_CANDIDATE_SETS`, `MINIGRID_VISIBILITIES`, `multiple_experiment_MiniGrid_fault_benchmark` |
| `main.py` | flags `--minigrid --mg_group --mg_num_groups --mg_noise --mg_fault_rate` + dispatch |
| `sb_minigrid.sbatch` | cluster run (see §10) |
| `experiments_scripts/train_minigrid_ppo.py` | policy training |
| `experiments_scripts/minigrid_empty_smoke.py` | single end-to-end sanity test |
| `experimental plot scripts/plot_minigrid_benchmark.py` | benchmark curves |
- **env kwargs (`DOMAIN_KWARGS`):** none for MiniGrid (stochasticity is via the wrapper, not a make-kwarg).

---

## 10. Cluster & outputs
- **sbatch:** `sb_minigrid.sbatch` — one submission = one `(noise, fault_rate)`; `--array 0-389`
  + `--mg_num_groups 390` → **MAX PARALLELISM: 1 diagnosis per task**; 1 core/task.
- **Launch a full noise level** (all 3 fault rates into ONE per-noise folder):
  ```
  R="experimental results/MiniGrid_Empty_16x16_v0/minigrid_bench_noise0_7"; mkdir -p "$R/logs"
  for FR in 0.3 0.5 0.8; do
    sbatch --export=ALL,MG_NOISE=0.7,MG_FR=$FR --output="$R/logs/fr${FR}-%A_%a.out" sb_minigrid.sbatch
  done
  ```
  → 3 arrays × 390 = **1170 tasks** per noise level (~200+ run concurrently).
- **Output layout:** `experimental results/MiniGrid_Empty_16x16_v0/minigrid_bench_noise{tag}/`
  with `xlsx/` (per-task results) + `logs/` (SLURM) + `plots/` (+ `_MERGED.xlsx`).
- **Plots:** `plot_minigrid_benchmark.py <noise_tag>` → 6 figures (rank vs visibility & vs fault
  rate, 2 pooled, 2 time), metric = **avg real-fault rank** (1=best, 10=worst; random=5.5), drops
  `PLOT_PROVENANCE.txt`.

---

## 11. Results so far (avg real-fault rank of 10; lower better; random = 5.5)
| noise | overall mean rank | top-1 | notes |
|---|---|---|---|
| 0.7 | **2.34** | 0.461 | easiest; monotonic in fault rate & visibility |
| 0.5 | **2.93** | 0.354 | middle of the gradient |
| 0.3 | **3.68** | 0.174 | hardest/limit but still > random |

Clean 3-level noise gradient (0.7 → 0.5 → 0.3 = 2.34 → 2.93 → 3.68). Cross-noise comparison plots:
`noise_comparison/plots/` (`plot_minigrid_noise_comparison.py`). All 1170 diagnoses per noise; avg
~25 s/diagnosis.

---

## 12. Gotchas / caveats
- **`import os as _os`** in `p_single_experiments.py` — use `_os`, not `os` (a bare-`os` NameError
  silently wasted a whole benchmark run once).
- **pygame-ce:** `pip install minigrid` overwrites `pygame`; on Windows it's blocked by Application
  Control (`import pygame` breaks). Fix: `pip uninstall -y pygame-ce pygame && pip install pygame==2.6.1`.
  On Linux/cluster pygame-ce is fine.
- **Perf:** `gen_obs` is stubbed to a no-op on the diagnosis simulator (never used; ~75% of runtime),
  env-checker disabled, view maps precomputed. A "fast reset" was tried and REVERTED (grid-gen
  consumes np_random → changes results).
- **Noise level gates diagnosability:** 0.3 is near the diagnosable limit; don't treat it as a
  primary operating point.
