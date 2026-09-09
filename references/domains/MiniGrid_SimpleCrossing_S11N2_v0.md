# Domain Card — MiniGrid SimpleCrossing S11N2

> Status: **integration wired, GATED on the trained obs-policy.** All plumbing (wrapper, per-seed
> layout, comparator, refiner, per-layout policy loader, driver, main, sbatch) is in place and tested
> with a stub net. What remains before a real run: (a) the trained obs-input PPO for this domain must
> exist and generalize; (b) resolve the trajectory-length knobs (§5). Reuses **Empty's benchmark**
> (26 faults + static hardest-10 candidates) and **fault_rate 0.5** for now — see [[simplecrossing-integration-decisions]].

## 1. Identity
- **Code key:** `MiniGrid_SimpleCrossing_S11N2_v0`   **Gym id:** `MiniGrid-SimpleCrossing-S11N2-v0`
- **Custom-registered** (not a stock id) via `_register_minigrid_custom_envs()` in `h_wrappers.py`:
  `CrossingEnv(size=11, num_crossings=2, obstacle_type=Wall)`.
- **Partially observed**, like Empty: the diagnoser sees only the egocentric 7×7×3 view + direction.
- **Key difference from Empty — PER-SEED LAYOUT:** agent start `(1,1)` facing East and goal `(9,9)`
  are FIXED, but the **2 wall barriers and their gaps MOVE every seed** (verified 10/10 distinct
  layouts). This is the FrozenLake-style map diversity, but from the seed rather than a map file.
  Walls block movement (you can't stand on them); a slip into a wall just stalls.

## 2. Environment stochasticity (the "slippery" analog)
- Same mechanism as Empty: our `SeededStochasticActionWrapper(prob=MINIGRID_ACTION_PROB)` — with
  prob the intended action executes, else a random one of the 3 real actions {0=left,1=right,
  2=forward}. Seedable (folds into per-trace seeding); MiniGrid's own global-RNG wrapper is NOT used.
- **No new code needed** — `make_wrapped_env`'s `if is_minigrid:` branch wraps EVERY `MiniGrid*`
  domain, so Crossing gets it automatically. Verified live: intended=FORWARD executed≠intended =
  20.4% @noise0.7, 34.0% @noise0.5 (matches `(1-prob)·2/3`).
- **Trained/benchmarked noise levels:** 0.5 and 0.7 (`MINIGRID_SUPPORTED_NOISE` also lists 0.3, but
  we do not train a 0.3 Crossing policy). Selected with `--mg_noise` (picks env noise + policy).

## 3. Policy
- **Obs-input PPO** (`MlpPolicy` on the 151-dim vector = flatten(image)/255 ++ one_hot(direction)),
  size-independent — the SAME net input shape as Empty, so it can navigate ANY layout from the view.
  It must **generalize** across the random per-seed layouts (train resets to a fresh layout each
  episode). **Gate:** greedy success on held-out seeds before it's used for diagnosis.
- **Served FAST via PER-LAYOUT tabulation** (`MiniGridPerLayoutTabulatedPolicy`, `h_rl_models.py`):
  the net is loaded once; the first time the policy is queried while a given layout is active, it
  tabulates the net's greedy action over exactly that layout's FREE states (rebuilding each state's
  observation from the wrapper's stored view-map image bytes — no re-render), caches it keyed by the
  layout seed, and serves O(1) lookups after. Behaviourally identical to `model.predict`.
- **Training:** `experiments_scripts/train_minigrid_ppo.py` + `sb_minigrid_train_crossing.sbatch`
  (array noise{0.7,0.5}×seed{0,1}, 10M steps, SubprocVecEnv). `--device` flag added for CPU/GPU A/B.
- **Model path the loader expects** (place the promoted winner here after training):
  `environments/MiniGrid_SimpleCrossing_S11N2_v0/models/PPO/MiniGrid_SimpleCrossing_S11N2_v0__PPO__noise{prob}.zip`
  Training saves to `runs/crossing_s11n2_train/<tag>/model.zip`; copy the generalizing one across.

## 4. State, observation, comparison (PO specifics)
- **Raw state:** `(col, row, dir)`. **Refiner:** identity (`minigrid_refiner`).
- **View maps are WALL-AWARE and PER-LAYOUT:** `build_minigrid_view_maps_for_layout(domain, seed)`
  enumerates only FREE (non-wall) interior cells × 4 dirs — wall cells are never states. Cached by
  `(domain, layout_seed)`.
- **Comparator:** the default MiniGrid **view comparator**, but for per-seed-layout domains it reads
  the CURRENT instance's map from `_MINIGRID_ACTIVE_MAPS[domain]` (set by the wrapper on first reset),
  not a single fixed map. Two states with the same (image, direction) are equal.
- **`set_state`:** localizes the observed view to the states consistent with it (in THIS layout) and
  samples one consistent position (keeps the observed heading). Never lands on a wall (tested).
- **Per-seed-layout reset contract** (`MiniGridSetStepWrapper.reset`): the FIRST reset fixes
  `_layout_seed` (= that seed's layout) and publishes `_MINIGRID_ACTIVE_MAPS` + `_MINIGRID_ACTIVE_LAYOUT_SEED`;
  EVERY reset rebuilds that SAME layout, then reseeds noise + belief RNGs from the per-call seed →
  layout fixed across a diagnosis, noise varies per trace. Tested in `experiments_scripts/crossing_reset_test.py`.

## 5. Diagnosis settings (the standardized knobs)
- **epsilon** 0.04, **fault_rate** 0.5 (single value for now), **visibilities** 20/40/60/80/100,
  **num_seeds** 3 — all reused from Empty.
- **Trajectory-length knobs (#8) — RESOLVED (measured with the trained policies, fault_rate 0.5):**
  a fault often blocks the goal so the agent wanders to the cap, making faulted trajectories LONG
  (median ~163 states @noise0.5, ~90 @noise0.7). Set `max_exec_len = 120` for Crossing (rich
  trajectories, safely < MAX_STATES=200; the old 80 truncated most). The 60-step floor dropped ~29%
  of noise-0.7 instances (those where the agent still reached the goal quickly), so
  `MIN_TRAJECTORY_LEN` is lowered to **30** for Crossing (domain-aware in `single_experiment_prepare_inputs_non_determinstic`).

## 6. Fault space & candidate-set construction
- **Reuses Empty's** exactly (decision #10, "lazy" but clean; diversity already comes from the seeds):
  26 injected faults = all `{0,1,2}→{0,1,2}` minus identity; per-instance STATIC hardest-10 candidate
  set = `{T, identity}` + 8 hardest confusers by Hamming distance (seeded 42). Shared constants
  `MINIGRID_FAULTS`, `MINIGRID_CANDIDATE_SETS` in `p_single_experiments.py`.
- **Watch flag:** some (seed layout × fault) combos may be too easy. If results come out too good /
  flat, build a harder seed-based benchmark (profile the policy on the seed's layout → command counts
  → pick a* + candidates). Not now.

## 7. Seeds benchmark — human explanation
- Work-unit = (fault × seed × visibility) = 26×3×5 = **390**; `instance_index = fault_idx*num_seeds +
  seed_idx`, `instance_seed = 10 + instance_index`, passed as `instance_seed*SEED_BLOCK` (the block
  base). For Crossing this block base IS the layout seed: the trajectory env and the diagnoser both
  first-reset from it, so they share ONE layout while the Monte-Carlo noise varies per trace.

## 8. Experiment entry point (called from main)
```
python main.py --minigrid --mg_domain crossing \
    --mg_noise {0.5|0.7} --mg_fault_rate 0.5 --epsilon 0.04 \
    --mg_group $SLURM_ARRAY_TASK_ID --mg_num_groups 390 \
    -o known/minigrid_bench_noise{tag}
```
Driver: `multiple_experiment_MiniGrid_fault_benchmark(..., domain_name="MiniGrid_SimpleCrossing_S11N2_v0")`
(the SAME generalized driver as Empty; `--mg_domain` selects the name).

## 9. Files & functions (this domain's implementation = the per-domain contract)
- `h_wrappers.py` — `_register_minigrid_custom_envs`, `MINIGRID_PER_SEED_LAYOUT`,
  `build_minigrid_view_maps_for_layout`, `MiniGridSetStepWrapper` (per-seed-layout reset),
  `_MINIGRID_ACTIVE_MAPS`, `_MINIGRID_ACTIVE_LAYOUT_SEED`.
- `h_rl_models.py` — `MiniGridPerLayoutTabulatedPolicy`, `_load_minigrid_net`, the per-seed-layout
  branch in `load_trained_model`.
- `h_raw_state_comparators.py` — per-seed-layout view comparator (reads `_MINIGRID_ACTIVE_MAPS`).
- `h_state_refiners.py` — `minigrid_refiner` registered for this domain.
- `p_single_experiments.py` — the generalized driver (domain-aware `max_exec_len`).
- `main.py` — `--mg_domain {empty,crossing}`.
- `sb_minigrid.sbatch` — `MG_DOMAIN={empty|crossing}` (results auto-separate by domain dir).
- Tests: `experiments_scripts/crossing_map_test.py`, `crossing_reset_test.py`.

## 10. Cluster & outputs
- Same sbatch as Empty with `MG_DOMAIN=crossing`. Results land under this domain's own dir:
  `experimental results/MiniGrid_SimpleCrossing_S11N2_v0/{known|unknown}/minigrid_bench_noise{tag}/`
  with `xlsx/` + `logs/` + `plots/`. Plot scripts point at the domain via their `RESULTS_DIR`.

## 11. Results so far
- **Policy (generalization gate — PASSED).** From a 30-run overnight sweep, best held-out policies
  (success on 200 unseen layouts): **noise 0.5 → `long40`** (40M steps): success **1.000**, avg_return
  0.874, avg_steps 67.8; **noise 0.7 → `hi_lr`** (lr 5e-4, ent 0.02, net 128): success **1.000**,
  avg_return 0.929, avg_steps 38.2. Both promoted to `models/PPO/..._noise{0.5,0.7}.zip`. Full
  diagnosis pipeline validated end-to-end (per-layout tabulation; trajectory + diagnoser share one
  layout; ranks produced; no drops).
- **Benchmark:** not run yet — ready to launch (`MG_DOMAIN=crossing sbatch sb_minigrid.sbatch`).
  Fill in avg real-fault rank per noise once run.

## 12. Gotchas / caveats
- **Policy must GENERALIZE** across layouts — a policy that memorized one layout will misnavigate
  others and corrupt the trajectories. This is the release gate.
- **Trajectory floor** (see §5) can silently drop short goal-terminating episodes — resolve before
  trusting a run's instance count.
- Per-seed-layout module state (`_MINIGRID_ACTIVE_*`) is shared across wrapper instances by design;
  it's safe because trajectory + diagnoser use the same `instance_seed` and the benchmark loop is
  sequential per unit. Do not reset a fresh Crossing wrapper with a non-instance seed mid-diagnosis.
- pygame-ce caveat (GIF render) same as Empty.
