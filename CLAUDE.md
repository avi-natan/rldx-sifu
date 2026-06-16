# CLAUDE.md — rldx-sifu

Project context for Claude Code. Loaded automatically each session.

## Project domain

Reinforcement-learning **fault diagnosis**. When a policy runs in the real world, a fault can
corrupt action execution (e.g. a corrupt wheel makes `RIGHT` execute as `FORWARD`). We don't
know which fault occurred and want to **detect it**.

**Problem setup:** given
- a trajectory `s1, a1, s2, a2, …, sn, an`,
- a list of candidate **fault modes**,
- a **simulator** and a **deterministic policy**,

infer the **true fault** that produced the trajectory.

**Algorithms:** `SIF` and `SIFU` (the diagnosers). Implemented in `p_diagnosers.py`.

## Assumption progression (where the research is)

- **Previous student (on `master`):** deterministic policy, **deterministic** action effects,
  partial observability (large hidden % of trajectory → gaps), non-intermittent faults → later
  intermittent faults. Paper: *"Diagnosing Non-Intermittent Anomalies in RL Policy Executions"*,
  Natan, Stern, Kalech — DX 2024.
- **Ahmad (current work):** deterministic policy still, but **stochastic action effects
  (stochastic env)**. To make the transition tractable he peeled assumptions back, then
  re-adds them one at a time. Currently: **fully observed**, **non-intermittent**,
  **known fault rate**. A **unknown-fault-rate** variant of the algorithm was recently added
  but not yet fully adopted.

## Environments

- **FrozenLake** (stochastic) — working.
- **Taxi-v4** (stochastic) — added, **not yet tested**.
- **CliffWalking** — planned next.
- Previous work used deterministic Gymnasium envs (Acrobot, CartPole, MountainCar, Taxi,
  LunarLander) — see the many `single_experiment_<Env>_<W|SN|SIF|SIFU…>` functions.

## Branch model (important)

- `master` — **previous student's** work. **Do not touch.**
- `new-master` — Ahmad's baseline (his "master").
- `new-master-ai` — **active AI-assisted branch** (work happens here).
- Fork point (merge-base) between `master` and `new-master`: `59e9c563`.
- To see only Ahmad's work: `git log master..new-master` / `git diff master..new-master`.

## Where the real code is (Ahmad's changes vs `master`)

Weight these heavily; treat `master`-only code as background.

- `p_diagnosers.py` — diagnosis algorithms (**SIF / SIFU**), incl. unknown-fault-rate support.
- `p_single_experiments.py` — experiment drivers, incl. `single_experiment_stochastic_FrozenLake`,
  `single_experiment_stochastic_Taxi_v4`.
- `p_pipeline.py` — pipeline orchestration (`run_experimental_setup_new`).
- `p_executor.py` — execution.
- `h_wrappers.py`, `h_rl_models.py`, `h_state_refiners.py`, `h_raw_state_comparators.py` —
  env / fault / state plumbing.
- `main.py` — entry point.
- `train_taxi_v4_ppo.py`, `frozen_lake_random_envs*.py` — env setup / training.
- `scripts/` — analysis & plotting (`explore_experiemnts.py`, `plot_experiments.py`,
  `fault_rate_comparsion.py`).

## Running

- `main.py` selects which experiment runs by **commenting/uncommenting** calls in the
  `if __name__ == '__main__'` block (e.g. `single_experiment_stochastic_FrozenLake()` vs
  `single_experiment_stochastic_Taxi_v4()`).
- CLI flags: `--epsilon` (adaptive MC confidence threshold), `-ufr/--unknown_fault_rate`,
  `-n/--maps_num`. Note: `main.py` currently also hard-sets these after parsing.
- Experiment input files live in `experimental inputs/` (e.g. `e5000_Taxi-1.json`).
- **Environment:** Python **3.11** (`requirements.txt`). `requirements_py38_backup.txt` keeps
  the previous student's Python 3.8 setup.
- `README.md` is the **previous student's** (Python 3.8.7, his paper) — do not assume it
  describes Ahmad's current setup; don't rewrite it unless asked.

## Code scan findings (2026-06-16) — see `CODE_SCAN.md` for the full conclusion

Key durable facts (full detail, line refs, and issue table live in **`CODE_SCAN.md`** at repo root):
- **Two diagnoser families** in `p_diagnosers.py`: legacy *deterministic* (`W, SN, SIF, SIFU…SIFU8`,
  exact/symbolic) vs Ahmad's *stochastic* (`fault_identification_non_deterministic_FO / PO /
  PO_unknown_fault_rate`, adaptive Monte-Carlo, likelihood ranking). Work on the stochastic ones.
- The stochastic diagnosers **do** pass env stochastic kwargs (slippery/rainy) via `make_wrapped_env`;
  only the legacy `SIFU*/W/SN` build bare envs (latent trap if reused on stochastic domains).
- **Taxi-v4 gotcha (likely why it's "untested"):** the loader uses
  `environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip` (old), but the freshly-trained
  `Taxi_v4_PPO_rainy_0.7_steps_1000000_seed_42` has **no `.zip`** and is never loaded.
- **`main.py:90-92` gotcha:** parses `--epsilon/-ufr/-n` then hard-overwrites them, so the CLI flags
  are currently inert; experiment selection is by commenting/uncommenting calls.
- Stochastic drivers currently hardcode a single map (`loaded[1]`), `fault_rate=0.5`, `epsilon=0.03`,
  Taxi `rainy_probability=0.7`.

## Working rules for Claude

- **Branch:** work on `new-master-ai`; never modify `master`.
- **Commits:** make frequent checkpoint commits at logical stopping points with clear
  messages so changes are revertible. **Pushing to `origin/new-master-ai` is OK** (Ahmad
  authorized it); **never push to `master` or `new-master`.**
- **Persistence:** when a code decision/finding seems worth persisting, **ask Ahmad** before
  writing it into this file.
- **References:** papers and cluster docs go in `references/` — read them on request.

## Parked / out of scope (for now)

- Cluster / HPC setup & usage — later (docs will go in `references/`).
