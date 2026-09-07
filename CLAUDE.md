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

## Papers & theory mapping (in `references/`)

Two foundational papers (see `references/README.md`):
- **`recent_paper_1.pdf` — "What Went Wrong? Diagnosing Anomalies in RL Policy Executions."**
  Defines the **RLDX** problem and the algorithms **SIF** (breadth-first hypothesis tracking)
  and **SIFU** (conflict-directed: observation *gaps* → *conflicts*, gap **S**orting, fault-mode
  **F**iltering). The `SIFU2…SIFU8` in code are its **ablation variants** (combinations of
  C/S/F). **Crucially, this paper assumes a *deterministic* transition function.**
- **`recent_paper_2.pdf` — "Diagnosing Non-Intermittent Anomalies in RL Policy Executions"**
  (Natan, Stern, Kalech, DX 2024). Previous student's published work; the `W`/`SN` baseline
  diagnosers and the non-intermittent→intermittent framing.

**Where Ahmad's work sits:** he **relaxes paper 1's deterministic-transition assumption** to
**stochastic transitions**, which is why exact symbolic filtering (SIF/SIFU) gives way to his
**Monte-Carlo + likelihood-ranking** diagnosers (`fault_identification_non_deterministic_*`).
Paper 1's listed future work — *ranking methods for diagnoses* and *hidden/unknown fault modes* —
maps directly onto his likelihood ranking and unknown-fault-rate variant.

## Environments

- **FrozenLake** (stochastic) — working.
- **Taxi-v4** (stochastic) — added, **not yet tested**.
- **MiniGrid** (partially observed; branch `minigrid-integration`) — `MiniGrid-Empty-16x16`
  (and `-Empty-Random-6x6`). **The diagnoser sees only the egocentric VIEW** (true partial
  observability). `MiniGridSetStepWrapper.set_state` localizes the observed view to the states
  consistent with it and SAMPLES one `(col,row,dir)` (position AND direction — a view often
  reveals neither; up to 252 states share the empty-centre view), and the comparator compares
  VIEWS — both are the DEFAULTS for MiniGrid domains, no mode flag. Stochasticity via a custom
  `SeededStochasticActionWrapper` (MiniGrid's own `StochasticActionWrapper` draws its coin from
  the GLOBAL numpy RNG → not seed-controlled → do not use it). Policy is a deterministic greedy
  navigator (`MiniGridEmptyHardcodedPolicy`). Perf: `gen_obs` is stubbed + env-checker disabled +
  view maps precomputed (~2.5x). Benchmark: `multiple_experiment_MiniGrid_fault_benchmark`
  (settings mirror FrozenLake: eps 0.04, fr 0.3/0.5/0.8, vis 20-100, 10 candidate faults, only
  the 9 real faults injected), cluster-split by flat work-unit (`--minigrid --mg_group/--mg_num_groups`,
  `sb_minigrid.sbatch`, 1 core/task). Smoke test: `experiments_scripts/minigrid_empty_smoke.py`.
  A first 100-instance cluster run is done (results ~ on par with fully-observed FrozenLake/Taxi).
  Install caveat: `pip install minigrid` pulls `pygame-ce`, which overwrites `pygame`; on Windows
  it's blocked by Application Control (fix = `pip uninstall -y pygame-ce pygame && pip install
  pygame==2.6.1`); on Linux/cluster pygame-ce works fine.
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
- `experiments_scripts/` — Taxi experiment & training scripts (`train_taxi_v4_ppo.py`,
  `eval_taxi_policy.py`, `run_epsilon_sweep.py`, `hard_taxi_benchmark*.py`, `hard_taxi_data.py`,
  `analyze_hard_instance.py`, `train_taxi.sbatch`); run from repo root. `frozen_lake_random_envs*.py`
  — env setup / training.
- `experimental plot scripts/` — **the current, relevant plot pipeline** (sibling of `experimental
  results/`). Holds the scripts that generate every plot under `experimental results/`:
  `plot_epsilon_sweep.py` (6-figure epsilon sweeps), `plot_fault_rate_view.py`,
  `plot_known_vs_unknown.py`, `plot_frozenlake_unknown.py` (FrozenLake unknown-fr per-experiment
  + xlsx merge), `plot_frozenlake_unknown_faultrate.py` (fault-rate view across the unknown
  experiments), `plot_frozenlake_known_vs_unknown.py` (known vs unknown at eps=0.04, by fault
  rate), and the shared helper `plot_provenance.py`. Run from repo root, e.g.
  `./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_epsilon_sweep.py"`. They import
  each other by bare module name (resolved via the running script's own dir), so keep them in
  one folder (one level under the repo root, so their `repo_root` math holds).
  - **STANDING RULE — plot provenance:** every script that writes plots must, after saving
    them, drop a `PLOT_PROVENANCE.txt` into each plots folder recording the producing script's
    full path (+ timestamp, plot list, input xlsx folders). Use the shared helper
    `from plot_provenance import write_plot_provenance` — call
    `write_plot_provenance(plots_dir, created_files, input_sources=[...])` once. Do NOT hand-roll
    a per-script copy.
- `scripts/` — older / one-off analysis & plotting (`explore_experiemnts.py`,
  `plot_experiments.py`, `fault_rate_comparsion.py`, `analyze_sweep.py`, …); legacy, not behind
  the current `experimental results/` plots.

## Running

- `main.py` selects which experiment runs by **commenting/uncommenting** calls in the
  `if __name__ == '__main__'` block (e.g. `single_experiment_stochastic_FrozenLake()` vs
  `single_experiment_stochastic_Taxi_v4()`).
- CLI flags: `--epsilon` (adaptive MC confidence threshold), `-ufr/--unknown_fault_rate`,
  `-n/--maps_num`. Note: `main.py` currently also hard-sets these after parsing.
- Experiment input files live in `experimental inputs/` (e.g. `e5000_Taxi-1.json`).
- **Environment — ALWAYS Python 3.11:**
  - Local: use **`.venv_domains`** + `requirements.txt` (py3.11). The old `.venv` +
    `requirements_py38_backup.txt` is the previous student's py3.8 setup — do not use.
  - Cluster: use conda env **`rldx_py311`**. `rldx_conda` is the old py3.8 env. ⚠️ The existing
    `~/rldx.sbatch` still activates `rldx_conda` and must be switched to `rldx_py311`.
  - ⚠️ The py3.8→3.11 migration (done to enable Taxi-v4 / FrozenLake) **may be imperfect** —
    watch for lingering dependency conflicts / errors. See [[env-python311-and-migration-caveat]].
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
- **Proactive Claude Code guidance:** Ahmad is new to Claude Code and will miss setup/usage
  opportunities. Whenever something would help — running `/init`, a project `.claude/` folder,
  a new skill, a permission rule, a useful slash command, a keyboard shortcut, or any essential
  usage tip — **proactively point it out**, briefly explain what it does, and let him decide.
  Don't assume he already knows.

## Cluster / HPC (in scope)

- BGU CIS Slurm cluster. Connect via terminal SSH: `ssh <bgu_username>@slurm.bgu.ac.il`
  (on BGU-WIFI/campus, or VPN otherwise). Manager node = submit only; compute nodes run jobs.
  Passwordless key auth is configured as `ssh bgu`.
- **⚠️ HANDS-OFF:** never touch `~/EOM/` or **any GPU job** (not ours) — only act on our
  CPU/`main`-partition rldx jobs. See [[cluster-hands-off-boundaries]] and `references/CLUSTER.md`.
- **This project is CPU-bound** (diagnosers + Monte-Carlo) — submit CPU jobs, no GPU.
- Use **job arrays** for experiment sweeps (epsilons/fault-rates/seeds/maps), not thousands of
  tiny jobs; write results to `/scratch` and copy back. Full tailored guide + sbatch template:
  **`references/CLUSTER.md`** (source: `references/cluster_user_guide.pdf`).
- "Cluster-ready" enabler: parametrize `main.py` by `SLURM_ARRAY_TASK_ID` (ties to the
  `main.py:90-92` inert-CLI gotcha and the hardcoded single map).
