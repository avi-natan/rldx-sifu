# Handoff brief — train a better Taxi-v4 PPO policy (cluster)

> Paste-to-start for a fresh Claude Code session. `CLAUDE.md` and memory auto-load, so this
> brief only covers the **task specifics**. A parallel session is working the hard-Taxi
> benchmark — see "Stay in your lane" below.

## Goal
Train a **genuinely better Taxi-v4 PPO policy** (stochastic / rainy env) and make it the one
the code actually uses. A stronger policy → cleaner faulty trajectories → fewer undiagnosable
("no fault fired") benchmark instances downstream.

## Current state of the models (`environments/Taxi_v4/models/PPO/`)
- **`Taxi_v4__PPO.zip`** (Dec 2025, 906 KB) — the **GOOD** model, ~**11/20** seeds solved.
  This is the file the loader reads (see below). Trained for the *deterministic* Taxi, but it's
  the best we currently have.
- **`Taxi_v4_PPO_rainy_0.7_steps_1000000_seed_42.zip`** (Jun 13, 905 KB) — freshly trained on
  the **rainy** env for 1M steps, but **BROKEN: 0/20 solved**. So the current training recipe
  produces a *worse* policy than the old deterministic one. Figuring out why is the heart of
  this task. (Do not delete either; keep both until a better one is proven.)

## The training script — `train_taxi_v4_ppo.py`
- Env: `gym.make("Taxi-v4", is_rainy=True, rainy_probability=0.7, fickle_passenger=False)`,
  wrapped in `Monitor`.
- `PPO("MlpPolicy", env, verbose=1, seed=seed)` — **all default hyperparameters**, only the seed
  is set. No eval callback, no entropy/lr tuning, no vectorized envs. This is the most likely
  reason 1M steps still gives 0/20: defaults + sparse Taxi reward + rain.
- **Gotcha (line 29):** `args.timesteps = 1000000` hard-overrides the `--timesteps` CLI flag, so
  the flag is currently inert (mirror of the `main.py:90` override). Fix this if you want CLI
  control of training length.
- **Save-path gotcha:** it saves to
  `environments/Taxi_v4/models/PPO/Taxi_v4_PPO_rainy_{p}_steps_{t}_seed_{s}.zip`, but the
  **loader reads `Taxi_v4__PPO.zip`** (see `h_rl_models.load_trained_model` /
  `build_taxi_hardcoded_policy`, path `f"{models_dir}/{domain_name}__{ml_model_name}.zip"`).
  So after you train a winner, **copy it to `Taxi_v4__PPO.zip`** (or update the loader) for it to
  take effect.

## How "policy quality" is measured
Roll out the policy on N seeds (e.g. 20) in the rainy env, deterministic actions, count how many
deliver the passenger (episode terminates with the dropoff reward) within the step cap.
That's the 11/20 vs 0/20 metric. Build a small eval script (or reuse one) and report success@N;
don't trust training reward curves alone for Taxi.

## Ideas to actually get a better policy (pick/iterate)
- **Tune PPO hyperparameters:** `n_steps`, `batch_size`, `learning_rate`, `ent_coef` (Taxi needs
  exploration — raise it), `gamma`, `n_epochs`. Default `ent_coef=0.0` often starves exploration.
- **Train longer and/or with vectorized envs** (`make_vec_env`, several parallel copies).
- **Add `EvalCallback`** (SB3) on a held-out env to checkpoint the best model by success rate,
  not the last one.
- Consider whether **rainy_probability=0.7 is too hard** to learn from scratch — a curriculum
  (start lower, anneal up) or first training deterministic then fine-tuning rainy may help.
- Sanity-check the env wrapper matches what the diagnoser's simulator uses (`make_wrapped_env`,
  `h_wrappers.py`) so a model good in training is good at diagnosis time.

## Cluster (BGU CIS Slurm) — see `references/CLUSTER.md` for the full guide + sbatch template
- Connect: `ssh bgu` (needs BGU-WIFI/campus or **VPN**). Manager node = submit only.
- **Python env: `rldx_py311`** (conda). NOT `rldx_conda` (old py3.8). ⚠️ `~/rldx.sbatch` still
  activates `rldx_conda` — switch it to `rldx_py311` before submitting.
- **This is CPU work.** Taxi PPO is small; request a CPU job on the `main` partition. No GPU.
- Sync the repo to `new-master-ai` on the cluster before running. Write outputs to `/scratch`,
  copy the resulting `.zip` back.
- **⚠️ HANDS-OFF:** never touch `~/EOM/` or **any GPU job** (not ours — they appear sometimes).
  Only act on our CPU `main`-partition rldx jobs.

## Stay in your lane (parallel session active)
Another session is editing the hard-benchmark code: `hard_taxi_benchmark.py`, `hard_taxi_data.py`,
`p_pipeline.py`, `p_single_experiments.py`, `run_hard_taxi_benchmark.py`. **Don't edit those.**
Your files: `train_taxi_v4_ppo.py`, a new eval script, the sbatch, and the model `.zip`s.

## Working rules (from CLAUDE.md)
- Branch `new-master-ai`; never modify `master`/`new-master`. Push to `origin/new-master-ai` is OK.
- Commit frequently at logical checkpoints so nothing is lost.
- Always Python 3.11 (`.venv_domains` local / `rldx_py311` cluster).

---
**First step suggestion:** read `train_taxi_v4_ppo.py` and `references/CLUSTER.md`, build a quick
success@20 eval to reproduce the 0/20 vs 11/20 gap locally, then design the improved training run
to submit on the cluster.
