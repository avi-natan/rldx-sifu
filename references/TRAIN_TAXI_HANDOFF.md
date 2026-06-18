# Handoff brief — train a better Taxi-v4 PPO policy (cluster)

> **Paste-to-start for a fresh Claude Code session.** `CLAUDE.md` and the user's memory auto-load,
> so the project domain is already known. This brief adds (1) branch/commit rules, (2) that a
> **parallel Claude session is actively editing this repo right now**, (3) full cluster connection
> + structure, and (4) the complete task checklist. Read `references/CLUSTER.md` for cluster depth
> and `references/gym_envs.md` for the Taxi env spec.

---

## 0. Your goal
Train a **genuinely better Taxi-v4 PPO policy** for the **stochastic (rainy) env**, evaluate it
honestly, and — *only after coordinating* (see §3) — make it the policy the code loads. A stronger
policy yields cleaner faulty trajectories and fewer undiagnosable instances downstream.

---

## 1. Repo & branch model (know this cold)
- **Remote:** `origin` = `github.com/avi-natan/rldx-sifu`.
- **Branches:**
  - `master` — previous student's published work. **NEVER touch, never commit, never push.**
  - `new-master` — Ahmad's baseline. **Don't modify.**
  - `new-master-ai` — **THE active working branch. You work here. So does the other session.**
  - (merge-base of `master`/`new-master` is `59e9c563`; to see only Ahmad's work:
    `git diff master..new-master-ai`.)
- **You and the other Claude session share the SAME local working tree** (same machine, same
  `.git`, same checkout of `new-master-ai`). There is **no push/pull needed between the two
  sessions** — commits from one are immediately in the other's history. Coordination is purely
  about **not editing the same file at the same time** (see §3).
- **Commit rules:**
  - Commit **frequently** at logical checkpoints with clear messages (a parallel session means an
    uncommitted edit can be clobbered — commit early so nothing is lost).
  - End every commit message with:
    `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  - **Pushing to `origin/new-master-ai` is OK** (Ahmad authorized it) — and you'll *need* it to get
    code onto the cluster. **Never push `master` or `new-master`.**
  - As of this brief, local `new-master-ai` == `origin/new-master-ai` (just pushed, commit
    `36706a1f`). Run `git log --oneline -5` at start to see the latest.

---

## 2. The training task — current state & the script
### Models (`environments/Taxi_v4/models/PPO/`)
- **`Taxi_v4__PPO.zip`** (Dec 2025, 906 KB) — the **GOOD** model, ~**11/20** seeds solved. This is
  the file the loader actually reads. Trained for *deterministic* Taxi but currently our best.
- **`Taxi_v4_PPO_rainy_0.7_steps_1000000_seed_42.zip`** (Jun 13, 905 KB) — freshly trained on the
  **rainy** env for 1M steps, but **BROKEN: 0/20 solved.** So the current recipe is *worse* than
  the old model. Understanding why is the core of this task. **Keep both `.zip`s** until a proven
  winner exists.

### `train_taxi_v4_ppo.py`
- Env: `gym.make("Taxi-v4", is_rainy=True, rainy_probability=0.7, fickle_passenger=False)` + `Monitor`.
- `PPO("MlpPolicy", env, verbose=1, seed=seed)` — **all default hyperparameters**, only seed set.
  No eval callback, no exploration tuning, no vec envs. Most likely why 1M steps still gives 0/20.
- **Gotcha (line 29):** `args.timesteps = 1000000` hard-overrides the `--timesteps` CLI flag (inert
  flag — same pattern as `main.py:90`). Fix if you want CLI control.
- **Save-path vs loader mismatch:** it saves to
  `…/Taxi_v4_PPO_rainy_{p}_steps_{t}_seed_{s}.zip`, but the loader reads **`Taxi_v4__PPO.zip`**
  (`h_rl_models.load_trained_model` / `build_taxi_hardcoded_policy`, path
  `f"{models_dir}/{domain_name}__{ml_model_name}.zip"`). To put a new model into service you must
  **copy it to `Taxi_v4__PPO.zip`** — but see §3 first; that step is gated.

### Measuring quality
Roll out the policy on N seeds (e.g. 20) in the rainy env, deterministic actions, count deliveries
(episode ends with the dropoff reward) within the step cap → **success@N**. That's the 11/20 vs
0/20 metric. Build a small standalone eval; don't trust training reward curves alone.

### Levers to try
`ent_coef` (default 0.0 starves exploration — raise it), `learning_rate`, `n_steps`, `batch_size`,
`n_epochs`, `gamma`; train longer; **vectorized envs** (`make_vec_env`, parallel copies);
**`EvalCallback`** to checkpoint the *best* model by success rate; consider a **curriculum** on
`rainy_probability` (learn deterministic/low-rain first, anneal to 0.7). Verify the training env
matches the diagnoser's simulator (`make_wrapped_env`, `h_wrappers.py`).

---

## 3. ⚠️ Coordination with the active parallel session (READ THIS)
Another Claude session is **right now** building/running the hard Taxi-v4 diagnosis benchmark.

**Files that session OWNS — do NOT edit:**
`hard_taxi_benchmark.py`, `hard_taxi_data.py`, `p_pipeline.py`, `p_single_experiments.py`,
`run_hard_taxi_benchmark.py`, `references/HARD_TAXI_SPEC.md`, `references/RESEARCH_STATUS.md`.

**Files that are YOURS:** `train_taxi_v4_ppo.py`, a new eval script (e.g. `eval_taxi_policy.py`),
your sbatch, the model `.zip`s.

**`h_rl_models.py` is shared/sensitive** — the other session added `TaxiHardcodedPolicy` + the
loader branch there. Prefer **not** to edit it. To put a new model into service, **copy the winning
zip to `Taxi_v4__PPO.zip`** rather than changing the loader path.

**🔴 The big dependency — policy promotion is GATED:**
The hard benchmark (its frozen `COUNTS_TABLE`, execution-fault picks, and candidate sets in
`hard_taxi_data.py`) was generated **against the current `Taxi_v4__PPO.zip`**. If you overwrite that
file with a new policy, **the entire frozen benchmark becomes stale** (different policy → different
trajectories → different action counts → different valid faults). So:
- **Do NOT overwrite `Taxi_v4__PPO.zip` unilaterally.** Train, evaluate, and save your candidate
  under its descriptive name. Surface the result to Ahmad. Promotion + benchmark regeneration is a
  **coordinated** step between both sessions.

---

## 4. Cluster — connection & structure (BGU CIS Slurm)
Full guide: `references/CLUSTER.md`. Essentials:

### Connect
- Network: BGU campus / BGU-WIFI, **or any network + BGU VPN** (off-campus → VPN first).
- `ssh bgu` (alias preconfigured: `slurm.bgu.ac.il`, user `ahmade`, key auth). Lands on
  `slurm-login-02`, home `/home/ahmade`. **Login node = submit/monitor ONLY; never compute on it.**

### Structure / current state (observed 2026-06-17)
- **Repo on cluster:** `~/rldx_repo/rldx-sifu`, was on `new-master` (behind). To get latest:
  `cd ~/rldx_repo/rldx-sifu && git fetch && git checkout new-master-ai && git pull`.
- **Conda env: `rldx_py311`** (Python 3.11 — canonical). **NOT** `rldx_conda` (old py3.8), **NOT**
  `eom+` (EOM, not ours).
- **`~/rldx.sbatch`** exists but is edited **in place** per run and currently does
  `source activate rldx_conda` → **must switch to `rldx_py311`**. It targets `--partition main`,
  `--gpus=0`. `~/example.sbatch` is the cluster's template.
- Shared `/home` + `/storage` across nodes; per-job `/scratch` is wiped at job end (copy results
  back).

### Running a training job (CPU)
- **This is CPU work** — Taxi PPO is small. Request a CPU job on `--partition main`, **no `--gpus`**.
- Make/adapt a training sbatch that: `module load anaconda; source activate rldx_py311;
  cd ~/rldx_repo/rldx-sifu; python -u train_taxi_v4_ppo.py …`. Write the model to `/scratch`, copy
  the `.zip` back to the repo (or `/storage`) at the end.
- Submit with env deactivated: `sbatch <your>.sbatch`. Monitor: `squeue --me`,
  `less rldx-<jobid>.out`. Cancel a specific job: `scancel <jobid>`.

### ⚠️ Cluster hands-off boundaries (NEVER touch)
- **`~/EOM/`** — never read/write/move/delete anything inside it.
- **Any GPU job** (partition `gpu`, e.g. jobs named `my_job`) — **not ours.** Expect them to appear;
  never `scancel`/modify them. Target **specific** job ids; never blanket-cancel.

---

## 5. Everything that needs to be done (checklist)
1. **Reproduce the gap locally** — write `eval_taxi_policy.py` (success@20, rainy, deterministic);
   confirm old `Taxi_v4__PPO.zip` ≈ 11/20 and the rainy zip ≈ 0/20.
2. **Diagnose the failure** — inspect training (reward curve, episode length); hypothesize
   (exploration starvation, reward sparsity under rain, too-hard `rainy_probability`).
3. **Improve `train_taxi_v4_ppo.py`** — remove the `timesteps` override; add `EvalCallback` +
   best-model checkpointing; tune hyperparameters; optionally vec envs / curriculum. Keep CLI flags
   live.
4. **Local smoke train** (short) to confirm the pipeline + eval work end-to-end before cluster.
5. **Cluster setup** — VPN/`ssh bgu`; sync repo to `new-master-ai`; confirm `rldx_py311`; write the
   training sbatch (CPU, `rldx_py311`, `/scratch` + copy-back). (Optionally fix `~/rldx.sbatch`'s
   `rldx_conda`→`rldx_py311` while there.)
6. **Submit the real training run**; monitor; copy the resulting `.zip` back; commit it under its
   descriptive name (do **not** overwrite `Taxi_v4__PPO.zip` — see §3).
7. **Evaluate the new model** (success@N). If clearly better, **surface to Ahmad** for the gated
   promotion + benchmark-regeneration step.
8. **Commit + push** to `origin/new-master-ai` at each checkpoint. If a finding is worth persisting
   to `CLAUDE.md`, **ask Ahmad first** (project rule).

---

## 6. Quick rules recap
- Branch `new-master-ai` only; never `master`/`new-master`. Commit often; push `origin/new-master-ai`
  OK. Co-Author trailer on commits.
- Python **3.11** everywhere (`.venv_domains` local / `rldx_py311` cluster).
- Don't edit the parallel session's files (§3). Don't overwrite `Taxi_v4__PPO.zip` (§3).
- Cluster: CPU only, `rldx_py311`, never `~/EOM/` or GPU jobs.
