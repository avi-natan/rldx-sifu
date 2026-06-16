# Cluster usage — conclusions for rldx-sifu

Distilled from `references/cluster_user_guide.pdf` (BGU CIS HPC / Slurm, 2024), tailored to
this project. Italic = Slurm CLI. This is the **manager node = launch only, compute nodes =
run** model: you SSH to the login node and submit jobs that Slurm schedules onto compute nodes.

## ⚠️ Hands-off boundaries (NEVER touch)
- **`~/EOM/` directory** — never read, write, move, or delete anything inside it.
- **Any GPU job** (partition `gpu`, e.g. jobs named `my_job`) — **not ours.** Expect them to be
  there; never `scancel`/modify them. Only ever act on our **CPU / `main`-partition** rldx jobs
  (`rldx_job`) under `~/rldx_repo/rldx-sifu`. When canceling, target specific job ids — never blanket-cancel.

## Current cluster state (observed 2026-06-17, via key-based `ssh bgu`)
Much is already set up — this is **not** a blank slate:
- **Login:** `ahmade@slurm.bgu.ac.il` (lands on `slurm-login-02`), home `/home/ahmade`.
  **Passwordless key auth is configured** (alias `bgu` in `~/.ssh/config` → key `~/.ssh/id_ed25519`).
- **Repo cloned:** `~/rldx_repo/rldx-sifu`, currently on branch **`new-master` @ `a1cbad7`** —
  **behind** Ahmad's local `new-master-ai`. To run the latest: `git fetch` + `git checkout new-master-ai`.
- **Conda env — use `rldx_py311`** (Python 3.11, canonical for current work). `rldx_conda` is the
  OLD py3.8 env (previous student's setup); `eom+` is EOM work (not ours). ⚠️ The existing
  `~/rldx.sbatch` still does `source activate rldx_conda` (py3.8) and **must be switched to
  `rldx_py311`**. Note: the py3.8→3.11 migration may be imperfect — watch for dependency errors.
- **Existing `~/rldx.sbatch`:** `--partition main`, `--time 2-00:00:00`, `--gpus=0`,
  `module load anaconda; source activate rldx_conda; cd ~/rldx_repo/rldx-sifu; python main.py "$@"`.
  Already contains **commented job-array scaffolding** for an epsilon sweep
  (`EPSILONS=(0.1 0.2 0.5 0.75 1); EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}`).
- `~/example.sbatch` (the cluster's template) is also present.

**Immediate gaps:** (1) cluster repo is behind `new-master-ai`; (2) `~/rldx.sbatch` activates the
old py3.8 `rldx_conda` — switch to `rldx_py311`; (3) `main.py` doesn't yet read
`SLURM_ARRAY_TASK_ID`, so the array scaffolding is inert until `main.py` is parametrized (ties to
the `main.py:90-92` inert-CLI gotcha).

## How Ahmad actually runs this project (from `~/.bash_history`)
- **Sync repo:** `cd ~/rldx_repo/rldx-sifu && git reset --hard && git fetch && git pull`.
- **One shared `~/rldx.sbatch`, edited in place** with `nano` to fit each run (NOT a separate
  sbatch per experiment). Same habit for `train_pong.sbatch`. So the sbatch on the cluster reflects
  the *last* run's config — don't assume it's a clean template.
- **Epsilon "sweep" = manual repeated submits:** `sbatch rldx.sbatch --epsilon 0.04` … `1.0`
  (values seen: 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0).
- **Monitor:** `squeue --me`, `tail`/`cat` the `rldx_job-<id>.out`. **Cancel:** `scancel <id>`.
- ⚠️ **Past epsilon sweeps were probably inert:** `--epsilon X` → `python main.py "$@"`, but
  `main.py:90` overwrites `args.epsilon = 0.03` and the active `single_experiment_stochastic_*`
  hardcode epsilon internally → those runs likely all used a fixed epsilon, not the swept value.
  **Verify old `.out` files before trusting epsilon comparisons.** Strongest motivation to make
  `main.py` truly arg-driven (and job-array ready).

## 0. Connecting (replacing MobaXterm with terminal SSH)
- **Prereq network:** be on **BGU campus / BGU-WIFI**, OR any network **+ BGU VPN**. (FAQ: "Can I
  ssh when away? — use VPN.")
- **Login node:** `slurm.bgu.ac.il`, port 22, **BGU username + password**.
- **Terminal SSH** (Windows PowerShell/Git-Bash both have OpenSSH):
  ```bash
  ssh <bgu_username>@slurm.bgu.ac.il
  ```
- Optional convenience — add to `~/.ssh/config` so you can just `ssh bgu`:
  ```
  Host bgu
      HostName slurm.bgu.ac.il
      User <bgu_username>
  ```
- **NEVER compute on the manager node.** It's only for submitting/monitoring.
- Admission to the cluster must be granted by IT first.

## 1. This project is CPU-bound — skip the GPU
The diagnosers (SIF/SIFU and the stochastic Monte-Carlo identifiers) and experiment drivers are
**CPU work**. Submit to the **CPU cluster** (no `--gpus`). GPU is only relevant if you retrain a
policy (`train_taxi_v4_ppo.py` PPO) — even then PPO on these toy envs is small. Default: **no GPU**.

## 2. First-time setup (conda env on the manager node)
Anaconda is preinstalled (do **not** install it). Create the project env once on the login node:
```bash
module load anaconda
conda create -n rldx python=3.11
conda activate rldx
pip install -r requirements.txt
conda deactivate
```
All nodes share your `/home` + `/storage`, so files uploaded to the login node are visible on
compute nodes.

## 3. Submitting a job (sbatch) — template for this repo
Copy the cluster's `/storage/example.sbatch` as a base, or use this CPU template:
```bash
#!/bin/bash
#SBATCH --partition main                 ### use 'main' (no QoS)
#SBATCH --time 0-08:00:00                ### D-H:MM:SS, < 7 day partition cap
#SBATCH --job-name rldx
#SBATCH --output rldx-%J.out             ### %J = job id
#SBATCH --mail-user=<you>@post.bgu.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=6                ### CPU job: ok to request a few cores

echo "JOBID=$SLURM_JOBID NODE=$SLURM_JOB_NODELIST"
module load anaconda
source activate rldx
python -u main.py                        ### -u = unbuffered, see logs live
```
Submit with the env **deactivated** in your shell: `sbatch rldx.sbatch`.

## 4. Job arrays — the right tool for your experiment sweeps
You run many near-identical experiments (epsilon sweeps, fault rates, seeds, the 100 FrozenLake
maps). **Do NOT fire thousands of tiny `sbatch` calls** (scheduler + I/O "silent killer"). Use a
**job array** instead — one queue entry, parallel tasks distinguished by `$SLURM_ARRAY_TASK_ID`:
```bash
#SBATCH --array=0-48%10      ### 49 tasks (e.g. one per map), max 10 running at once
...
python -u main.py $SLURM_ARRAY_TASK_ID
#SBATCH --output rldx-%A_%a.out   ### %A = array job id, %a = task id
```
Then in Python: `task = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))` to pick the map/epsilon/seed.
> Practical fit: `main.py` currently selects experiments by commenting/uncommenting and
> hard-codes params (`main.py:90-92`, single map `loaded[1]`). To use arrays well we'd wire
> `SLURM_ARRAY_TASK_ID` → an experiment/param index. Good first "cluster-ready" refactor.

## 5. I/O: write results to local SSD, copy back at the end
Your experiments write Excel/CSV. Many tasks writing to shared storage at once can hang the
metadata server. Best practice: write to the compute node's **`/scratch`**, copy final results
back to `/storage` (or `/home`) at the end:
```bash
#SBATCH --tmp=10G
export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
# ... run code writing into $SLURM_SCRATCH_DIR ...
cp -r $SLURM_SCRATCH_DIR/results $SLURM_SUBMIT_DIR/
```
**`/scratch` is wiped when the job ends** — always copy back.

## 6. Monitoring & control
- *squeue --me* — my jobs (`ST`: `PD` pending, `R` running)
- *less rldx-<jobid>.out* — watch output
- *scancel <jobid>* / *scancel --name rldx* — cancel; *scancel -t PENDING -u <user>* — all pending
- *sacct -j <jobid> --format=JobName,MaxRSS,AllocTRES,State,Elapsed,ExitCode* — post-mortem (RAM etc.)
- *sinfo -Nel* — node info; *sres* — current resource availability
- Pending `REASON`: `Resources` (cluster full), `Priority` (queued behind others),
  `QOSMaxJobsPerUserLimit` (too many concurrent jobs), `PartitionTimeLimit` (time > 7 days).

## 7. Interactive use (for debugging on a compute node)
```bash
sinteractive --time 0-2:00:00        ### CPU interactive job (no --gpu for us)
# wait for allocation, note the hostname + jobid, then:
srun --jobid=<jobid> --pty bash      ### attach to the job's env
# ... debug: conda activate rldx; python main.py ...
scancel <jobid>                      ### ALWAYS release when done
```
Keep the SSH session open for the duration; closing it ends interactive work. Useful for IDEs:
the guide covers **PyCharm Pro** and **VS Code (Remote-SSH)** pointed at
`~/.conda/envs/rldx/bin/python` on the **compute** node (never the manager node).

## 8. Moving code/results to & from the cluster
- **Git** (recommended for this repo): git is on the manager node. Set up a GitHub SSH key
  (`ssh-keygen -t ed25519`; add `~/.ssh/id_ed25519.pub` to GitHub), then clone/pull `rldx-sifu`
  there. Our branch `new-master-ai` is already pushed, so `git clone` + `git checkout new-master-ai`.
- **Files:** WinSCP (GUI) or `scp`/`rsync` from the terminal, e.g.
  `scp -r results <user>@slurm.bgu.ac.il:/home/<user>/rldx-sifu/`.

## 9. Etiquette / limits (shared resource)
- Use **minimum** RAM (default 24G/GPU; override `--mem=48G` only if needed; >58G → ask IT).
- Release idle resources (*scancel*) even on breaks; delete unused files.
- Don't allocate more than you need; one GPU per job max (N/A for us).

## 10. Project-specific TODOs to become "cluster-ready"
1. **Parametrize `main.py`** by an index/args so a **job array** can sweep maps/epsilons/seeds
   (replace the comment-toggling + `main.py:90-92` hard-overrides; make the parsed CLI flags live).
2. **Route experiment outputs** to `$SLURM_SCRATCH_DIR` then copy back (avoid shared-FS thrash).
3. Recreate the **`rldx` conda env (py3.11)** from `requirements.txt` on the cluster.
4. Pull `new-master-ai` on the cluster via Git SSH.
