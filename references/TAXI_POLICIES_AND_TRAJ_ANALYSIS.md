# Taxi-v4 — Policies, Evaluation & Trajectory Analysis (handoff)

Produced 2026-06-20 by the policy-training session. All work is **local in `runs/`**, nothing
promoted, the served `Taxi_v4__PPO.zip` is **untouched**. Python env: `.venv_domains` (py3.11);
run scripts with `.venv_domains/Scripts/python.exe`.

Action codes: `0`=South/**DOWN**, `1`=North/**UP**, `2`=East/**RIGHT**, `3`=West/**LEFT**,
`4`=Pickup, `5`=Dropoff. `rainy_probability` = P(intended action executes); **lower = more
stochastic** (1.0 = deterministic). The diagnoser/benchmark default is rainy **0.7**.

---

## 0. IMPORTANT CORRECTION — the "policy-quality gap" was a measurement artifact

The handoff brief's premise ("old `Taxi_v4__PPO.zip` ≈ 11/20 / ~55%, needs replacing") **does not
reproduce.** Re-measured with the clean deterministic `eval_taxi_policy.py` harness (and validated
that the harness discriminates — the known-bad 1M model scores 0/20 through it):

| policy | rainy 0.7 | rainy 0.5 |
|---|---|---|
| **served `Taxi_v4__PPO.zip`** (trained on *deterministic* env) | 98/100 (20/20 on seeds 0–19) | 94/100 |
| `Taxi_v4_PPO_rainy_0.7_steps_1000000_seed_42.zip` (the **collapsed** 1M model) | 0/20 | 0/20 |

So the *served* policy was always strong; the genuinely-bad one is the **1M rainy model** (still
sitting in the models dir — make sure nothing points at it). A deterministic-trained Taxi policy
generalizes to rain because Taxi is closed-loop/forgiving (a slip just costs an extra step and the
policy re-decides each step).

**Implication:** the 3 new policies below only *tie* the served policy — they were solving a gap
that doesn't really exist. They're fine as backups / for matched-stochasticity experiments, but
there's no strong reason to promote any of them.

---

## 1. The three new policies (matched to stochasticity level)

| rainy level | path (repo-relative) | success | avg traj length (successful) |
|---|---|---|---|
| **0.7** | `runs/taxi_v4/shaped_rainy_s2/final_model.zip` | ~96% | **20.6** steps |
| **0.5** | `runs/taxi_v4/shaped_rainy_s05_warm/best_model.zip` | ~95% | **33.3** steps |
| **0.3** | `runs/taxi_v4/shaped_rainy_s03_warm/best_model.zip` | ~88–93%¹ | **69.8** steps |

¹ At rainy 0.3 success is partly **capped by the 200-step limit**, not policy weakness (max
successful length = 199; ~12% timeouts are slow slip-heavy runs that would deliver with a higher cap).

These are `.zip` files in the **untracked `runs/` dir** → they exist only on this machine. Ask the
training session to copy them somewhere durable if you want them preserved.

### How they were trained (recipe that works)
From-scratch shaped PPO **collapses** at rainy 0.5/0.3 (sparse reward → "never pickup", flat −200).
The fix is a **curriculum warm-start** + **potential-based reward shaping** (`TaxiRewardShaping`,
training-env only, policy-invariant so the learned policy is still valid for the unshaped env the
diagnoser uses):
- 0.7: `train_taxi_v4_ppo.py --shaped --rainy_probability 0.7 --n_envs 8 --ent_coef 0.03 --eval_freq 6250` (~1.4M steps)
- 0.5: same but `--init_from runs/taxi_v4/shaped_rainy_s2/final_model.zip --rainy_probability 0.5 --timesteps 800000`
- 0.3: same but `--init_from runs/taxi_v4/shaped_rainy_s05_warm/best_model.zip --rainy_probability 0.3`

**Trajectory length doubles each time stochasticity halves** (20→33→70 steps for 0.7→0.5→0.3) →
lower rainy_probability gives longer trajectories = more observed gaps per instance (relevant to the
sim-hungry / hard-instance regime).

---

## 2. How to evaluate each policy

**Success@N (honest metric — deterministic policy on the plain UNSHAPED rainy env, success =
terminal +20 within the 200-step cap; exactly what the diagnoser uses):**

```bash
.venv_domains/Scripts/python.exe eval_taxi_policy.py \
  --model runs/taxi_v4/shaped_rainy_s05_warm/best_model.zip \
  --rainy_probability 0.5 --n_seeds 100 --base_seed 0
```
Flags: `--model` (.zip), `--rainy_probability`, `--n_seeds`, `--base_seed` (shift for fresh seeds,
e.g. 100 / 1000), `-v` (per-seed outcomes). **Match `--rainy_probability` to the policy's level.**
Always sanity-check fresh seed ranges (base_seed 100, 1000) — single ranges can be lucky.

**Trajectory length** (fast: tabulate the deterministic policy into a 500-state table, like
`h_rl_models.build_taxi_hardcoded_policy`, then roll out). Inline snippet:

```python
import statistics as st, gymnasium as gym
from stable_baselines3 import PPO
m = PPO.load("runs/taxi_v4/shaped_rainy_s05_warm/best_model.zip")
table = {s:int(m.predict(s,deterministic=True)[0]) for s in range(m.observation_space.n)}
env = gym.make("Taxi-v4", is_rainy=True, rainy_probability=0.5, fickle_passenger=False)
N=1000; succ_len=[]; succ=0
for i in range(N):
    obs,_=env.reset(seed=i); d=False; steps=0
    for steps in range(1,201):
        obs,r,term,trunc,_=env.step(table[int(obs)])
        if term: d=(r==20); break
        if trunc: break
    if d: succ+=1; succ_len.append(steps)
print(succ, st.mean(succ_len))   # report SUCCESS-only length (failures hit the 200 cap and inflate)
```
Report **successful-episode length** (failures hit the cap and inflate the mean). 1000 episodes is
cheap with the table.

---

## 3. Last discussed issue — per-step trajectory visualization & policy-field analysis

We dissected one concrete instance to understand what a trajectory looks like and how the policy
behaves spatially. **Reference instance:** `seed=42, rainy=0.7, served/hardcoded policy`:

```
states  (29): [386,286,286,186,86, 98,98,98,198,298,278,178,158,178,158,258,238,218,238,218,238,218,218,218,238,218,318,418,410]
actions (28): [1,1,1,1,4,  0,0,0,0,3,3,3,0,3,0,3,3,0,3,0,3,0,0,0,3,0,0,5]
```
- Start state 386 = taxi(3,4), passenger at G(0,4), dest Y(4,0). Delivers in **28 steps**.
- **Pickup splits the trajectory** at action index 4 (`Pickup`, taxi at (0,4)). BEFORE = reach G;
  AFTER = carry to Y then Dropoff. Visual grids (per-cell step indices; multiple numbers in a cell =
  rain slips kept the taxi there) clearly show a clean BEFORE phase and an AFTER phase with a
  "slip storm" bouncing between (2,0)↔(2,1) (steps 11–20) where moves repeatedly failed.

**Findings worth carrying forward:**

1. **The policy is goal-directed and wall-aware, not a fixed direction.** For a given destination the
   commanded action ≈ "reduce Manhattan distance to target, routing around walls." Example — policy
   command per cell, passenger aboard, dest=Y(4,0):
   ```
        c0     c1     c2     c3     c4
   r0  DOWN   DOWN   DOWN   UP     DOWN
   r1  DOWN   LEFT   DOWN   LEFT   DOWN
   r2  DOWN   LEFT   LEFT   LEFT   LEFT
   r3  DOWN   UP     LEFT   UP     UP
   r4 DROPOFF UP     UP     UP     UP
   ```
   DOWN/LEFT dominate because Y is the bottom-left corner, but **UP appears** in the bottom-right
   because column 0 is walled off from column 1 in rows 3–4, so a trapped taxi must go UP to the open
   row 2 to reach Y. (For dest=G the field would be mostly UP/RIGHT — it's destination-specific.)

2. **"RIGHT is never used" is geometry, not policy.** In the seed-42 episode RIGHT(East) is commanded
   0 times (the only unused action), because the route runs up-then-down-left and never needs
   eastward motion. Counts: DOWN 13, LEFT 9, UP 4, PICKUP 1, DROPOFF 1, RIGHT 0. A different
   start/pickup/dest layout uses RIGHT plenty. Within a single phase several actions are 0 (BEFORE:
   only UP+PICKUP; AFTER: only DOWN+LEFT+DROPOFF), so RIGHT is tied-least per-phase, uniquely-least
   over the whole episode.

These analyses were done with throwaway inline scripts (no new tracked files beyond `eval_taxi_policy.py`
and `analyze_hard_instance.py`); regenerate from the states/actions above + the env decode/encode.

---

## Decision raised (your call): do we even want a "better" policy for the diagnosis problem?
Recommendation on record: **use a good deterministic policy as the canonical setup** (realistic,
matches paper 1, makes faults identifiable, reproducible). Don't use a *bad* policy to manufacture
longer paths — longer paths give more gaps = more evidence = *easier* per-instance ID, and the only
sim-hunger they add is the trivial "more gaps × sims each." Manufacture real difficulty via
fault_rate↓ / visibility↓ / rainy_probability↓ / near-twin candidates (the validated hard regime).
A "worse policy" only belongs in the thesis as an explicit *robustness ablation*.
