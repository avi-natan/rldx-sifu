# Code Scan Conclusion — rldx-sifu (2026-06-16)

Scan of the codebase weighted toward Ahmad's changes (`master..new-master`), with the rest
read for context. Verified the two highest-impact claims directly (not just agent reports).

## 1. System architecture (end-to-end)

```
experiment driver (p_single_experiments.py)
   e.g. single_experiment_stochastic_FrozenLake / _Taxi_v4
        │  sets domain, fault modes, fault_rate, epsilon, visibility, unknown_fault_rate
        ▼
pipeline (p_pipeline.py)
   run_NON_DETERMINSTIC_single_experiment_PO / _FO
        │  prepare inputs → mask observations → call diagnoser → rank → write Excel
        ▼
executor (p_executor.py)  ──► generates the REAL trajectory
   make_wrapped_env(domain) + load_trained_model(domain)
   per step: a = policy(s); if rng.random() < fault_rate: a = fault_mode(a); env.step(a)
        ▼
diagnoser (p_diagnosers.py)  ──► infers the fault
   builds its OWN simulator, replays under each candidate fault, scores by likelihood
        │  uses: state refiners (h_state_refiners.py), comparators (h_raw_state_comparators.py),
        │        wrappers + set_state (h_wrappers.py), policy (h_rl_models.py)
        ▼
results → Excel  →  analysis/plots (scripts/*)
```

**Key data structures**
- Trajectory: `[s0, a0, s1, a1, …, sn]` (alternating); `faulty_actions_indices` marks faulted steps.
- Fault mode: a callable `action -> faulty_action`, keyed by a list-string like `"[0,0,2,3]"`
  (FrozenLake, 4 actions) or `"[0,0,2,3,4,5]"` (Taxi, 6 actions); built by `FaultModelGeneratorDiscrete`.
- Diagnosis result: `sorted_faults = [(fault_key, score), …]` ranked by log-likelihood, plus
  metrics (rank of true fault, top-1/2/3 hit, timing); unknown-rate path adds `best_rate_per_fault`.

## 2. The diagnosis algorithms (the heart — `p_diagnosers.py`, ~2583 lines)

Two families:

**A. Legacy deterministic family** (mostly previous student): `W`, `SN`, `SIF`, `SIFU`,
`SIFU2…SIFU8`. Symbolic filtering — maintain a hypothesis set `G`, replay each candidate fault,
eliminate hypotheses whose simulated state diverges from the observed state. SIFU variants differ
in how they exploit observation gaps and pre-filter fault modes. Dispatch table `diagnosers` at the
bottom of the file. These are **exact** (no sampling) and assume deterministic transitions.

**B. Stochastic family (Ahmad's core work)** — three functions:
- `fault_identification_non_deterministic_FO` (FO, fixed fault-rate grid `[0,0.1,…,1.0]`, 200 tries/step).
- `fault_identification_non_deterministic_PO` (PO, **known** fault rate).
- `fault_identification_non_deterministic_PO_unknown_fault_rate` (PO, **unknown** rate — added in
  commit `f7b9500d`; tries a list of candidate rates and picks the best per fault).

Their engine is **adaptive Monte Carlo** (`simulate_m_traces_adaptive_monte_carlo`):
- Between two observed states (a "gap"), repeatedly simulate traces (applying the candidate fault at
  rate `fault_rate`) and count how often the simulated end-state matches the observed one.
- Stop when the 95% CI half-width `1.96·√(p̂(1−p̂)/n)` drops below **epsilon** (`--epsilon`, default
  0.03), or when `max_tries` is hit. `max_tries` scales as `2500·(0.025/ε)² + 150·gap·(0.025/ε)²`.
- Score a fault by summing `log(p̂)` over gaps; rank faults by total log-likelihood. Intermittency
  model: `P(obs | fault, rate) = rate·P(obs|fault) + (1−rate)·P(obs|healthy)`.

## 3. Ahmad's changes vs `master` (20 commits; by weight)

| File | Lines (+) | What he added |
|------|-----------|---------------|
| `p_single_experiments.py` | +1143 | `single_experiment_stochastic_FrozenLake` / `_Taxi_v4` drivers |
| `p_diagnosers.py` | +727 | the entire stochastic family + adaptive MC + unknown-rate |
| `p_pipeline.py` | +441 | `run_NON_DETERMINSTIC_single_experiment_FO/PO`, non-det input prep (≥60-step retry, 50-try cap), seeded observation masking |
| `h_rl_models.py` | +115 (new) | `load_trained_model`, `FrozenLakeHardcodedPolicy` (JSON policies) |
| `h_wrappers.py` | +80 | `TaxiV4SetStepWrapper`, `make_wrapped_env`, `DOMAIN_KWARGS` (slippery/rainy) |
| `p_executor.py` | +13 | **seeded** per-instance RNG for reproducible fault injection |
| `frozen_lake_random_envs*.py`, `train_taxi_v4_ppo.py`, `scripts/*` | new | env/policy generation, PPO training, analysis & plotting |

Story of the work (chronological): add Taxi-v4 stochastic support → move to Python 3.11 → debug the
PO diagnoser → **add unknown-fault-rate** → tune adaptive-MC scaling → Taxi-v4 PPO training.

## 4. Verified findings (checked directly, correcting the scan agents)

**(a) Do the diagnosis simulators use the right stochastic kwargs? — MOSTLY YES (agents overstated a bug).**
The agents claimed diagnosis builds envs with `gym.make(name.replace('_','-'))` and drops the
slippery/rainy kwargs, implying the stochastic diagnosis is invalid. I checked every `simulator =`
line in `p_diagnosers.py`:
- The **stochastic functions Ahmad actually runs** — FO (line 183), PO unknown-rate (443), PO known-rate
  (670), and `SIF` (876) — all use `make_wrapped_env(...)`, which **does** inject `DOMAIN_KWARGS`
  (`is_slippery`, `is_rainy`, `rainy_probability`). ✅ Correct.
- Only the **legacy** `W` (21), `SN` (75), and `SIFU`/`SIFU2…8` (1020, 1185, …) use the bare
  `gym.make(... .replace('_','-'))` **without** kwargs → they'd simulate a non-slippery/non-rainy env.
  These are the previous student's deterministic algorithms and are **not** on Ahmad's stochastic path,
  so this is a latent inconsistency, not a live bug — *unless* you later run SIFU on a stochastic domain.

**(b) Taxi-v4 model wiring — REAL issue, confirmed.**
`load_trained_model` builds the path `environments/{domain}/models/{algo}/{domain}__{algo}.zip` →
`environments/Taxi_v4/models/PPO/Taxi_v4__PPO.zip`. Two files exist there:
- `Taxi_v4__PPO.zip` — **Dec 2025, old** ← this is what actually loads.
- `Taxi_v4_PPO_rainy_0.7_steps_1000000_seed_42` — **Jun 2026, newly trained, no `.zip`** ← **orphaned**, never loaded.
So Taxi-v4 experiments silently use the **old** policy, not the freshly-trained rainy one. This is a
strong candidate for why Taxi-v4 is "untested / not behaving." Fix = rename/point the loader at the new file.

## 5. Issues & risks (prioritized)

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | High | Newly-trained Taxi-v4 rainy model is orphaned (no `.zip`); old model loads instead | `train_taxi_v4_ppo.py` save name vs `h_rl_models.py:117-118` |
| 2 | Med | `main.py` parses `--epsilon/-ufr/-n` then **hard-overwrites** them (`args.epsilon=0.03`, etc.) → CLI is inert | `main.py:90-92` |
| 3 | Med | Stochastic drivers hardcode a single map (`loaded[1]`), `fault_rate=0.5`, `epsilon=0.03`, rainy_prob=0.7 — not parameterized | `p_single_experiments.py` (stochastic drivers), `h_wrappers.py:169-170` |
| 4 | Med | Legacy `SIFU*`/`W`/`SN` simulate without stochastic kwargs (see 4a) — fine today, a trap if reused for stochastic | `p_diagnosers.py:1020,1185,…` |
| 5 | Low | `FrozenLakeHardcodedPolicy` set via **global** `h_rl_models.HARD_CODED_POLICY` — not concurrency-safe | `h_rl_models.py:94,114`; set in `p_single_experiments.py` |
| 6 | Low | `scripts/*` have Windows-only hardcoded `C:/Users/ahmad/Downloads/...` paths | `scripts/explore_experiemnts.py`, `fault_rate_comparsion.py`, etc. |
| 7 | Low | Dead/broken legacy: `single_experiment_FrozenLake_NON_DETERMINSTIC` calls a non-existent function; `eval()` used in SIFU fault-mode filtering | `p_single_experiments.py`, `p_diagnosers.py` |
| 8 | Low | `# TODO more sophisticated ranking`; refiner TODO "identity for existing ones" | `p_pipeline.py:180`, `h_wrappers.py:18` |

## 6. Open questions for Ahmad (to confirm before acting)
1. **Taxi-v4 model**: should the loader use the new rainy model? (likely the fix to "untested Taxi-v4")
2. **CLI vs hardcode** (`main.py:90-92`): intentional pinning for now, or should flags actually drive runs?
3. **Single-map stochastic drivers**: meant for debugging one map, or should they sweep maps like the
   `multiple_experiment_*` functions?
4. **`-ufr` unknown-rate path**: how far along — is it the next thing to fully switch to?

---
*Note: line numbers are approximate anchors; the file is large and evolving. The two "verified findings"
in §4 were checked against the live source, not taken from agent summaries.*
