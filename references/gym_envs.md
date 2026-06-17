# Gymnasium environments — specs & domains catalog (for RLDX fault diagnosis)

Condensed from the Farama Gymnasium docs (toy_text). Source:
<https://gymnasium.farama.org/environments/toy_text/>. Focus = the properties that matter for the
stochastic diagnosers (action space → fault-mode space, stochasticity → diagnosis signal,
transition probabilities → Monte-Carlo cost). See [[research-state-post-24.5]].

## Domains catalog (Table-1 style; seeded with FrozenLake + Taxi)

| Domain | Obs space | #Actions | Stochasticity (knob, default) | Reward | Truncation | Used here |
|---|---|---|---|---|---|---|
| **FrozenLake-v1** | `Discrete(n²)` = 16 (4×4) / 64 (8×8); `row*ncols+col` | 4 | slip: `is_slippery`, `success_rate` **1/3** (1/3 to each perpendicular; no backward) | goal +1, else 0 | 100 (4×4) / 200 (8×8) | ✅ working |
| **Taxi-v3/v4** | `Discrete(500)` (404 reachable) | 6 | rain: `is_rainy`(F), `rainy_probability` **0.8** *(code uses 0.7)*; also `fickle_passenger`(F), `fickle_probability` 0.3 | −1/step, +20 delivery, −10 illegal pickup/dropoff | 200 | ⚠️ added, untested |
| **CliffWalking-v0** | `Discrete(48)` (4×12) | 4 | (planned — fill when added) | per-step −1, cliff −100 | — | 🔜 planned |

> Fill remaining columns (discrete/continuous, static/dynamic, avg trajectory length) as we run each
> domain. Mirrors Table 1 of the SIF/SIFU paper (`references/recent_paper_1.pdf`).

## FrozenLake-v1
- **Actions `Discrete(4)`:** `0=Left, 1=Down, 2=Right, 3=Up`.
- **Obs `Discrete(n²)`:** integer `row*ncols + col`. Start = state 0; goal = `n²−1`. Tiles S/G/F/H.
- **Slippery (`is_slippery=True`):** intended dir w.p. `success_rate` (default **1/3**), each of the two
  **perpendicular** dirs w.p. 1/3. No backward slip. `success_rate` tunable.
- **Rewards:** goal +1, hole/frozen 0 (customizable via `reward_schedule`).
- **Termination:** reach goal or fall in hole. **Truncation:** 100 (4×4) / 200 (8×8).
- **Randomness:** slip only; random maps via `generate_random_map()` (cf. `frozen_lake_random_envs*.py`).

## Taxi-v3 / v4
- **Actions `Discrete(6)`:** `0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff`.
- **Obs `Discrete(500)`:** `((taxi_row*5 + taxi_col)*5 + passenger_loc)*4 + destination`. Passenger loc
  `0=R,1=G,2=Y,3=B,4=in-taxi`; dest `0–3`. **404 reachable**; 300 valid starts
  (25 positions × 4 pass-locs × 3 dests).
- **Rewards:** −1/step, +20 delivery, −10 illegal pickup/dropoff.
- **Stochastic option `is_rainy`** (default False): intended dir w.p. `rainy_probability`
  (env default **0.8**, **our code uses 0.7**), lateral split evenly.
- **Also `fickle_passenger`** (default False, `fickle_probability` 0.3): destination can change after the
  first post-pickup move — a *second* stochasticity source; set False unless explicitly modeled.
- **Extras:** `info["action_mask"]` (which actions actually move — wall guard); `info["prob"]`
  (transition prob of the taken step → usable to cut MC simulation cost).
- **Termination:** successful dropoff. **Truncation:** 200 steps.

## Implications for the diagnosers
- **Fault modes = action permutations** → indexed by these action ids. Taxi's 6 actions (incl.
  pickup/dropoff, the −10 actions) give a much richer fault-mode space than FrozenLake's 4 moves.
- **Taxi has two stochasticity layers** (rainy + fickle) vs FrozenLake's one (slip). For clean MC
  likelihoods, keep `fickle_passenger=False` unless deliberately modeled.
- **`success_rate` / `rainy_probability` set the noise level = the diagnosis signal strength.** Low
  noise → policy rarely deviates → weak signal. Ties to the open question "is a too-good policy even
  desirable?" — a near-optimal, rarely-deviating policy may give fewer diagnosis signals.
- **`info["prob"]`** could feed the adaptive Monte-Carlo estimator directly, reducing simulations
  (the dominant cost, per the profiling finding).
