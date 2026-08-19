# Taxi-v4 policy library

Good Taxi-v4 PPO policies, kept in one place with informative names.
Success rates are `success@100` (deterministic rollout — the mode the diagnoser uses),
measured on the rainy env at `rainy_probability=0.7` unless noted.

The **served** policy (`../Taxi_v4__PPO.zip`, the one the diagnosers load) is
`taxi_v4__rainy07_shaped_final__98of100.zip` — the **final** model of the
rainy-0.7 shaped run.

| file | md5 | success@0.7 | source run | notes |
|------|-----|-------------|------------|-------|
| `taxi_v4__rainy07_shaped_final__98of100.zip` | `62aa934d` | **98/100** | `runs/taxi_v4/shaped_rainy_s2/final_model.zip` | **SERVED** — final policy trained on rainy=0.7 |
| `taxi_v4__rainy07_champion__100of100.zip` | `aab3b43b` | 100/100 | `runs/taxi_v4/shaped_rainy_s2/best_model.zip` | best checkpoint of the same rainy-0.7 run |
| `taxi_v4__deterministic_taxiv3_oldstrong__98of100.zip` | `f14cbfef` | 98/100 | Taxi-v3 PPO policy | deterministic-env policy (reused); old served policy |
| `taxi_v4__rainy05_specialist__98at07_97at05.zip` | `0b63487a` | 98/100 (97/100 @0.5) | `runs/taxi_v4/shaped_rainy_s05_warm/best_model.zip` | tuned for rainy=0.5 (warm-started from s2) |
| `taxi_v4__rainy05_specialist_final__94of100.zip` | `03ace9b1` | 94/100 | `runs/taxi_v4/shaped_rainy_s05_warm/final_model.zip` | rainy=0.5 run, final |
| `taxi_v4__lowrain_s03_best__94of100.zip` | `802bcb26` | 94/100 | `runs/taxi_v4/shaped_rainy_s03_warm/best_model.zip` | low-rain run, best |
| `taxi_v4__lowrain_s03_final__91of100.zip` | `5766d83d` | 91/100 | `runs/taxi_v4/shaped_rainy_s03_warm/final_model.zip` | low-rain run, final |

Excluded (collapsed / failed runs, ≤40/100 @0.7, kept only under `runs/`):
`shaped_rainy_s1`, `shaped_rainy_s05`, `shaped_det`, `shaped_det2`, `det`,
`rainy_scratch`, `rainy_highent`, `smoke`.
