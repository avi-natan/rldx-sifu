"""B0 A/B test on MiniGrid Empty: same instances, diagnosed twice —
  A  = exact STATE comparison (approach A)
  B0 = EGOCENTRIC-VIEW comparison (states in, views compared)
and prints how the true fault's rank/score changes. Everything else is identical
(same faulty trajectory, gaps, seeds, epsilon, fault rate); ONLY the comparator differs.

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/minigrid_empty_B0_ab.py
"""
import os, sys, random as pyrandom
sys.path.insert(0, os.path.abspath("."))

from h_wrappers import make_wrapped_env
from h_rl_models import load_trained_model
from h_state_refiners import refiners
import h_raw_state_comparators as C
from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import fault_identification_non_deterministic_PO

DOMAIN = "MiniGrid_Empty_Random_6x6_v0"
MODEL = "PPO"
RENDER = "rgb_array"
FAULT_RATE = 0.5
EPSILON = 0.05
MAX_LEN = 18
SEEDS = [777, 101, 2024, 55, 900]

gen = FaultModelGeneratorDiscrete()
FAULTS = {
    "swap_LR":      "{0:1,1:0,2:2,3:3,4:4,5:5,6:6}",   # injected (true) fault
    "fwd_to_left":  "{0:0,1:1,2:0,3:3,4:4,5:5,6:6}",
    "fwd_to_right": "{0:0,1:1,2:1,3:3,4:4,5:5,6:6}",
    "identity":     "{0:0,1:1,2:2,3:3,4:4,5:5,6:6}",
}
TRUE_FAULT = "swap_LR"
candidate_fault_modes = {n: gen.generate_fault_model(s) for n, s in FAULTS.items()}


def faulty_trajectory(sim, policy, refiner, fault, seed):
    s, _ = sim.reset(seed=seed)
    traj = [s]
    rng = pyrandom.Random(seed)
    for _ in range(MAX_LEN):
        a, _ = policy.predict(refiner(s)); a = int(a)
        if rng.random() < FAULT_RATE:
            a = fault(a)
        s, r, term, trunc, info = sim.step(a)
        traj.append(s)
        if term or trunc:
            break
    obs = list(traj)
    for i in range(1, len(obs) - 1):
        if i % 2 == 0:
            obs[i] = None
    return obs


def diagnose(observations, seed):
    out = fault_identification_non_deterministic_PO(
        debug_print=False, render_mode=RENDER, instance_seed=seed,
        ml_model_name=MODEL, domain_name=DOMAIN, observations=observations,
        candidate_fault_modes=candidate_fault_modes, epsilon=EPSILON, fault_rate=FAULT_RATE,
    )
    order = [f for f, _ in out["sorted_faults"]]
    return order, order.index(TRUE_FAULT) + 1, dict(out["sorted_faults"])


def main():
    sim = make_wrapped_env(DOMAIN, RENDER)
    policy = load_trained_model(DOMAIN, MODEL)
    refiner = refiners[DOMAIN]

    exact_cmp = C.minigrid_compare
    view_cmp = C.make_minigrid_view_comparator(DOMAIN)

    print(f"{'seed':>6} | {'rank A (states)':>15} | {'rank B0 (views)':>15} | note")
    print("-" * 62)
    ranks_a, ranks_b0 = [], []
    for seed in SEEDS:
        obs = faulty_trajectory(sim, policy, refiner, candidate_fault_modes[TRUE_FAULT], seed)

        C.comparators[DOMAIN] = exact_cmp
        _, rank_a, _ = diagnose(obs, seed)

        C.comparators[DOMAIN] = view_cmp
        order_b, rank_b, scores_b = diagnose(obs, seed)

        C.comparators[DOMAIN] = exact_cmp  # restore
        note = "same" if rank_a == rank_b else ("WORSE under views" if rank_b > rank_a else "better")
        print(f"{seed:>6} | {rank_a:>15} | {rank_b:>15} | {note}")
        ranks_a.append(rank_a); ranks_b0.append(rank_b)

    print("-" * 62)
    print(f"avg rank  A (states) = {sum(ranks_a)/len(ranks_a):.2f}  |  "
          f"B0 (views) = {sum(ranks_b0)/len(ranks_b0):.2f}   (rank 1 = best; higher = worse)")


if __name__ == "__main__":
    main()
