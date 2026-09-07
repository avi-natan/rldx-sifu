"""Approach B on MiniGrid Empty: diagnose from OBSERVATIONS (views) only.

Compares, on the same instances:
  A = exact-state set_state + exact-state comparison (full observability)
  B = belief set_state (localize the view -> sample a consistent (col,row,dir)) +
      view comparison (true partial observability)

B never uses the exact state: at each gap it only knows the observed view, looks up all
states consistent with it, and samples one (position AND direction) to roll forward from;
the hit test compares views. Prints how the true fault's rank changes A -> B.

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/minigrid_empty_B_ab.py \
      --domain MiniGrid_Empty_16x16_v0 --max_len 55
"""
import os, sys, argparse, random as pyrandom
sys.path.insert(0, os.path.abspath("."))

import h_wrappers
from h_wrappers import make_wrapped_env
from h_rl_models import load_trained_model
from h_state_refiners import refiners
import h_raw_state_comparators as C
from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import fault_identification_non_deterministic_PO

_ap = argparse.ArgumentParser()
_ap.add_argument("--domain", default="MiniGrid_Empty_16x16_v0")
_ap.add_argument("--max_len", type=int, default=55)
_ap.add_argument("--hide_every", type=int, default=2,
                 help="hide 1 of every N interior states (2 = ~50%% visible; 3 = ~67%%; 1000 = ~100%%)")
_ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101, 2024, 55, 900])
_args, _ = _ap.parse_known_args()

DOMAIN = _args.domain
MODEL = "PPO"
RENDER = "rgb_array"
FAULT_RATE = 0.5
EPSILON = 0.05
MAX_LEN = _args.max_len
HIDE_EVERY = _args.hide_every
SEEDS = _args.seeds

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
    """Generate the observed trajectory under TRUE dynamics (belief mode OFF here)."""
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
        if i % HIDE_EVERY == 0:
            obs[i] = None
    return obs


def diagnose(observations, seed):
    out = fault_identification_non_deterministic_PO(
        debug_print=False, render_mode=RENDER, instance_seed=seed,
        ml_model_name=MODEL, domain_name=DOMAIN, observations=observations,
        candidate_fault_modes=candidate_fault_modes, epsilon=EPSILON, fault_rate=FAULT_RATE,
    )
    order = [f for f, _ in out["sorted_faults"]]
    return order.index(TRUE_FAULT) + 1


def main():
    refiner = refiners[DOMAIN]
    exact_cmp = C.minigrid_compare
    view_cmp = C.make_minigrid_view_comparator(DOMAIN)

    # generate the observed trajectories once, under true dynamics (belief mode off)
    h_wrappers.MINIGRID_BELIEF_MODE = False
    gen_sim = make_wrapped_env(DOMAIN, RENDER)
    policy = load_trained_model(DOMAIN, MODEL)
    trajs = {seed: faulty_trajectory(gen_sim, policy, refiner,
                                     candidate_fault_modes[TRUE_FAULT], seed) for seed in SEEDS}

    print(f"domain={DOMAIN}")
    print(f"{'seed':>6} | {'rank A (state)':>14} | {'rank B (views+belief)':>21} | note")
    print("-" * 66)
    ra, rb = [], []
    for seed in SEEDS:
        obs = trajs[seed]

        # A: full observability
        h_wrappers.MINIGRID_BELIEF_MODE = False
        C.comparators[DOMAIN] = exact_cmp
        rank_a = diagnose(obs, seed)

        # B: partial observability (localize + sample + compare views)
        h_wrappers.MINIGRID_BELIEF_MODE = True
        C.comparators[DOMAIN] = view_cmp
        rank_b = diagnose(obs, seed)

        # restore defaults
        h_wrappers.MINIGRID_BELIEF_MODE = False
        C.comparators[DOMAIN] = exact_cmp

        note = "same" if rank_a == rank_b else ("WORSE under B" if rank_b > rank_a else "better")
        print(f"{seed:>6} | {rank_a:>14} | {rank_b:>21} | {note}")
        ra.append(rank_a); rb.append(rank_b)

    print("-" * 66)
    print(f"avg rank  A (full state) = {sum(ra)/len(ra):.2f}  |  "
          f"B (views+belief) = {sum(rb)/len(rb):.2f}   (1 = best; higher = worse)")


if __name__ == "__main__":
    main()
