"""Smoke test: MiniGrid Empty (approach A, full states) end-to-end through the
stochastic diagnoser.

Approach A = the diagnoser runs on the FULL environment state (agent col,row,dir),
not the agent's egocentric observation. Generates a faulty trajectory with a KNOWN
injected fault + action stochasticity (StochasticActionWrapper), masks some interior
states to create observation gaps, and checks the diagnoser ranks the true fault first.

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/minigrid_empty_smoke.py
"""
import os, sys, random as pyrandom
sys.path.insert(0, os.path.abspath("."))

from h_wrappers import make_wrapped_env
from h_rl_models import load_trained_model
from h_state_refiners import refiners
from h_raw_state_comparators import comparators
from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import fault_identification_non_deterministic_PO

DOMAIN = "MiniGrid_Empty_Random_6x6_v0"
MODEL = "PPO"
RENDER = "rgb_array"

gen = FaultModelGeneratorDiscrete()
# candidate fault modes over the 7 MiniGrid actions (0=left, 1=right, 2=forward, 3-6 no-ops)
FAULTS = {
    "swap_LR":      "{0:1,1:0,2:2,3:3,4:4,5:5,6:6}",   # <-- the injected (true) fault
    "fwd_to_left":  "{0:0,1:1,2:0,3:3,4:4,5:5,6:6}",
    "fwd_to_right": "{0:0,1:1,2:1,3:3,4:4,5:5,6:6}",
    "identity":     "{0:0,1:1,2:2,3:3,4:4,5:5,6:6}",
}
TRUE_FAULT = "swap_LR"
candidate_fault_modes = {name: gen.generate_fault_model(spec) for name, spec in FAULTS.items()}

INSTANCE_SEED = 777
FAULT_RATE = 0.5
EPSILON = 0.05
MAX_LEN = 18


def rollout(sim, policy, refiner, fault=None, fault_rate=0.0, seed=INSTANCE_SEED):
    s, _ = sim.reset(seed=seed)
    traj = [s]
    rng = pyrandom.Random(seed)
    for _ in range(MAX_LEN):
        a, _ = policy.predict(refiner(s)); a = int(a)
        if fault is not None and rng.random() < fault_rate:
            a = fault(a)
        s, r, term, trunc, info = sim.step(a)
        traj.append(s)
        if term or trunc:
            break
    return traj, term


def main():
    sim = make_wrapped_env(DOMAIN, RENDER)
    policy = load_trained_model(DOMAIN, MODEL)
    refiner = refiners[DOMAIN]

    # 1. sanity: healthy policy reaches the goal
    healthy, reached = rollout(sim, policy, refiner)
    print(f"healthy trajectory ({len(healthy)} states), reached_goal={reached}: {healthy}")

    # 2. faulty (observed) trajectory with the known fault
    traj, _ = rollout(sim, policy, refiner, candidate_fault_modes[TRUE_FAULT], FAULT_RATE)
    print(f"faulty trajectory  ({len(traj)} states): {traj}")

    # 3. mask every other interior state to create gaps (keep endpoints)
    observations = list(traj)
    for i in range(1, len(observations) - 1):
        if i % 2 == 0:
            observations[i] = None
    print(f"observations (with gaps): {observations}")

    # 4. run the diagnoser
    out = fault_identification_non_deterministic_PO(
        debug_print=False, render_mode=RENDER,
        instance_seed=INSTANCE_SEED, ml_model_name=MODEL,
        domain_name=DOMAIN, observations=observations,
        candidate_fault_modes=candidate_fault_modes,
        epsilon=EPSILON, fault_rate=FAULT_RATE,
    )
    sf = out.get("sorted_faults")
    print("\nsorted_faults (best first):", sf)
    print("TRUE fault:", TRUE_FAULT, "| ranked #", [f for f, _ in sf].index(TRUE_FAULT) + 1)


if __name__ == "__main__":
    main()
