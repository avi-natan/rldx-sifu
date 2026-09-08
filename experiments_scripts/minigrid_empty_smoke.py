"""Smoke test: MiniGrid Empty partial-observability diagnosis, end-to-end.

MiniGrid is partially observed: the diagnoser only ever sees the agent's egocentric VIEW.
The wrapper's set_state localizes the observed view to the states consistent with it and
SAMPLES one (position AND direction); the comparator compares views. Both are the defaults
for MiniGrid domains (h_wrappers / h_raw_state_comparators), so we just run the diagnoser.

This injects a known fault + action stochasticity, masks interior states into observation
gaps, and checks the true fault ranks near the top.

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/minigrid_empty_smoke.py
"""
import os, sys, random as pyrandom
sys.path.insert(0, os.path.abspath("."))

from h_wrappers import make_wrapped_env
from h_rl_models import load_trained_model
from h_state_refiners import refiners
from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import fault_identification_non_deterministic_PO

DOMAIN = "MiniGrid_Empty_16x16_v0"
MODEL = "PPO"
RENDER = "rgb_array"
FAULT_RATE = 0.5
EPSILON = 0.05
MAX_LEN = 40
INSTANCE_SEED = 777

gen = FaultModelGeneratorDiscrete()
# candidate fault modes over the 7 MiniGrid actions (0=left,1=right,2=forward,3-6 no-ops)
FAULTS = {
    "swap_LR":      "{0:1,1:0,2:2,3:3,4:4,5:5,6:6}",   # injected (true) fault
    "fwd_to_left":  "{0:0,1:1,2:0,3:3,4:4,5:5,6:6}",
    "fwd_to_right": "{0:0,1:1,2:1,3:3,4:4,5:5,6:6}",
    "identity":     "{0:0,1:1,2:2,3:3,4:4,5:5,6:6}",
}
TRUE_FAULT = "swap_LR"
candidate_fault_modes = {n: gen.generate_fault_model(s) for n, s in FAULTS.items()}


def main():
    sim = make_wrapped_env(DOMAIN, RENDER)   # partial-obs wrapper (localize view + sample state)
    policy = load_trained_model(DOMAIN, MODEL)
    refiner = refiners[DOMAIN]

    # generate the observed (faulty) trajectory under the true dynamics
    s, _ = sim.reset(seed=INSTANCE_SEED)
    traj = [s]
    rng = pyrandom.Random(INSTANCE_SEED)
    fault = candidate_fault_modes[TRUE_FAULT]
    for _ in range(MAX_LEN):
        a, _ = policy.predict(refiner(s)); a = int(a)
        if rng.random() < FAULT_RATE:
            a = fault(a)
        s, r, term, trunc, info = sim.step(a)
        traj.append(s)
        if term or trunc:
            break

    # mask every other interior state to create observation gaps
    observations = list(traj)
    for i in range(1, len(observations) - 1):
        if i % 2 == 0:
            observations[i] = None

    out = fault_identification_non_deterministic_PO(
        debug_print=False, render_mode=RENDER, instance_seed=INSTANCE_SEED,
        ml_model_name=MODEL, domain_name=DOMAIN, observations=observations,
        candidate_fault_modes=candidate_fault_modes, epsilon=EPSILON, fault_rate=FAULT_RATE,
    )
    sf = out["sorted_faults"]
    print("sorted_faults (best first):", sf)
    print("TRUE fault:", TRUE_FAULT, "| ranked #", [f for f, _ in sf].index(TRUE_FAULT) + 1)


if __name__ == "__main__":
    main()
