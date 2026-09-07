import numpy as np


def acrobot_compare(raw_state1, raw_state2):
    return np.array_equal(raw_state1, raw_state2)

def cart_pole_compare(raw_state1, raw_state2):
    return np.array_equal(raw_state1, raw_state2)

def mountain_car_compare(raw_state1, raw_state2):
    return np.array_equal(raw_state1, raw_state2)

def taxi_compare(raw_state1, raw_state2):
    s1 = int(raw_state1)
    s2 = int(raw_state2)
    return s1 == s2

def frozen_lake_compare(raw_state1, raw_state2):
    s1 = int(raw_state1)
    s2 = int(raw_state2)
    return s1 == s2

def minigrid_compare(raw_state1, raw_state2):
    # raw state is (agent_col, agent_row, agent_dir); exact tuple equality (approach A)
    return tuple(raw_state1) == tuple(raw_state2)


def make_minigrid_view_comparator(domain_name, render_seed=0):
    """Approach B0/B: compare states by the EGOCENTRIC VIEW they produce, not by the
    exact (col,row,dir). Two states that look identical to the agent count as equal.

    Uses the precomputed state->view map (built once per domain), so each comparison is an
    O(1) dict lookup rather than a gen_obs() render — critical because the Monte-Carlo hit
    test runs this hundreds of thousands of times. Swap it into `comparators[domain_name]`
    for B0/B mode; the exact `minigrid_compare` stays the default (approach A).
    """
    from h_wrappers import build_minigrid_view_maps
    state2view, _ = build_minigrid_view_maps(domain_name, render_seed=render_seed)

    def view_compare(raw_state1, raw_state2):
        k1 = (int(raw_state1[0]), int(raw_state1[1]), int(raw_state1[2]))
        k2 = (int(raw_state2[0]), int(raw_state2[1]), int(raw_state2[2]))
        return state2view.get(k1, k1) == state2view.get(k2, k2)

    return view_compare


comparators = {
    "Acrobot_v1": acrobot_compare,
    "CartPole_v1": cart_pole_compare,
    "MountainCar_v0": mountain_car_compare,
    "Taxi_v3": taxi_compare,
    "Taxi_v4": taxi_compare,
    "FrozenLake_v1": frozen_lake_compare,
    "MiniGrid_Empty_Random_6x6_v0": minigrid_compare,
    "MiniGrid_Empty_16x16_v0": minigrid_compare,
}
