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

def make_minigrid_view_comparator(domain_name, render_seed=0):
    """Compare two MiniGrid states by the EGOCENTRIC VIEW they produce, not by the exact
    (col,row,dir): two states that look identical to the agent count as equal. This is the
    diagnosis signal under partial observability — the diagnoser only sees views.

    Uses the precomputed state->view map (built once per domain), so each comparison is an
    O(1) dict lookup rather than a gen_obs() render — critical because the Monte-Carlo hit
    test runs this hundreds of thousands of times.
    """
    from h_wrappers import build_minigrid_view_maps
    state2view, _ = build_minigrid_view_maps(domain_name, render_seed=render_seed)

    def view_compare(raw_state1, raw_state2):
        k1 = (int(raw_state1[0]), int(raw_state1[1]), int(raw_state1[2]))
        k2 = (int(raw_state2[0]), int(raw_state2[1]), int(raw_state2[2]))
        return state2view.get(k1, k1) == state2view.get(k2, k2)

    return view_compare


def _lazy_minigrid_view_comparator(domain_name):
    """A comparators[domain] entry that builds the view comparator on first use (so importing
    this module doesn't build a MiniGrid env), then caches it."""
    box = {}
    def cmp(raw_state1, raw_state2):
        fn = box.get("fn") or box.setdefault("fn", make_minigrid_view_comparator(domain_name))
        return fn(raw_state1, raw_state2)
    return cmp


comparators = {
    "Acrobot_v1": acrobot_compare,
    "CartPole_v1": cart_pole_compare,
    "MountainCar_v0": mountain_car_compare,
    "Taxi_v3": taxi_compare,
    "Taxi_v4": taxi_compare,
    "FrozenLake_v1": frozen_lake_compare,
    # MiniGrid is partially observed -> compare by egocentric VIEW (built lazily on first use).
    "MiniGrid_Empty_Random_6x6_v0": _lazy_minigrid_view_comparator("MiniGrid_Empty_Random_6x6_v0"),
    "MiniGrid_Empty_16x16_v0": _lazy_minigrid_view_comparator("MiniGrid_Empty_16x16_v0"),
}
