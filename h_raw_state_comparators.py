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
    from h_wrappers import (build_minigrid_view_maps, MINIGRID_PER_SEED_LAYOUT,
                            _MINIGRID_ACTIVE_MAPS)
    per_seed = domain_name in MINIGRID_PER_SEED_LAYOUT
    # Fixed-layout domains (Empty): one precomputed map. Per-seed-layout domains (SimpleCrossing):
    # the map differs per instance, so read the CURRENT instance's map (set by the wrapper on reset).
    fixed_map = None if per_seed else build_minigrid_view_maps(domain_name, render_seed=render_seed)[0]

    def view_compare(raw_state1, raw_state2):
        state2view = _MINIGRID_ACTIVE_MAPS.get(domain_name) if per_seed else fixed_map
        k1 = (int(raw_state1[0]), int(raw_state1[1]), int(raw_state1[2]))
        k2 = (int(raw_state2[0]), int(raw_state2[1]), int(raw_state2[2]))
        # NO silent fallbacks: a missing map or an unmapped state means the comparator would
        # otherwise degrade to raw (col,row,dir) equality and silently corrupt the whole diagnosis
        # (views compared as if they were exact states). Both are wiring bugs -> fail LOUD instead.
        if state2view is None:
            raise RuntimeError(
                f"MiniGrid view comparator: NO active view map for per-seed-layout domain "
                f"'{domain_name}'. The env must be reset (which publishes _MINIGRID_ACTIVE_MAPS) "
                f"before any state is compared. This is a wiring bug, not a recoverable state.")
        try:
            v1, v2 = state2view[k1], state2view[k2]
        except KeyError as miss:
            raise KeyError(
                f"MiniGrid view comparator: state {miss.args[0]} is not in the active view map for "
                f"'{domain_name}' (per_seed={per_seed}, |map|={len(state2view)}). Every compared "
                f"state must be a valid FREE cell of the CURRENT layout; a miss means a wall / "
                f"off-grid / wrong-layout state leaked in. Refusing to fall back to raw equality.") from None
        return v1 == v2

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
    "MiniGrid_SimpleCrossing_S11N2_v0": _lazy_minigrid_view_comparator("MiniGrid_SimpleCrossing_S11N2_v0"),
}
