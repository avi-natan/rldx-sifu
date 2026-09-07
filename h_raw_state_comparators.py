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
    """Approach B0: compare states by the EGOCENTRIC VIEW they produce, not by the
    exact (col,row,dir). Two states that look identical to the agent count as equal.

    Returns a comparator closure holding one cached MiniGrid env used purely to render
    views (Empty's grid is fixed across seeds, so any state's view is well-defined).
    Swap it into `comparators[domain_name]` to run the diagnoser in B0 mode; the exact
    `minigrid_compare` stays the default (approach A).
    """
    import gymnasium
    import minigrid  # noqa: F401  (registers the envs)

    env = gymnasium.make(domain_name.replace('_', '-'))
    env.reset(seed=render_seed)
    u = env.unwrapped

    def _view_bytes(s):
        u.agent_pos = (int(s[0]), int(s[1]))
        u.agent_dir = int(s[2])
        return u.gen_obs()["image"].tobytes()

    def view_compare(raw_state1, raw_state2):
        return _view_bytes(raw_state1) == _view_bytes(raw_state2)

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
