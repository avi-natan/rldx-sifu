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
    # raw state is (agent_col, agent_row, agent_dir); exact tuple equality
    return tuple(raw_state1) == tuple(raw_state2)


comparators = {
    "Acrobot_v1": acrobot_compare,
    "CartPole_v1": cart_pole_compare,
    "MountainCar_v0": mountain_car_compare,
    "Taxi_v3": taxi_compare,
    "Taxi_v4": taxi_compare,
    "FrozenLake_v1": frozen_lake_compare,
    "MiniGrid_Empty_Random_6x6_v0": minigrid_compare,
}
