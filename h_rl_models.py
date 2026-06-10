import numpy as np
from stable_baselines3 import PPO, A2C, DQN

models = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}


"""
1. 

Map:
SFFFFFFF
FFHFFFFF
FFFHFFFF
HFFFFFFF
FFFHFFFH
FFFFFFFF
FFFFFFFF
FFFFHFFG

Policy:
↓ ↓ → ↓ ↓ ↓ ↓ ←
↓ ↓ H → ↓ ↓ ↓ ←
→ ↓ ↓ H ↓ ↓ ↓ ←
H ↓ ↓ → ↓ ↓ ↓ ←
↓ ↓ ↓ H ↓ ↓ ↓ H
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
→ → → → → ↓ ↓ ↓
→ → → ↑ H → → G

"""



"""
2. 

Map:
SFFFFFFF
FFHFFFFF
FFFHFFFF
HFFFFFFF
FFFHFFFH
FFFFFFFF
FFFFFFFF
FFFFHFFG

Policy:
→ → → ↓ ← ← ← ←
H H H ↓ ↓ ← ← ↑
→ → → → ↓ H ↑ ↑
H H H H ↓ H H ↑
→ → → → → ↓ H H
H H H H H ↓ ← ←
→ → → → → → ↓ ←
→ ↑ → ↑ H → → G

"""


LEFT  = 0
DOWN  = 1
RIGHT = 2
UP    = 3

HARD_CODED_POLICY = {
    # row 0 (states 0–7)
    0: DOWN,  1: DOWN,  2: RIGHT, 3: DOWN,  4: DOWN,  5: DOWN,  6: DOWN,  7: LEFT,

    # row 1 (8–15)
    8: DOWN,  9: DOWN, 10: DOWN,  11: RIGHT, 12: DOWN, 13: DOWN, 14: DOWN, 15: LEFT,

    # row 2 (16–23)
    16: RIGHT, 17: DOWN, 18: DOWN, 19: DOWN, 20: DOWN, 21: DOWN, 22: DOWN, 23: LEFT,

    # row 3 (24–31)
    24: DOWN, 25: DOWN, 26: DOWN, 27: RIGHT, 28: DOWN, 29: DOWN, 30: DOWN, 31: LEFT,

    # row 4 (32–39)
    32: DOWN, 33: DOWN, 34: DOWN, 35: DOWN, 36: DOWN, 37: DOWN, 38: DOWN, 39: DOWN,

    # row 5 (40–47)
    40: DOWN, 41: DOWN, 42: DOWN, 43: DOWN, 44: DOWN, 45: DOWN, 46: DOWN, 47: DOWN,

    # row 6 (48–55)
    48: RIGHT, 49: RIGHT, 50: RIGHT, 51: RIGHT, 52: RIGHT, 53: DOWN, 54: DOWN, 55: DOWN,

    # row 7 (56–63)
    56: RIGHT, 57: RIGHT, 58: RIGHT, 59: UP,   60: DOWN, 61: RIGHT, 62: RIGHT, 63: RIGHT,
}
HARD_CODED_POLICY = None

class FrozenLakeHardcodedPolicy:
    def __init__(self, policy_dict):
        self.policy_dict = policy_dict

    def predict(self, obs, deterministic=True):
        # obs will be an int or a 0-D numpy array after your refiner
        if isinstance(obs, np.ndarray):
            s = int(obs.item())
        else:
            s = int(obs)
        action = self.policy_dict[s]
        return action, None


def load_trained_model(domain_name, ml_model_name, env=None):

    # should be deleted after i have a good policy
    if domain_name == "FrozenLake_v1":
        assert HARD_CODED_POLICY is not None, "FrozenLake policy not set"
        return FrozenLakeHardcodedPolicy(HARD_CODED_POLICY)

    print(f"hereee: {domain_name}")
    print(f"hereee: {ml_model_name}")

    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    print(f"hereee3: {model_path}")
    if env is None:
        return models[ml_model_name].load(model_path)
    else:
        return models[ml_model_name].load(model_path, env=env)