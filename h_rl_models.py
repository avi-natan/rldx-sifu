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


class TaxiHardcodedPolicy:
    """Deterministic policy as a {state: action} lookup table.

    Taxi has only Discrete(500) states, so we precompute the trained model's
    deterministic action for every state once and then serve diagnosis from a
    dict lookup instead of a neural-net forward pass on every simulated step.
    Behaviour is identical to model.predict(obs, deterministic=True).
    """
    def __init__(self, policy_dict):
        self.policy_dict = policy_dict

    def predict(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray):
            s = int(obs.item())
        else:
            s = int(obs)
        return self.policy_dict[s], None


# cache the precomputed table per model path so we build it once per process
_TAXI_POLICY_CACHE = {}

def build_taxi_hardcoded_policy(model_path, ml_model_name):
    """Load the trained model once and tabulate its deterministic action for
    every Taxi state, returning a TaxiHardcodedPolicy. Cached by model_path."""
    if model_path in _TAXI_POLICY_CACHE:
        return _TAXI_POLICY_CACHE[model_path]

    model = models[ml_model_name].load(model_path)
    n_states = model.observation_space.n
    table = {s: int(model.predict(s, deterministic=True)[0]) for s in range(n_states)}

    policy = TaxiHardcodedPolicy(table)
    _TAXI_POLICY_CACHE[model_path] = policy
    print(f"[TaxiHardcodedPolicy] tabulated {n_states} states from {model_path}")
    return policy


def load_trained_model(domain_name, ml_model_name, env=None):

    # should be deleted after i have a good policy
    if domain_name == "FrozenLake_v1":
        assert HARD_CODED_POLICY is not None, "FrozenLake policy not set"
        return FrozenLakeHardcodedPolicy(HARD_CODED_POLICY)

    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"

    # Taxi: serve the deterministic policy from a precomputed 500-state table
    # (huge speedup for Monte-Carlo, which calls the policy on every step).
    if domain_name == "Taxi_v4":
        return build_taxi_hardcoded_policy(model_path, ml_model_name)

    if env is None:
        return models[ml_model_name].load(model_path)
    else:
        return models[ml_model_name].load(model_path, env=env)