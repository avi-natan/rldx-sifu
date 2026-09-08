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


class MiniGridEmptyHardcodedPolicy:
    """Deterministic greedy navigation policy for a MiniGrid Empty room.

    Consumes the raw state (agent_col, agent_row, agent_dir) and returns a discrete
    action toward the fixed goal at (width-2, height-2).
      actions:     0 = turn left, 1 = turn right, 2 = move forward
      directions:  0 = east(+x), 1 = south(+y), 2 = west(-x), 3 = north(-y)
    Defined for EVERY state (the Monte-Carlo diagnoser queries off-path states too).
    Behaves like a deterministic trained policy: same state -> same action.
    """
    TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2

    def __init__(self, goal):
        self.goal = goal

    def predict(self, obs, deterministic=True):
        x, y, d = int(obs[0]), int(obs[1]), int(obs[2])
        gx, gy = self.goal
        dx, dy = gx - x, gy - y
        if dx == 0 and dy == 0:
            return self.FORWARD, None  # at goal; episode terminates on arrival
        # desired heading: close the larger axis first
        if abs(dx) >= abs(dy):
            desired = 0 if dx > 0 else 2
        else:
            desired = 1 if dy > 0 else 3
        if d == desired:
            return self.FORWARD, None
        # rotate toward the desired heading (dir+1 = turn right, dir-1 = turn left)
        diff = (desired - d) % 4
        return self.TURN_LEFT if diff == 3 else self.TURN_RIGHT, None


def minigrid_obs_vector(image, direction):
    """The EXACT observation the trained policy consumes, rebuilt from a state's raw MiniGrid
    observation: flatten(image)/255 ++ one_hot(direction). MUST match
    experiments_scripts/train_minigrid_ppo.ImgDirFlatWrapper (that is what the net was trained on).
    """
    img = image.astype(np.float32).ravel() / 255.0
    d = np.zeros(4, dtype=np.float32)
    d[int(direction)] = 1.0
    return np.concatenate([img, d])


class MiniGridTabulatedPolicy:
    """The trained OBS-input PPO policy, TABULATED over the (small, enumerable) MiniGrid state
    space so diagnosis is an O(1) dict lookup instead of a neural-net forward pass on every one
    of the Monte-Carlo diagnoser's hundreds of thousands of steps. Mirrors TaxiHardcodedPolicy.

    table[(col,row,dir)] = the greedy action the net outputs for THAT state's observation. Because
    the observation is a deterministic function of the state, this is behaviourally identical to
    model.predict(obs, deterministic=True); aliased states (same view) get the same action, exactly
    as the obs-based net would. predict() receives the raw (col,row,dir) state (the identity
    refiner), so no gen_obs is needed at diagnosis time.
    """
    def __init__(self, table):
        self.table = table

    def predict(self, obs, deterministic=True):
        key = (int(obs[0]), int(obs[1]), int(obs[2]))
        return self.table.get(key, 2), None   # default FORWARD if ever queried off-grid (shouldn't happen)


# cache the tabulated policy per model path so we build it once per process
_MINIGRID_POLICY_CACHE = {}

def build_minigrid_tabulated_policy(model_path, ml_model_name, domain_name):
    """Load the trained SB3 model once and tabulate its greedy action for every interior MiniGrid
    state, returning a MiniGridTabulatedPolicy. Cached by model_path."""
    if model_path in _MINIGRID_POLICY_CACHE:
        return _MINIGRID_POLICY_CACHE[model_path]

    import gymnasium, minigrid  # noqa: F401
    model = models[ml_model_name].load(model_path)
    env = gymnasium.make(domain_name.replace('_', '-'))
    env.reset(seed=0)
    u = env.unwrapped
    table = {}
    for x in range(1, u.width - 1):
        for y in range(1, u.height - 1):
            for d in range(4):
                u.agent_pos = (x, y); u.agent_dir = d
                o = u.gen_obs()
                vec = minigrid_obs_vector(o["image"], o["direction"])
                table[(x, y, d)] = int(model.predict(vec, deterministic=True)[0])
    env.close()

    policy = MiniGridTabulatedPolicy(table)
    _MINIGRID_POLICY_CACHE[model_path] = policy
    print(f"[MiniGridTabulatedPolicy] tabulated {len(table)} states from {model_path}")
    return policy


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

    # MiniGrid Empty: the trained OBS-input PPO policy, tabulated for fast lookup. We load the
    # variant trained at the SAME noise level the env uses (MINIGRID_ACTION_PROB), so the policy
    # is diagnosed under the stochasticity it learned. (The MiniGridEmptyHardcodedPolicy greedy
    # navigator above is kept for reference but no longer used.)
    if domain_name.startswith("MiniGrid"):
        from h_wrappers import MINIGRID_ACTION_PROB
        models_dir = f"environments/{domain_name}/models/{ml_model_name}"
        model_path = f"{models_dir}/{domain_name}__{ml_model_name}__noise{MINIGRID_ACTION_PROB}.zip"
        return build_minigrid_tabulated_policy(model_path, ml_model_name, domain_name)

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