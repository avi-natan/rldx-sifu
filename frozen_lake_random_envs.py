import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from collections import deque

def is_solvable(desc):
    if isinstance(desc, np.ndarray):
        grid = [[c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c) for c in row] for row in desc]
    else:
        grid = [list(row) for row in desc]

    n = len(grid)
    start = (0, 0)
    goal = (n - 1, n - 1)

    if grid[start[0]][start[1]] != 'S':
        return False
    if grid[goal[0]][goal[1]] != 'G':
        return False

    q = deque([start])
    vis = {start}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in vis:
                if grid[nx][ny] != 'H':   # only holes are blocked
                    vis.add((nx, ny))
                    q.append((nx, ny))
    return False


def make_valid_frozenlake_env(size, seed=None):
    base_seed = seed
    max_tries = 10000

    for t in range(max_tries):
        cur_seed = None if base_seed is None else (base_seed + t)
        desc = generate_random_map(size=size, p=0.85, seed=cur_seed)  # list[str]

        if is_solvable(desc):
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
            return env, desc

    raise RuntimeError(f"Could not find a solvable map after {max_tries} tries.")


def value_iteration_policy(env):
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n
    gamma = 0.99
    tol = 1e-10

    V = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            q_sa = []
            for a in range(nA):
                q_sa.append(sum(p * (r + gamma * V[s2] * (not term))
                                for p, s2, r, term in P[s][a]))
            v_new = max(q_sa)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < tol:
            break

    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        pi[s] = int(np.argmax([
            sum(p * (r + gamma * V[s2] * (not term))
                for p, s2, r, term in P[s][a])
            for a in range(nA)
        ]))
    return pi

def print_map_and_policy(desc, policy):
    # desc is list[str] or np.ndarray
    if isinstance(desc, list):
        desc_np = np.array([list(row) for row in desc])
    else:
        desc_np = desc

    size = desc_np.shape[0]
    action_to_arrow = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    print(f"Size: {size}\n")
    print("Map:")
    for row in desc_np:
        print("".join(c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else c for c in row))

    print("\nPolicy:")
    for i in range(size):
        row_out = []
        for j in range(size):
            idx = i * size + j
            cell = desc_np[i][j].decode("utf-8") if isinstance(desc_np[i][j], (bytes, np.bytes_)) else desc_np[i][j]
            if cell == 'H':
                row_out.append('H')
            elif cell == 'G':
                row_out.append('G')
            else:
                row_out.append(action_to_arrow[policy[idx]])
        print(" ".join(row_out))


def generate_maps_and_policies(n=100, seed=42):
    rng = np.random.default_rng(seed)
    small_ratio = 0.2
    pairs = []

    for i in range(n):
        size = 4 if rng.random() < small_ratio else 8

        # derive per-map seed deterministically from the master seed
        env_seed = int(rng.integers(0, 2**31 - 1))

        env, desc = make_valid_frozenlake_env(size=size, seed=env_seed)
        policy = value_iteration_policy(env)
        env.close()

        pairs.append((desc, policy))

    return pairs




import json
import numpy as np

# actions
LEFT  = 0
DOWN  = 1
RIGHT = 2
UP    = 3

def policy_array_to_json_dict(policy: np.ndarray) -> dict:
    # JSON keys must be strings, so convert state index to str
    return {str(int(s)): int(policy[s]) for s in range(len(policy))}

def json_dict_to_policy_dict(d: dict) -> dict:
    # convert back to int keys
    return {int(k): int(v) for k, v in d.items()}

def export_pairs_to_json(pairs, json_path: str, meta=None):
    """
    pairs: list of (desc, policy_array)
      desc: list[str]
      policy_array: np.ndarray shape (nS,)
    """
    out = {
        "meta": meta or {},
        "pairs": []
    }

    for i, (desc, policy) in enumerate(pairs):
        size = len(desc)
        nS = size * size
        if len(policy) != nS:
            raise ValueError(f"Pair {i}: policy length {len(policy)} != {nS} for size {size}")

        out["pairs"].append({
            "size": size,
            "desc": desc,  # EXACTLY like FROZENLAKE_DESC
            "policy": policy_array_to_json_dict(policy)  # { "0": 1, "1": 2, ... }
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def load_pairs_from_json(json_path: str):
    """
    Returns list of (FROZENLAKE_DESC, frozen_lake_hard_coded_policy)
      FROZENLAKE_DESC: list[str]
      frozen_lake_hard_coded_policy: dict[int -> int]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for i, item in enumerate(data["pairs"]):
        desc = item["desc"]
        policy = json_dict_to_policy_dict(item["policy"])

        size = item.get("size", len(desc))
        nS = size * size
        if len(desc) != size:
            raise ValueError(f"Pair {i}: desc length {len(desc)} != size {size}")
        if len(policy) != nS:
            raise ValueError(f"Pair {i}: policy size {len(policy)} != {nS}")

        pairs.append((desc, policy))

    return pairs

# ---- example usage with your existing function ----
if __name__ == "__main__":
    # your function already returns list[(desc:list[str], policy:np.ndarray)]
    pairs = generate_maps_and_policies(n=100, seed=42)

    export_pairs_to_json(
        pairs,
        json_path="frozenlake_100_pairs.json",
        meta={
            "n": 100,
            "generator_seed": 42,
            "actions": {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3},
            "is_slippery": False,
        }
    )

    loaded = load_pairs_from_json("frozenlake_100_pairs.json")

    # now you have EXACT shapes you wanted:
    FROZENLAKE_DESC = loaded[0][0]
    frozen_lake_hard_coded_policy = loaded[0][1]

    print(FROZENLAKE_DESC)
    print(list(frozen_lake_hard_coded_policy.items())[:8])
    print_map_and_policy(FROZENLAKE_DESC, frozen_lake_hard_coded_policy)
    print("ffffffffffffff")
    print(loaded[100])

"""

experiment 2:
map 8x8 hard coded policy , used fault [0,3,2,3]
all with fault prob of 0.3 -> we get 22 states until we get to goal
fault_probability = 0.3
instance_seed = 10
possible_fault_mode_names = [
"[0,0,2,3]",
"[2,1,2,3]",
"[0,3,2,1]",
"[0,0,0,3]",
"[0,0,2,0]",
"[0,1,0,0]",
"[0,0,0,0]",
"[0,2,2,1]",
"[2,1,0,3]",
"[1,0,2,3]",
"[0,3,2,3]"
]

100% visible -> 2 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

80% visible -> 3 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

60% visible -> 4ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

30% visible -> 68 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

20% visible ->  783 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

10% visible -> 9171 ms = 9 sec
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

0% visible -> 325303 ms = 325 sec
final diagnoses: ('[0,1,0,0]', '[2,1,0,3]', '[0,2,2,1]', '[1,0,2,3]', '[0,0,2,3]', '[0,3,2,1]', '[0,3,2,3]', '[0,0,0,3]', '[0,0,0,0]')
Wrong
"""
"""

experiment 1:
map 8x8 hard coded policy , used fault [0,3,2,3]
all with fault prob of 0.5 -> we get 129 states until we get to goal
fault_probability = 0.5
instance_seed = 10
possible_fault_mode_names = [
"[0,0,2,3]",
"[2,1,2,3]",
"[0,3,2,1]",
"[0,0,0,3]",
"[0,0,2,0]",
"[0,1,0,0]",
"[0,0,0,0]",
"[0,2,2,1]",
"[2,1,0,3]",
"[1,0,2,3]",
"[0,3,2,3]"
]

100% visible -> 6 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

95% visible -> 22 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

90% visible -> 118 ms
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

85% visible -> 1567 ms = 1.5 sec
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

80% visible ->  25532 ms = 25 sec
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

75% visible -> 224475 ms = 224 sec = 3.7 min 
final diagnoses: ('[0,3,2,3]', '[0,3,2,1]') Correct

70% visible -> 20 min stopped it without getting results
"""