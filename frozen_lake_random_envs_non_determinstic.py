import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from collections import deque

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def is_solvable(desc):
    """Solvable ignoring slipperiness: path exists from S to G without stepping on holes."""
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
                if grid[nx][ny] != 'H':
                    vis.add((nx, ny))
                    q.append((nx, ny))
    return False


def make_valid_frozenlake_env(size, seed=None, p_safe=0.92, is_slippery=True, max_tries=10000):
    """
    Create a random solvable map, and return a FrozenLake env with desired slipperiness.
    """
    base_seed = seed
    for t in range(max_tries):
        cur_seed = None if base_seed is None else (base_seed + t)
        desc = generate_random_map(size=size, p=p_safe, seed=cur_seed)  # list[str]
        if is_solvable(desc):
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
            return env, desc
    raise RuntimeError(f"Could not find a solvable map after {max_tries} tries.")


def risk_averse_reachability_policy(env, tol=1e-12, max_iters=200000):
    """
    Risk-averse policy for stochastic FrozenLake:
    Maximize probability of reaching goal before falling into a hole.

    V[G]=1, V[H]=0
    V[s]=max_a sum_{s'} P(s'|s,a)*V[s']
    """
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n

    # Identify terminal states from the map
    desc = env.unwrapped.desc
    n = desc.shape[0]

    def cell_type(s):
        r, c = divmod(s, n)
        ch = desc[r][c]
        return ch.decode("utf-8") if isinstance(ch, (bytes, np.bytes_)) else ch

    goal_states = set()
    hole_states = set()
    for s in range(nS):
        t = cell_type(s)
        if t == "G":
            goal_states.add(s)
        elif t == "H":
            hole_states.add(s)

    V = np.zeros(nS, dtype=float)
    for s in goal_states:
        V[s] = 1.0
    for s in hole_states:
        V[s] = 0.0

    # Value iteration for reachability probability
    for _ in range(max_iters):
        delta = 0.0
        for s in range(nS):
            if s in goal_states or s in hole_states:
                continue

            best = -1.0
            for a in range(nA):
                val = 0.0
                for p, s2, r, terminated in P[s][a]:
                    # V already encodes terminal success/failure via s2 being in G/H
                    val += p * V[s2]
                if val > best:
                    best = val

            delta = max(delta, abs(best - V[s]))
            V[s] = best

        if delta < tol:
            break

    # Extract greedy policy from V
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        if s in goal_states or s in hole_states:
            pi[s] = 0
            continue
        q = np.zeros(nA, dtype=float)
        for a in range(nA):
            q[a] = sum(p * V[s2] for p, s2, r, terminated in P[s][a])
        pi[s] = int(np.argmax(q))

    return pi, V


def print_map_policy_and_safety(desc, policy, safety=None):
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

    print("\nRisk-averse policy (maximize P(reach G before H)):")
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
                row_out.append(action_to_arrow[int(policy[idx])])
        print(" ".join(row_out))

    if safety is not None:
        print("\nSafety value V(s) = P(reach G before H):")
        for i in range(size):
            row_out = []
            for j in range(size):
                idx = i * size + j
                cell = desc_np[i][j].decode("utf-8") if isinstance(desc_np[i][j], (bytes, np.bytes_)) else desc_np[i][j]
                if cell == 'H':
                    row_out.append("  H   ")
                elif cell == 'G':
                    row_out.append("  G   ")
                else:
                    row_out.append(f"{safety[idx]:0.3f}")
            print(" ".join(row_out))


def generate_maps_and_risk_averse_policies(n=100, seed=42):
    rng = np.random.default_rng(seed)
    small_ratio = 0.2
    pairs = []

    for i in range(n):
        size = 4 if rng.random() < small_ratio else 8
        env_seed = int(rng.integers(0, 2**31 - 1))

        env, desc = make_valid_frozenlake_env(size=size, seed=env_seed, is_slippery=True)
        policy, safety_V = risk_averse_reachability_policy(env)
        env.close()

        pairs.append((desc, policy))
    return pairs


# ---- JSON export/load helpers (same as your style) ----
import json

def policy_array_to_json_dict(policy: np.ndarray) -> dict:
    return {str(int(s)): int(policy[s]) for s in range(len(policy))}

def json_dict_to_policy_dict(d: dict) -> dict:
    return {int(k): int(v) for k, v in d.items()}

def export_pairs_to_json(pairs, json_path: str, meta=None):
    out = {"meta": meta or {}, "pairs": []}
    for i, (desc, policy) in enumerate(pairs):
        size = len(desc)
        nS = size * size
        if len(policy) != nS:
            raise ValueError(f"Pair {i}: policy length {len(policy)} != {nS} for size {size}")
        out["pairs"].append({
            "size": size,
            "desc": desc,
            "policy": policy_array_to_json_dict(policy)
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def load_pairs_from_json(json_path: str):
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


if __name__ == "__main__":
    # Demo on one map
    env, desc = make_valid_frozenlake_env(size=8, seed=10, is_slippery=True, p_safe = 0.92)
    pi, safety_V = risk_averse_reachability_policy(env)
    print_map_policy_and_safety(env.unwrapped.desc, pi, safety=safety_V)
    env.close()

    n = 50
    # Generate dataset
    pairs = generate_maps_and_risk_averse_policies(n=n, seed=42)
    export_pairs_to_json(
        pairs,
        json_path="frozenlake_100_pairs_risk_averse_slippery.json",
        meta={
            "n": n,
            "generator_seed": 42,
            "actions": {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3},
            "is_slippery": True,
            "policy_objective": "maximize P(reach G before H)"
        }
    )
