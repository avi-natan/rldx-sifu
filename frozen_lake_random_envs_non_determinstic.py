"""
Stochastic (slippery) FrozenLake map + policy benchmark generator.

Design:
- ALL maps are 8x8 (no 4x4).
- A difficulty gradient is created by varying `p_safe` (fraction of frozen/safe tiles):
  lower p_safe => more holes => harder navigation.
- The policy for each map is the OPTIMAL DISCOUNTED policy (gamma=0.99) computed by value
  iteration over the *slippery* transition model. This is both slip-aware (avoids holes)
  and goal-directed (discount prefers shorter routes) -- unlike a pure reachability policy,
  which is safe but dawdles and never actually reaches the goal in bounded time.
- A solvability GATE keeps only maps where the optimal policy's exact probability of
  reaching G within `horizon` steps is >= `floor`. This guarantees every map is
  hard-but-solvable (no degenerate/impossible maps).

Reproducibility / variance:
- Everything derives from one master `seed` (default 42). Each of the n maps draws a fresh
  sub-seed from that master RNG, so the n maps are all DIFFERENT (that is the variance),
  while the whole set is reproducible (same seed -> same 100 maps and policies).
"""
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from collections import deque

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# Difficulty gradient over the 100 maps: (p_safe, count). Lower p_safe = more holes = harder.
DEFAULT_TIERS = [(0.92, 40), (0.85, 30), (0.78, 30)]   # 40 easy / 30 medium / 30 hard


def is_solvable(desc):
    """Solvable ignoring slipperiness: a path exists from S to G without stepping on holes."""
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
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in vis:
                if grid[nx][ny] != 'H':
                    vis.add((nx, ny))
                    q.append((nx, ny))
    return False


def make_valid_frozenlake_env(size, seed=None, p_safe=0.92, is_slippery=True, max_tries=10000):
    """Create a random SOLVABLE map, return (env, desc) with the requested slipperiness."""
    base_seed = seed
    for t in range(max_tries):
        cur_seed = None if base_seed is None else (base_seed + t)
        desc = generate_random_map(size=size, p=p_safe, seed=cur_seed)  # list[str]
        if is_solvable(desc):
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
            return env, desc
    raise RuntimeError(f"Could not find a solvable map after {max_tries} tries.")


def value_iteration_policy_slippery(env, gamma=0.99, tol=1e-10):
    """Optimal DISCOUNTED policy over the env's (slippery) transition model.

    Maximizes E[ sum gamma^t r_t ] with FrozenLake's reward (+1 at G, 0 elsewhere; H and G
    terminate). Because reaching a hole ends the episode with no future reward, the optimal
    discounted policy avoids holes; because of discounting it prefers shorter routes -> it is
    both risk-aware and goal-directed.
    """
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            v = max(sum(p * (r + gamma * V[s2] * (not term)) for p, s2, r, term in P[s][a])
                    for a in range(nA))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break

    pi = np.array([
        int(np.argmax([
            sum(p * (r + gamma * V[s2] * (not term)) for p, s2, r, term in P[s][a])
            for a in range(nA)
        ]))
        for s in range(nS)
    ], dtype=int)
    return pi


def finite_horizon_success(env, pi, horizon=200):
    """EXACT probability that following the fixed policy `pi` reaches G within `horizon`
    steps (slippery dynamics). This is what actually matters for a bounded-length trajectory
    -- a policy that only reaches G 'eventually' (dawdling) scores low here, as it should."""
    P = env.unwrapped.P
    nS = env.observation_space.n
    desc = env.unwrapped.desc
    n = desc.shape[0]

    def cell(s):
        r, c = divmod(s, n)
        ch = desc[r][c]
        return ch.decode("utf-8") if isinstance(ch, (bytes, np.bytes_)) else ch

    goal = {s for s in range(nS) if cell(s) == "G"}
    hole = {s for s in range(nS) if cell(s) == "H"}

    p = np.zeros(nS)
    for s in goal:
        p[s] = 1.0
    for _ in range(horizon):
        pnew = p.copy()
        for s in range(nS):
            if s in goal or s in hole:
                continue
            a = int(pi[s])
            pnew[s] = sum(prob * p[s2] for prob, s2, r, term in P[s][a])
        p = pnew
    return float(p[0])   # state 0 = start (S at top-left)


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
                row_out.append(action_to_arrow[int(policy[idx])])
        print(" ".join(row_out))


def _p_safe_schedule(n, tiers):
    """Expand (p_safe, count) tiers into a length-n list; pad the last tier if counts < n."""
    sched = []
    for p_safe, count in tiers:
        sched += [p_safe] * count
    if len(sched) < n:
        sched += [tiers[-1][0]] * (n - len(sched))
    return sched[:n]


def generate_good_maps_and_policies(n=100, seed=42, tiers=DEFAULT_TIERS, floor=0.2,
                                    horizon=200, gamma=0.99, max_resample=400, verbose=False):
    """Generate n solvable 8x8 slippery maps, each with the optimal discounted policy, keeping
    only maps whose optimal policy reaches G within `horizon` steps with prob >= `floor`.

    Returns (pairs, infos):
      pairs[i] = (desc, policy_array)
      infos[i] = {'p_safe', 'opt_success', 'attempts'}
    Fully determined by `seed` (each map draws a fresh sub-seed from the master RNG).
    """
    rng = np.random.default_rng(seed)
    p_safe_list = _p_safe_schedule(n, tiers)

    pairs, infos = [], []
    for i in range(n):
        p_safe = p_safe_list[i]
        chosen = None
        for attempt in range(1, max_resample + 1):
            env_seed = int(rng.integers(0, 2**31 - 1))
            env, desc = make_valid_frozenlake_env(size=8, seed=env_seed, p_safe=p_safe, is_slippery=True)
            pi = value_iteration_policy_slippery(env, gamma=gamma)
            succ = finite_horizon_success(env, pi, horizon=horizon)
            env.close()
            if succ >= floor:
                chosen = (desc, pi, succ, attempt)
                break
        if chosen is None:
            raise RuntimeError(f"map {i} (p_safe={p_safe}): no map >= floor {floor} after {max_resample} tries")
        desc, pi, succ, attempt = chosen
        pairs.append((desc, pi))
        infos.append({"p_safe": p_safe, "opt_success": round(succ, 4), "attempts": attempt})
        if verbose:
            print(f"  map {i:3d}: p_safe={p_safe} opt_success={succ:.3f} (attempts={attempt})")
    return pairs, infos


# ---- JSON export/load helpers ----
import json


def policy_array_to_json_dict(policy: np.ndarray) -> dict:
    return {str(int(s)): int(policy[s]) for s in range(len(policy))}


def json_dict_to_policy_dict(d: dict) -> dict:
    return {int(k): int(v) for k, v in d.items()}


def export_pairs_to_json(pairs, json_path, meta=None, infos=None):
    out = {"meta": meta or {}, "pairs": []}
    for i, (desc, policy) in enumerate(pairs):
        size = len(desc)
        nS = size * size
        if len(policy) != nS:
            raise ValueError(f"Pair {i}: policy length {len(policy)} != {nS} for size {size}")
        entry = {"size": size, "desc": list(desc), "policy": policy_array_to_json_dict(policy)}
        if infos is not None:
            entry.update({"p_safe": infos[i]["p_safe"], "opt_success": infos[i]["opt_success"]})
        out["pairs"].append(entry)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_pairs_from_json(json_path):
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
    N = 100
    SEED = 42
    TIERS = DEFAULT_TIERS
    FLOOR = 0.2
    HORIZON = 200

    pairs, infos = generate_good_maps_and_policies(
        n=N, seed=SEED, tiers=TIERS, floor=FLOOR, horizon=HORIZON, verbose=True)

    succs = [d["opt_success"] for d in infos]
    print(f"\ngenerated {len(pairs)} maps | opt_success: "
          f"min={min(succs):.3f} mean={sum(succs)/len(succs):.3f} max={max(succs):.3f}")

    export_pairs_to_json(
        pairs,
        json_path="frozenlake_100_pairs_risk_averse_slippery.json",
        infos=infos,
        meta={
            "n": N,
            "generator_seed": SEED,
            "actions": {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3},
            "is_slippery": True,
            "size": 8,
            "policy_objective": "optimal discounted return (gamma=0.99) over slippery model",
            "difficulty_tiers_p_safe": TIERS,
            "solvability_floor": FLOOR,
            "solvability_horizon": HORIZON,
        },
    )
    print("wrote frozenlake_100_pairs_risk_averse_slippery.json")
