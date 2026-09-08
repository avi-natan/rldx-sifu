"""Sanity test for the custom MiniGrid-SimpleCrossing-S11N2 env.

Checks two things before we build the full integration on it:
  1. REPRODUCIBLE: the same seed always yields the exact same map (walls, agent, goal).
  2. DIVERSE: different seeds yield different barrier/gap layouts, while agent start and goal stay
     FIXED (so the per-seed diversity comes purely from the obstacle/gap moving).

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/crossing_map_test.py
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import gymnasium
import h_wrappers

ENV_ID = "MiniGrid-SimpleCrossing-S11N2-v0"


def layout(env, seed):
    """A hashable signature of the map after reset(seed): agent, goal, and interior wall cells."""
    env.reset(seed=seed)
    u = env.unwrapped
    goal = None
    walls = set()
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None:
                continue
            if c.type == "goal":
                goal = (x, y)
            elif c.type == "wall" and 0 < x < u.width - 1 and 0 < y < u.height - 1:
                walls.add((x, y))   # interior walls = the crossing barriers (border excluded)
    return (tuple(int(v) for v in u.agent_pos), int(u.agent_dir), goal, frozenset(walls))


def gaps(interior_walls, width, height):
    """Interior cells that are NOT walls, per barrier row/col actually used — i.e. the openings."""
    rows = {y for (_, y) in interior_walls}
    cols = {x for (x, _) in interior_walls}
    opens = []
    for y in rows:  # horizontal barrier on this row -> the open interior cells are the gap
        opens += [(x, y) for x in range(1, width - 1) if (x, y) not in interior_walls]
    for x in cols:
        opens += [(x, y) for y in range(1, height - 1) if (x, y) not in interior_walls]
    return sorted(set(opens))


def main():
    h_wrappers._register_minigrid_custom_envs()
    assert ENV_ID in gymnasium.envs.registration.registry, f"{ENV_ID} not registered!"
    print(f"registered: {ENV_ID}\n")
    env = gymnasium.make(ENV_ID, disable_env_checker=True)
    u = env.unwrapped

    # 1) REPRODUCIBLE: same seed -> identical map, 5 times
    sigs = [layout(env, 42) for _ in range(5)]
    same = all(s == sigs[0] for s in sigs)
    print(f"[reproducibility] seed 42 x5 -> identical map: {same}")
    assert same, "same seed produced DIFFERENT maps!"

    # 2) DIVERSE: different seeds -> different barriers/gaps; agent+goal FIXED
    layouts = {s: layout(env, s) for s in range(10)}
    agents = {l[0] for l in layouts.values()}
    dirs = {l[1] for l in layouts.values()}
    goals = {l[2] for l in layouts.values()}
    wallsets = [l[3] for l in layouts.values()]
    distinct_layouts = len({w for w in wallsets})
    print(f"[fixed] agent start(s): {agents}  dir(s): {dirs}  goal(s): {goals}")
    print(f"[diversity] distinct barrier layouts over 10 seeds: {distinct_layouts}/10")
    assert len(agents) == 1 and len(dirs) == 1 and len(goals) == 1, "agent/goal MOVED across seeds!"
    assert distinct_layouts >= 8, f"barriers barely varied ({distinct_layouts}/10)"

    print("\nexample gaps per seed (the openings move):")
    for s in range(4):
        g = gaps(layouts[s][3], u.width, u.height)
        print(f"  seed {s}: {len(layouts[s][3])} interior walls | gaps at {g}")

    print("\nALL CHECKS PASSED: same seed == same map; different seeds => different gaps; agent+goal fixed.")


if __name__ == "__main__":
    main()
