"""Test the two reset behaviors: Empty (fixed layout) and SimpleCrossing (per-seed layout).

Empty:   reset(any seed) -> SAME map, but the NOISE stream depends on the seed.
Crossing (our per-seed-layout fix): the FIRST reset fixes the instance layout; EVERY later reset
  keeps that SAME layout, while the NOISE + belief RNG follow the per-call (trace) seed. (Contrast:
  the RAW crossing env changes the map on every different seed -- which is exactly what we prevent.)

Also checks: set_state only lands on NON-WALL cells, and the comparator gives sane equal/not-equal.

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/crossing_reset_test.py
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import gymnasium
import h_wrappers
from h_wrappers import make_wrapped_env
from h_raw_state_comparators import comparators

EMPTY = "MiniGrid_Empty_16x16_v0"
CROSS = "MiniGrid_SimpleCrossing_S11N2_v0"


def grid_sig(env):
    u = env.unwrapped
    walls = frozenset((x, y) for x in range(u.width) for y in range(u.height)
                      if (u.grid.get(x, y) is not None and u.grid.get(x, y).type == "wall"
                          and 0 < x < u.width - 1 and 0 < y < u.height - 1))
    goal = next(((x, y) for x in range(u.width) for y in range(u.height)
                 if u.grid.get(x, y) is not None and u.grid.get(x, y).type == "goal"), None)
    return (walls, goal)


def noise_draw(sim):
    """First draw of the noise RNG after a reset (the SeededStochasticActionWrapper uses this)."""
    return round(float(sim.unwrapped.np_random.random()), 6)


def test_empty():
    print("=== EMPTY (fixed layout) ===")
    sim = make_wrapped_env(EMPTY, "rgb_array")
    sim.reset(seed=1); s1 = grid_sig(sim); n1 = noise_draw(sim)
    sim.reset(seed=2); s2 = grid_sig(sim); n2 = noise_draw(sim)
    print(f"  map same across seeds 1,2: {s1 == s2}   (walls={len(s1[0])}, goal={s1[1]})")
    print(f"  noise differs across seeds: {n1 != n2}   ({n1} vs {n2})")
    assert s1 == s2, "Empty map changed across seeds!"
    assert n1 != n2, "Empty noise did NOT vary with seed!"
    print("  EMPTY OK: fixed map, noise varies with seed.\n")


def test_crossing():
    print("=== CROSSING (per-seed layout, with our fix) ===")
    sim = make_wrapped_env(CROSS, "rgb_array")
    sim.reset(seed=100); L = grid_sig(sim)                     # FIRST reset fixes instance layout
    print(f"  instance layout fixed at seed {sim._layout_seed} (walls={len(L[0])}, goal={L[1]})")
    ok = True
    draws = []
    for trace_seed in (200, 300, 400):
        sim.reset(seed=trace_seed)
        same = grid_sig(sim) == L
        draws.append(noise_draw(sim))
        ok = ok and same
        print(f"  reset(trace seed {trace_seed}) -> same layout: {same}")
    print(f"  layout stays fixed across trace seeds: {ok}")
    print(f"  noise varies per trace seed: {len(set(draws)) == len(draws)}   (draws={draws})")
    assert ok, "Crossing layout CHANGED across trace seeds (fix broken)!"
    assert len(set(draws)) == len(draws), "Crossing noise did NOT vary per trace seed!"

    # contrast: the RAW env DOES change the map per seed (why we need the fix)
    raw = gymnasium.make("MiniGrid-SimpleCrossing-S11N2-v0")
    raw.reset(seed=200); rx = grid_sig(raw)
    raw.reset(seed=300); ry = grid_sig(raw)
    print(f"  [contrast] RAW env changes map per seed: {rx != ry}  <- this is what the fix prevents")
    assert rx != ry

    # different INSTANCE seed -> different layout; same instance seed -> same layout
    sim2 = make_wrapped_env(CROSS, "rgb_array"); sim2.reset(seed=100)
    sim3 = make_wrapped_env(CROSS, "rgb_array"); sim3.reset(seed=777)
    print(f"  new sim, seed 100 -> same layout as instance: {grid_sig(sim2) == L}")
    print(f"  new sim, seed 777 -> different layout:        {grid_sig(sim3) != L}")
    assert grid_sig(sim2) == L and grid_sig(sim3) != L

    # set_state only lands on NON-WALL cells; and it's a valid state in the layout's view map
    sim.reset(seed=200)
    u = sim.unwrapped
    walls = L[0]
    free_states = list(sim._state2view.keys())
    bad = 0
    for st in free_states[:300]:
        sim.set_state(st)
        pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        if pos in walls:
            bad += 1
    print(f"  set_state never lands on a wall: {bad == 0}  (checked 300, wall-hits={bad})")
    assert bad == 0

    # comparator sanity on this layout (uses the ACTIVE per-instance map)
    cmp = comparators[CROSS]
    s = free_states[0]
    print(f"  comparator(s,s)=True: {cmp(s, s)}   (identical state)")
    # two states with the SAME view are equal; a state vs itself-with-different-dir usually differs
    same_view = sim._view_to_states[sim._state2view[s]]
    if len(same_view) >= 2:
        print(f"  comparator(aliased pair)=True: {cmp(same_view[0], same_view[1])}  ({len(same_view)} share this view)")
        assert cmp(same_view[0], same_view[1])
    assert cmp(s, s)
    print("  CROSSING OK: layout fixed per instance, noise varies, set_state wall-safe, comparator works.\n")


if __name__ == "__main__":
    test_empty()
    test_crossing()
    print("ALL RESET TESTS PASSED.")
