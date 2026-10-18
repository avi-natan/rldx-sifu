"""
taxi-v3-easy-benchmark — an EASY Taxi-v4 diagnosis benchmark.

Built to match v2 (hard_taxi_benchmark_v2, class 2) as closely as possible — SAME seeds, SAME
instance structure (true fault E first, then E + 5 a*-maps + 4 b-twins = 10 candidates) — but
eased on BOTH hardness axes by moving the corrupted actions from RARE to FREQUENT:

    v2 class-2                      -> taxi-v3-easy
    a* = RAREST used action (>=2)   -> a* = MOST-used action      (fault fires a lot -> observable)
    b  = 2nd-RAREST action          -> b  = 2nd-MOST-used action  (twins differ from E on a FREQUENT
                                                                   action -> distinguishable)

Why this is easier (and why it's a controlled change, not a rigged one):
  * Observability: a* is commanded many times, so the true fault E actually fires repeatedly ->
    the 5 a*-maps (which differ from E only in how a* behaves) become easy to tell apart.
  * Distinguishability: the 4 twins keep E's a* corruption but differ on b; with b FREQUENT, that
    disagreement is seen many times -> the twins are no longer near-identical to E.
Everything else is identical to v2: same env (rainy 0.7), same table policy, same per-seed healthy
profiling, same E construction (a* -> a seeded-random other action), same 10-candidate layout.

SEED MATCHING: we take the EXACT seeds v2 selects for class 2 (by running v2's own generator) and
build an easy instance on each, so the two benchmarks share their seed set 1:1. Any v2 class-2 seed
is always valid here (it solved and used >=2 actions, so a most-used and 2nd-most-used action exist),
so we expect the same instance count as v2.

=====================================================================================
HOW WE GENERATE THE BENCHMARK  (exact steps -- this is the authoritative description)
=====================================================================================
Env/policy (fixed, same as v2): Taxi-v4 "rainy 0.7"; the hardcoded 500-state table policy
(load_trained_model -> TaxiHardcodedPolicy, tabulated from the promoted Taxi_v4__PPO.zip).
Taxi actions: 0=South 1=North 2=East 3=West 4=Pickup 5=Dropoff. A "fault" is a map m where m[a]
is what ACTUALLY executes when action a is commanded; HEALTHY = identity [0,1,2,3,4,5].

STEP 0 -- seed set. Run v2's own generator for class 2 (hard_taxi_benchmark_v2.build_benchmark,
          classes=(2,)) and take its selected main_seeds. We build one EASY instance per such seed,
          so the easy and hard (v2) benchmarks share their seeds 1:1.

For each seed s:
STEP 1 -- profile. Reset the env at base = s * SEED_BLOCK, roll the HEALTHY policy up to 200 steps,
          and count how many times each of the 6 actions is COMMANDED. (Skip s if it didn't solve.)
STEP 2 -- a* (the corrupted action) = the MOST-used commanded action  (class_picks(...)[3]).
          [v2 class-2 used the RAREST action here -> this is the first easing: the fault fires a lot.]
STEP 3 -- b (the twin axis) = the 2nd-MOST-used commanded action, != a*  (select_b_frequent).
          [v2 used the 2nd-RAREST -> second easing: the twins differ from E on a FREQUENT action.]
STEP 4 -- true fault E. Copy HEALTHY and redirect a* to a seeded-random OTHER action
          (RNG = Random(s*911+3), identical stream to v2) -> E[a*] = e_target.
STEP 5 -- 10 candidates (E listed first; same layout as v2):
            (5) the other maps of a*: HEALTHY with a* -> each of its 5 other targets (one of these
                is healthy-at-a*). Because a* is frequent, these are easy to separate from E.
            (4) b-twins: copy E (so they corrupt a* exactly like E) and additionally redirect b to
                4 distinct other actions. Because b is frequent, these differ visibly from E.
STEP 6 -- record (main_seed, base, 'v3_easy', a*, E, [10 candidates]).

Net effect vs v2 class-2: ONLY the action FREQUENCIES change (a* and b go rare -> frequent), which
eases BOTH hardness axes (observability of the fault, and distinguishability of the candidates).
Seeds, env, policy, E construction, and the 10-candidate layout are all identical to v2.
"""
import random

import gymnasium as gym  # noqa: F401  (registers the patched Taxi-v4)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root

from h_consts import SEED_BLOCK
from h_wrappers import make_wrapped_env
from h_rl_models import load_trained_model
from hard_taxi_benchmark import NUM_ACTIONS, HEALTHY, map_to_str
from hard_taxi_benchmark_v2 import profile_seed, used_actions, class_picks
from hard_taxi_benchmark_v2 import build_benchmark as _build_v2


def select_b_frequent(counts, a_star):
    """2nd-MOST-used commanded action != a* (the EASY twin axis; v2 used 2nd-rarest)."""
    others = [a for a in used_actions(counts) if a != a_star]
    if not others:
        return None
    return max(others, key=lambda a: (counts[a], -a))   # most-used among the rest (seeded-stable)


def build_candidates_easy(a_star, b, seed):
    """Return (E, [10 candidate maps]). Identical construction to v2.build_candidates:
    E corrupts a* to a seeded-random other action; candidates = E + 5 other a*-maps + 4 b-twins."""
    rng = random.Random(seed * 911 + 3)                 # same RNG stream as v2.build_candidates

    e_target = rng.choice([t for t in range(NUM_ACTIONS) if t != a_star])
    E = HEALTHY.copy()
    E[a_star] = e_target

    # all 6 maps of a* (a* -> 0..5); one is E, one is healthy-at-a*, four are other redirects
    a_maps = []
    for t in range(NUM_ACTIONS):
        m = HEALTHY.copy()
        m[a_star] = t
        a_maps.append(m)
    others = [m for m in a_maps if m != E]              # 5 maps

    # 4 twins: keep E's a*, redirect b (now a FREQUENT action) to 4 distinct other actions
    b_targets = [t for t in range(NUM_ACTIONS) if t != b]
    rng.shuffle(b_targets)
    b_variants = []
    for t in b_targets[:4]:
        m = E.copy()
        m[b] = t
        b_variants.append(m)

    candidates = [E] + others + b_variants              # 1 + 5 + 4 = 10
    return E, candidates


def build_benchmark(seeds_per_class=100, verbose=True, max_scan=8000):
    """Return a list of (main_seed, actual_seed, 'v3_easy', a_star, E_str, [cand_strs]) — one per
    v2 class-2 seed, built EASY. Matches v2's seed set 1:1."""
    # 1) get the EXACT seeds v2 selects for class 2
    v2 = _build_v2(seeds_per_class=seeds_per_class, classes=(2,), verbose=False, max_scan=max_scan)
    v2_seeds = [inst[0] for inst in v2]

    # 2) build an easy instance on each of those seeds
    env = make_wrapped_env("Taxi_v4", "rgb_array")      # rainy 0.7 (wrapper default) -- same as v2
    policy = load_trained_model("Taxi_v4", "PPO")        # hardcoded 0.7 table policy -- same as v2

    benchmark = []
    skipped = []
    for main_seed in v2_seeds:
        base = main_seed * SEED_BLOCK
        counts, _length, solved = profile_seed(env, policy, base)
        if not solved:
            skipped.append(main_seed); continue
        a_star = class_picks(counts, main_seed)[3]       # class-3 pick == MOST-used action
        b = select_b_frequent(counts, a_star)
        if a_star is None or b is None:
            skipped.append(main_seed); continue
        E, candidates = build_candidates_easy(a_star, b, main_seed)
        benchmark.append((
            main_seed, base, "v3_easy", a_star,
            map_to_str(E),
            [map_to_str(c) for c in candidates],
        ))
    if verbose:
        print(f"taxi-v3-easy: {len(benchmark)} instances (from {len(v2_seeds)} v2 class-2 seeds; "
              f"skipped {len(skipped)})")
    return benchmark


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    bench = build_benchmark(seeds_per_class=n)
    # also report seed match vs v2
    v2 = _build_v2(seeds_per_class=n, classes=(2,), verbose=False)
    v2_seeds = set(inst[0] for inst in v2)
    easy_seeds = set(inst[0] for inst in bench)
    print(f"v2 class-2 seeds: {len(v2_seeds)} | easy seeds: {len(easy_seeds)} | "
          f"shared: {len(v2_seeds & easy_seeds)} | only-v2: {len(v2_seeds - easy_seeds)}")
    print("sample instance:", bench[0] if bench else None)
