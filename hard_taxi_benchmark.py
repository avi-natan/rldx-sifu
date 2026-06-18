"""
Hard Taxi-v4 diagnosis benchmark generator.

Builds the curated, difficulty-graded benchmark described in
references/HARD_TAXI_SPEC.md. Deterministic + seeded so its outputs can be frozen
(hardcoded) into a stable, inspectable benchmark.

Action ids (Gymnasium Taxi): 0=South/DOWN, 1=North/UP, 2=East/RIGHT, 3=West/LEFT,
4=Pickup, 5=Dropoff. Healthy (no-fault) action map = [0,1,2,3,4,5].

A fault mode is a length-6 action map: position = action the policy COMMANDS,
value = action that actually EXECUTES.
"""

import random

NUM_ACTIONS = 6
HEALTHY = [0, 1, 2, 3, 4, 5]
ACTION_NAMES = {0: "DOWN", 1: "UP", 2: "RIGHT", 3: "LEFT", 4: "Pickup", 5: "Dropoff"}


def map_to_str(m):
    """Format an action map as the project's fault-mode string, e.g. '[0,0,2,3,4,5]'."""
    return "[" + ",".join(str(x) for x in m) + "]"


def describe_fault(m):
    """Human-readable description of how a map differs from healthy."""
    diffs = [f"{ACTION_NAMES[a]}->{ACTION_NAMES[m[a]]}" for a in range(NUM_ACTIONS) if m[a] != a]
    return "healthy" if not diffs else ", ".join(diffs)


# ----------------------------------------------------------------------------------
# Step 1: the 45-fault execution pool = 30 single-redirects + 15 single-swaps
# ----------------------------------------------------------------------------------

def single_redirects():
    """One action misfires (a->b, all else identity). 6*5 = 30 maps (Hamming-1)."""
    out = []
    for a in range(NUM_ACTIONS):
        for b in range(NUM_ACTIONS):
            if b == a:
                continue
            m = HEALTHY.copy()
            m[a] = b
            out.append(m)
    return out


def single_swaps():
    """One crossed pair (a<->b). C(6,2) = 15 maps (Hamming-2, bijective)."""
    out = []
    for a in range(NUM_ACTIONS):
        for b in range(a + 1, NUM_ACTIONS):
            m = HEALTHY.copy()
            m[a], m[b] = b, a
            out.append(m)
    return out


def execution_fault_pool():
    """The 45 realistic single-cause execution faults (redirects + swaps)."""
    return single_redirects() + single_swaps()


# ----------------------------------------------------------------------------------
# Step 2: per-seed action-count profile from ONE natural healthy rollout
#   reset(seed=s) -> single deterministic rollout. This is exactly the slip stream
#   the real diagnosis experiment uses for seed s (faithful, not a slip-proxy), and
#   it avoids the stuck-spam contamination that reseeding slips introduced.
#   We count the actions the policy COMMANDS (faults corrupt commanded actions).
# ----------------------------------------------------------------------------------

SEED_START = 42                # scan seeds upward from here
TARGET_SOLVED = 105            # solved-seed buffer (a few may be sparse-skipped later)
TARGET_INSTANCES = 100         # final benchmark size
MAX_ROLLOUT_STEPS = 200


def profile_seed(env, policy, seed, max_steps=MAX_ROLLOUT_STEPS):
    """One natural healthy rollout (reset(seed=s)). Returns (counts, length, solved)."""
    counts = {a: 0 for a in range(NUM_ACTIONS)}
    obs, _ = env.reset(seed=seed)
    steps = 0
    done = False
    term = False
    r = 0.0
    while not done and steps < max_steps:
        a = int(policy.predict(obs)[0])
        counts[a] += 1
        obs, r, term, trunc, _ = env.step(a)
        steps += 1
        done = term or trunc
    solved = bool(term and r > 0)   # Taxi: successful dropoff gives +20 and terminates
    return counts, steps, solved


def build_counts_table(target_solved=TARGET_SOLVED, seed_start=SEED_START):
    """Scan seeds upward from seed_start, profiling each, until `target_solved` SOLVE.
    Returns (table, solved_seeds, skipped_seeds) where table[s] = (counts, length, solved).
    Stuck seeds (policy never completes) are skipped. Deterministic/reproducible."""
    from h_wrappers import make_wrapped_env
    from h_rl_models import load_trained_model

    env = make_wrapped_env("Taxi_v4", "rgb_array")
    policy = load_trained_model("Taxi_v4", "PPO")  # hardcoded 500-state table

    table = {}
    solved_seeds = []
    skipped_seeds = []
    s = seed_start
    while len(solved_seeds) < target_solved:
        counts, length, solved = profile_seed(env, policy, s)
        table[s] = (counts, length, solved)
        (solved_seeds if solved else skipped_seeds).append(s)
        s += 1
    return table, solved_seeds, skipped_seeds


# ----------------------------------------------------------------------------------
# Step 3: per-seed execution-fault selection
#   For each seed, take the top-3 most-used actions; keep 45-pool faults whose faulty
#   action(s) land in that top-3 (so the fault fires a lot -> well-posed); pick one at
#   random, RNG seeded by the seed (reproducible).
# ----------------------------------------------------------------------------------

def faulty_actions(m):
    """Actions where map m differs from healthy (the action(s) the fault corrupts)."""
    return [a for a in range(NUM_ACTIONS) if m[a] != a]


def top_used_actions(counts, k=3):
    """The k most-commanded actions in a seed (count > 0), by count desc, action asc."""
    used = sorted(((a, c) for a, c in counts.items() if c > 0), key=lambda x: (-x[1], x[0]))
    return [a for a, _ in used[:k]]


EXEC_MIN_COUNT = 2   # the corrupted action must be commanded >= this (well-posed evidence)


def select_execution_fault(counts, seed, pool=None, k=3, min_count=EXEC_MIN_COUNT):
    """Pick one execution fault for a seed. Eligible faults are those with at least one
    corrupted action that is (a) among the seed's top-k used actions AND (b) commanded
    >= min_count times -> so the fault fires enough to be well-posed. Pick uniformly at
    random, RNG seeded by `seed` (reproducible). Returns the action map."""
    if pool is None:
        pool = execution_fault_pool()
    top = top_used_actions(counts, k)
    eligible_actions = {a for a in top if counts.get(a, 0) >= min_count}
    productive = [m for m in pool if set(faulty_actions(m)) & eligible_actions]
    assert productive, f"no eligible fault for seed {seed} (eligible_actions={eligible_actions})"
    return random.Random(seed).choice(productive)


def build_execution_faults(counts_table, solved_seeds, k=3):
    """{seed: execution-fault map} for all benchmark seeds."""
    pool = execution_fault_pool()
    return {s: select_execution_fault(counts_table[s], s, pool, k) for s in solved_seeds}


# ----------------------------------------------------------------------------------
# Step 4: per-seed graded candidate set (10 = E + 9 distractors)
#   score(C) = sum of count[a] over actions where C disagrees with E (low = hard).
#   7 NEAR-TWINS: agree with E on f, differ on ONE used non-f action -> 4 hard / 2 med
#     / 1 easy, tiered by that action's count (lowest nonzero = hardest).
#   2 ALTERNATIVE faults: healthy except corrupt a heavily-used action (disagree with
#     E on f) -> easy (test rejection).
# ----------------------------------------------------------------------------------

def candidate_score(counts, E, C):
    """Frequency-weighted disagreement between C and E on this seed."""
    return sum(counts.get(a, 0) for a in range(NUM_ACTIONS) if C[a] != E[a])


def build_candidate_set(counts, E, seed):
    """Return {'execution', 'hard', 'medium', 'easy', 'alternative'} maps, or None if
    the seed is too sparse to build 9 distinct distractors."""
    fa = set(faulty_actions(E))                       # E's faulty action(s)
    used_non_f = sorted([a for a in range(NUM_ACTIONS)
                         if counts.get(a, 0) > 0 and a not in fa],
                        key=lambda a: (counts[a], a))  # ascending by count, nonzero only
    if len(used_non_f) < 2:
        return None

    rng = random.Random(seed * 7919 + 13)
    seen = {map_to_str(E), map_to_str(HEALTHY)}

    def fresh_near_twin(action):
        """A new distinct near-twin that keeps E's corruption but edits `action`."""
        targets = [t for t in range(NUM_ACTIONS) if t != action]
        rng.shuffle(targets)
        for t in targets:
            C = E.copy(); C[action] = t
            if map_to_str(C) not in seen:
                seen.add(map_to_str(C)); return C
        return None

    def fresh_alternative(action):
        """A new distinct alternative fault: healthy except `action` corrupted."""
        targets = [t for t in range(NUM_ACTIONS) if t != action]
        rng.shuffle(targets)
        for t in targets:
            A = HEALTHY.copy(); A[action] = t
            if map_to_str(A) not in seen:
                seen.add(map_to_str(A)); return A
        return None

    low = used_non_f[0]                       # cheapest used non-f action  -> hard
    mid = used_non_f[len(used_non_f) // 2]    # middle                      -> medium
    high = used_non_f[-1]                      # priciest                    -> easy
    heavy = list(reversed(used_non_f))         # highest count first (for alternatives)

    out = {"execution": E, "hard": [], "medium": [], "easy": [], "alternative": []}
    for _ in range(4):
        out["hard"].append(fresh_near_twin(low))
    for _ in range(2):
        out["medium"].append(fresh_near_twin(mid))
    out["easy"].append(fresh_near_twin(high))
    for h in heavy[:2]:
        out["alternative"].append(fresh_alternative(h))

    # if anything came back None we ran out of distinct maps -> too sparse
    if any(c is None for k in ("hard", "medium", "easy", "alternative") for c in out[k]):
        return None
    return out


# ----------------------------------------------------------------------------------
# Step 5: assemble the full benchmark and freeze it
#   100 tuples (seed, execution fault, 10 candidate fault modes [E + 9 distractors])
#   + global EXECUTION_FAULT_POOL (45) + DISTRACTORS (union used) + COUNTS_TABLE.
# ----------------------------------------------------------------------------------

def build_benchmark(counts_table, solved_seeds, target=TARGET_INSTANCES):
    """Return (benchmark, distractor_union, skipped_sparse).
    benchmark = [(seed, E_str, [10 candidate strs])]; the execution fault is candidate[0].
    Builds up to `target` instances, skipping seeds too sparse for 9 distinct distractors."""
    pool = execution_fault_pool()
    benchmark = []
    distractor_union = set()
    skipped_sparse = []
    for s in solved_seeds:
        if len(benchmark) >= target:
            break
        counts = counts_table[s]
        E = select_execution_fault(counts, s, pool)
        cs = build_candidate_set(counts, E, s)
        if cs is None:
            skipped_sparse.append(s)
            continue
        distractors = cs["hard"] + cs["medium"] + cs["easy"] + cs["alternative"]
        candidates = [E] + distractors                       # 10, true fault first
        benchmark.append((s, map_to_str(E), [map_to_str(c) for c in candidates]))
        distractor_union.update(map_to_str(c) for c in distractors)
    return benchmark, sorted(distractor_union), skipped_sparse


def _run_step1():
    redirects, swaps, pool = single_redirects(), single_swaps(), execution_fault_pool()
    assert len(redirects) == 30 and len(swaps) == 15 and len(pool) == 45
    as_strings = [map_to_str(m) for m in pool]
    assert len(set(as_strings)) == 45, "duplicate fault maps"
    assert map_to_str(HEALTHY) not in as_strings, "healthy must not be in the fault pool"
    print(f"Execution-fault pool: {len(pool)} faults "
          f"({len(redirects)} redirects + {len(swaps)} swaps)\n")
    print("--- 30 single-redirects ---")
    for m in redirects:
        print(f"  {map_to_str(m):<18} {describe_fault(m)}")
    print("\n--- 15 single-swaps ---")
    for m in swaps:
        print(f"  {map_to_str(m):<18} {describe_fault(m)}")


def _run_step2():
    from h_wrappers import make_wrapped_env
    from h_rl_models import load_trained_model

    env = make_wrapped_env("Taxi_v4", "rgb_array")
    policy = load_trained_model("Taxi_v4", "PPO")

    # --- reproducibility check ---
    a1 = profile_seed(env, policy, 42)
    a2 = profile_seed(env, policy, 42)
    print(f"== reproducible (seed 42 same on re-run): {a1 == a2} ==\n")
    assert a1 == a2, "counts must be reproducible"

    table, solved_seeds, skipped_seeds = build_counts_table()
    hdr = (f"{'seed':>5} " + " ".join(f"{ACTION_NAMES[a]:>7}" for a in range(NUM_ACTIONS))
           + f" {'len':>5} {'solved':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in sorted(table):
        counts, length, solved = table[s]
        row = f"{s:>5} " + " ".join(f"{counts[a]:>7}" for a in range(NUM_ACTIONS))
        print(row + f" {length:>5} {('yes' if solved else 'STUCK'):>7}")
    print(f"\nsolved kept: {len(solved_seeds)} (seeds {solved_seeds[0]}..{solved_seeds[-1]})"
          f"   skipped (stuck): {skipped_seeds}")

    # freeze: only the 100 solved benchmark seeds + the hardcoded skip list
    with open("hard_taxi_data.py", "w") as f:
        f.write('"""Frozen hard-Taxi benchmark data (generated by hard_taxi_benchmark.py).\n')
        f.write('Do not edit by hand; regenerate with: python hard_taxi_benchmark.py 2\n"""\n\n')
        f.write("# stuck seeds (policy never completed) excluded from the benchmark\n")
        f.write(f"SKIPPED_SEEDS = {skipped_seeds}\n\n")
        f.write(f"# the {len(solved_seeds)} benchmark seeds (all solved cleanly)\n")
        f.write(f"SOLVED_SEEDS = {solved_seeds}\n\n")
        f.write("# {seed: {action_id: command_count}} from ONE natural healthy rollout/seed\n")
        f.write("COUNTS_TABLE = {\n")
        for s in solved_seeds:
            f.write(f"    {s}: {dict(table[s][0])},\n")
        f.write("}\n\n")
        f.write("# {seed: (trajectory_length, solved)}\n")
        f.write("ROLLOUT_META = {\n")
        for s in solved_seeds:
            f.write(f"    {s}: ({table[s][1]}, {table[s][2]}),\n")
        f.write("}\n")
    print("\nFrozen -> hard_taxi_data.py (SKIPPED_SEEDS + SOLVED_SEEDS + COUNTS_TABLE + ROLLOUT_META)")


def _run_step3():
    from hard_taxi_data import COUNTS_TABLE, SOLVED_SEEDS
    pool = execution_fault_pool()
    exec_faults = build_execution_faults(COUNTS_TABLE, SOLVED_SEEDS)

    print(f"{'seed':>5}  {'execution fault':<18} {'corrupts':<26} top-3 used")
    print("-" * 78)
    for s in SOLVED_SEEDS:
        m = exec_faults[s]
        top = [ACTION_NAMES[a] for a in top_used_actions(COUNTS_TABLE[s], 3)]
        print(f"{s:>5}  {map_to_str(m):<18} {describe_fault(m):<26} {top}")

    # quick distribution sanity: which fault TYPES got chosen
    n_redirect = sum(1 for s in SOLVED_SEEDS if len(faulty_actions(exec_faults[s])) == 1)
    n_swap = len(SOLVED_SEEDS) - n_redirect
    print(f"\nchosen: {n_redirect} redirects + {n_swap} swaps (of {len(SOLVED_SEEDS)} seeds)")
    # reproducibility
    again = build_execution_faults(COUNTS_TABLE, SOLVED_SEEDS)
    print(f"reproducible: {again == exec_faults}")


def _show_candidate_set(seed, counts, E):
    cs = build_candidate_set(counts, E, seed)
    fa = [ACTION_NAMES[a] for a in faulty_actions(E)]
    print(f"\n=== seed {seed} ===")
    print(f"  counts: { {ACTION_NAMES[a]: counts[a] for a in range(NUM_ACTIONS)} }")
    print(f"  execution fault E = {map_to_str(E):<18} corrupts {describe_fault(E)}  (f = {fa})")
    if cs is None:
        print("  TOO SPARSE -> skip"); return
    for tier in ("hard", "medium", "easy", "alternative"):
        for C in cs[tier]:
            agree_f = all(C[a] == E[a] for a in faulty_actions(E))
            print(f"  {tier:<11} {map_to_str(C):<18} score={candidate_score(counts, E, C):<3} "
                  f"agrees_on_f={agree_f}  ({describe_fault(C)})")


def _run_step4():
    from hard_taxi_data import COUNTS_TABLE, SOLVED_SEEDS
    pool = execution_fault_pool()
    # inspect two representative seeds: 42 (good spread) and 88 (sparse)
    for s in (42, 88):
        E = select_execution_fault(COUNTS_TABLE[s], s, pool)
        _show_candidate_set(s, COUNTS_TABLE[s], E)


def _run_step5():
    from hard_taxi_data import COUNTS_TABLE, SOLVED_SEEDS, SKIPPED_SEEDS, ROLLOUT_META

    benchmark, distractors, skipped_sparse = build_benchmark(COUNTS_TABLE, SOLVED_SEEDS)
    # reproducibility
    again, _, _ = build_benchmark(COUNTS_TABLE, SOLVED_SEEDS)
    assert again == benchmark, "benchmark must be reproducible"

    pool_strs = [map_to_str(m) for m in execution_fault_pool()]
    print(f"benchmark instances: {len(benchmark)}   sparse-skipped: {skipped_sparse}")
    print(f"execution pool: {len(pool_strs)}   distinct distractors used: {len(distractors)}")
    print("reproducible: True")
    s, E, cands = benchmark[0]
    print(f"\nexample tuple -> seed {s}, E={E}, candidates:")
    for c in cands:
        print(f"    {c}")

    # freeze the complete benchmark
    with open("hard_taxi_data.py", "w") as f:
        f.write('"""Frozen hard-Taxi benchmark (generated by hard_taxi_benchmark.py).\n')
        f.write('Do not edit by hand; regenerate with:\n')
        f.write('  python hard_taxi_benchmark.py 2   # counts (needs rollouts)\n')
        f.write('  python hard_taxi_benchmark.py 5   # assemble faults + candidates\n"""\n\n')
        f.write(f"SKIPPED_SEEDS = {SKIPPED_SEEDS}\n\n")
        f.write(f"SOLVED_SEEDS = {SOLVED_SEEDS}\n\n")
        f.write("# the 45 realistic execution faults (30 redirects + 15 swaps)\n")
        f.write("EXECUTION_FAULT_POOL = [\n")
        for x in pool_strs:
            f.write(f'    "{x}",\n')
        f.write("]\n\n")
        f.write(f"# all distinct distractor maps used across the benchmark ({len(distractors)})\n")
        f.write("DISTRACTORS = [\n")
        for x in distractors:
            f.write(f'    "{x}",\n')
        f.write("]\n\n")
        f.write("# {seed: {action_id: command_count}} from one natural healthy rollout/seed\n")
        f.write("COUNTS_TABLE = {\n")
        for s in SOLVED_SEEDS:
            f.write(f"    {s}: {dict(COUNTS_TABLE[s])},\n")
        f.write("}\n\n")
        f.write("# {seed: (trajectory_length, solved)}\n")
        f.write("ROLLOUT_META = {\n")
        for s in SOLVED_SEEDS:
            f.write(f"    {s}: {tuple(ROLLOUT_META[s])},\n")
        f.write("}\n\n")
        f.write("# 100 instances: (seed, execution_fault, [10 candidate fault modes; E first])\n")
        f.write("BENCHMARK = [\n")
        for s, E, cands in benchmark:
            f.write(f"    ({s}, \"{E}\", {cands}),\n")
        f.write("]\n")
    print("\nFrozen -> hard_taxi_data.py (BENCHMARK + EXECUTION_FAULT_POOL + DISTRACTORS + COUNTS_TABLE + ROLLOUT_META)")


if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "1"
    if step == "1":
        _run_step1()
    elif step == "2":
        _run_step2()
    elif step == "3":
        _run_step3()
    elif step == "4":
        _run_step4()
    elif step == "5":
        _run_step5()
    else:
        print(f"unknown step: {step}")
