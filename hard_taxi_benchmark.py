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
TARGET_SOLVED = 100            # keep the first 100 that solve; skip stuck ones
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


def select_execution_fault(counts, seed, pool=None, k=3):
    """Pick one execution fault whose faulty action(s) are among the seed's top-k used
    actions. Random, RNG seeded by `seed` (reproducible). Returns the action map."""
    if pool is None:
        pool = execution_fault_pool()
    top = set(top_used_actions(counts, k))
    productive = [m for m in pool if set(faulty_actions(m)) & top]
    assert productive, f"no productive fault for seed {seed} (top={top})"
    return random.Random(seed).choice(productive)


def build_execution_faults(counts_table, solved_seeds, k=3):
    """{seed: execution-fault map} for all benchmark seeds."""
    pool = execution_fault_pool()
    return {s: select_execution_fault(counts_table[s], s, pool, k) for s in solved_seeds}


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


if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "1"
    if step == "1":
        _run_step1()
    elif step == "2":
        _run_step2()
    elif step == "3":
        _run_step3()
    else:
        print(f"unknown step: {step}")
