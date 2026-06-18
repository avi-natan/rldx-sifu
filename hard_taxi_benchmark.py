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

SEEDS = list(range(42, 142))   # 100 seeds: 42..141
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


def build_counts_table(seeds=SEEDS):
    """{seed: (counts, length, solved)} for all seeds. Deterministic/reproducible."""
    from h_wrappers import make_wrapped_env
    from h_rl_models import load_trained_model

    env = make_wrapped_env("Taxi_v4", "rgb_array")
    policy = load_trained_model("Taxi_v4", "PPO")  # hardcoded 500-state table

    table = {}
    for s in seeds:
        table[s] = profile_seed(env, policy, s)
    return table


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

    table = build_counts_table()
    hdr = (f"{'seed':>5} " + " ".join(f"{ACTION_NAMES[a]:>7}" for a in range(NUM_ACTIONS))
           + f" {'len':>5} {'solved':>7}")
    print(hdr)
    print("-" * len(hdr))
    n_solved = 0
    for s in SEEDS:
        counts, length, solved = table[s]
        n_solved += solved
        row = f"{s:>5} " + " ".join(f"{counts[a]:>7}" for a in range(NUM_ACTIONS))
        print(row + f" {length:>5} {('yes' if solved else 'STUCK'):>7}")
    print(f"\nsolved: {n_solved}/{len(SEEDS)}   stuck: {len(SEEDS) - n_solved}/{len(SEEDS)}")

    # freeze: counts + length + solved per seed, importable
    with open("hard_taxi_data.py", "w") as f:
        f.write('"""Frozen hard-Taxi benchmark data (generated by hard_taxi_benchmark.py).\n')
        f.write('Do not edit by hand; regenerate with: python hard_taxi_benchmark.py 2\n"""\n\n')
        f.write("# {seed: {action_id: command_count}} from ONE natural healthy rollout/seed\n")
        f.write("COUNTS_TABLE = {\n")
        for s in SEEDS:
            f.write(f"    {s}: {dict(table[s][0])},\n")
        f.write("}\n\n")
        f.write("# {seed: (trajectory_length, solved)}\n")
        f.write("ROLLOUT_META = {\n")
        for s in SEEDS:
            f.write(f"    {s}: ({table[s][1]}, {table[s][2]}),\n")
        f.write("}\n")
    print("\nFrozen -> hard_taxi_data.py (COUNTS_TABLE + ROLLOUT_META)")


if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "1"
    if step == "1":
        _run_step1()
    elif step == "2":
        _run_step2()
    else:
        print(f"unknown step: {step}")
