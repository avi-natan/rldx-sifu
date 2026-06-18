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


if __name__ == "__main__":
    redirects = single_redirects()
    swaps = single_swaps()
    pool = execution_fault_pool()

    # sanity checks
    assert len(redirects) == 30, len(redirects)
    assert len(swaps) == 15, len(swaps)
    assert len(pool) == 45
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
