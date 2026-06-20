EPISODES = 1
SEED = 42
DETERMINISTIC = True

# ── Seed-namespacing scheme ───────────────────────────────────────────────────
# Each instance id n owns a contiguous block of SEED_BLOCK integers starting at
# n * SEED_BLOCK, so instances never share random streams. The non-deterministic
# drivers (run_NON_DETERMINSTIC_single_experiment_PO/FO) are PASSED this block base
# (n * SEED_BLOCK) directly and add the offsets below. Within a block:
#   base + 0                       -> trajectory env reset (slips)      (1 seed, numpy/gym)
#   base + FAULT_OFFSET            -> trajectory fault firing           (1 seed, random)
#   base + MASK_OFFSET             -> observation mask                  (1 seed, random)
#   base + CANDIDATE_OFFSET        -> candidate shuffle / random draw   (1 seed, random)
#   base + SIMULATION_OFFSET + t   -> Monte-Carlo trace t               (a RANGE)
# Every discrete single-seed consumer gets its own slot so no two share a stream
# (even the env reset and the fault RNG, though different libraries, are separated).
# The MC slot is a RANGE: one diagnosis walks t = 0 .. total_traces, so it sits well
# above the single-seed slots. Keep SIMULATION_OFFSET + max_tries below SEED_BLOCK so
# a block never bleeds into the next instance: 1_000_000 is safe for epsilon >= ~0.0025;
# use 10_000_000 for tiny epsilon (max_tries can exceed 1M).
SEED_BLOCK = 1_000_000
FAULT_OFFSET = 1
MASK_OFFSET = 2
CANDIDATE_OFFSET = 3
SIMULATION_OFFSET = 10
