EPISODES = 1
SEED = 42
DETERMINISTIC = True

# ── Seed-namespacing scheme ───────────────────────────────────────────────────
# Each instance id n owns a contiguous block of SEED_BLOCK integers starting at
# n * SEED_BLOCK, so instances never share random streams. Within a block:
#   n*SEED_BLOCK + 0                 -> trajectory (before diagnosis)
#   n*SEED_BLOCK + MASK_OFFSET       -> observation mask (before diagnosis)
#   n*SEED_BLOCK + SIMULATION_OFFSET + t -> Monte-Carlo trace t (during diagnosis)
# Tune SEED_BLOCK to widen each instance's space (must exceed the largest per-call
# max_tries; 1_000_000 is safe for epsilon >= ~0.0025, use 10_000_000 for tiny eps).
SEED_BLOCK = 1_000_000
MASK_OFFSET = 1
SIMULATION_OFFSET = 2
