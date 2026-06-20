EPISODES = 1
SEED = 42
DETERMINISTIC = True

# ── Seed-namespacing scheme ───────────────────────────────────────────────────
# Terminology: `seed` is the small human id (42, 43, ...); `base_seed` is seed * SEED_BLOCK.
# Each instance owns a contiguous block of SEED_BLOCK integers starting at base_seed, so
# instances never share random streams. The non-deterministic drivers
# (run_NON_DETERMINSTIC_single_experiment_PO/FO) are PASSED base_seed directly.
#
# Within a block, a "retry window" of WINDOW_SIZE consecutive seeds holds the four
# BEFORE-diagnosis sub-streams. Attempt r (r=0 on the first try; r>0 when a seed is
# "not good" and the trajectory is retried) shifts ALL four by r*WINDOW_SIZE, so each
# attempt is a fully disjoint instance (no overlap between attempts). RETRY_WINDOWS
# reserves how many attempts a seed may use. The seed for attempt r is:
#     base_seed + r*WINDOW_SIZE + <within-window offset>
#   offset 0  -> trajectory env reset (slips)      numpy/gym
#   offset 1  -> trajectory fault firing           random        (FAULT_OFFSET)
#   offset 2  -> observation mask                  random        (MASK_OFFSET)
#   offset 3  -> candidate shuffle / random draw   random        (CANDIDATE_OFFSET)
# The Monte-Carlo trace RANGE sits ABOVE every retry window:
#     base_seed + SIMULATION_OFFSET + t        (t = 0 .. total_traces)
# Keep SIMULATION_OFFSET + max_tries below SEED_BLOCK so a block never bleeds into the
# next instance: 1_000_000 is safe for epsilon >= ~0.0025; use 10_000_000 for tiny eps.
# NOTE (current code): only attempt r=0 is wired (offsets used directly, no r term).
# The r>0 retry plumbing is reserved here but not yet threaded through execute / run_PO /
# the diagnoser init reset -- see [[seeding-namespace-redesign]].
SEED_BLOCK = 1_000_000
WINDOW_SIZE = 4
RETRY_WINDOWS = 100
TRAJECTORY_OFFSET = 0
FAULT_OFFSET = 1
MASK_OFFSET = 2
CANDIDATE_OFFSET = 3
SIMULATION_OFFSET = WINDOW_SIZE * RETRY_WINDOWS   # = 400, above all retry windows

# MC gap-decorrelation stride. A diagnosis gap starting at observed index L uses MC seeds
#   base + SIMULATION_OFFSET + L + t*MAX_STATES   (t = 0,1,2,... over traces)
# so each gap sits on its own residue class mod MAX_STATES (gaps never collide), while the
# seed is independent of candidate/rate (candidates share -> CRN). MAX_STATES must be >= the
# longest possible trajectory (= max_exec_len) so every gap-start index L is < MAX_STATES.
# Worst case is 100% visibility: at most max_exec_len observed states. Keep
# SIMULATION_OFFSET + MAX_STATES + max_tries*MAX_STATES < SEED_BLOCK (use 10_000_000 for
# small epsilon, where max_tries grows ~ (0.025/epsilon)^2).
MAX_STATES = 200
