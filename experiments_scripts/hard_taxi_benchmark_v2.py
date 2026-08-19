"""
Hard Taxi-v4 diagnosis benchmark generator v2 — class-based, rainy 0.7.

Produces (up to) 200 instances = 4 difficulty classes x `seeds_per_class` (default 50),
each a tuple:
    (main_seed, actual_seed, class, chosen_action, execution_fault, [10 candidate maps])
- actual_seed = main_seed * SEED_BLOCK   (the instance block base; env reset uses slot 0)
- the execution fault E corrupts ONE commanded action `a*` (chosen per class), firing it to a
  RANDOM other action; the 10 candidates are E + 5 other maps of a* (all 6 maps of a*) + 4
  "tier-2" twins that keep E's a* and instead corrupt the 2nd-rarest action b.

Classes (by how often a* is commanded -> how often E fires -> hardness):
    2  least-used with count >= 2    (rarest movement)                 -> hardest
    3  most-used action                                                -> easiest
    4  random from what remains (excludes 1/2/3); drop if nothing left
(Former class 1 = least-used action was dropped: it is almost always Pickup/Dropoff, whose
corruption re-converges -> impossible, not hard. Its action is still excluded from class 4's
pick. Class ids kept as 2/3/4 so each id keeps its definition.)

Policy: the hardcoded 0.7-rainy table policy (load_trained_model -> TaxiHardcodedPolicy,
tabulated from the promoted Taxi_v4__PPO.zip). Profiling rolls the HEALTHY policy out at the
instance's block-base env seed (= what the runtime trajectory uses) and counts commanded
actions; the firing gate itself is a RUNTIME drop (the fault-offset sweep in p_pipeline).
"""
import random

import gymnasium as gym  # noqa: F401  (registers the patched Taxi-v4)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (script lives in experiments/)

from h_consts import SEED_BLOCK
from h_wrappers import make_wrapped_env
from h_rl_models import load_trained_model
from hard_taxi_benchmark import NUM_ACTIONS, HEALTHY, ACTION_NAMES, map_to_str, describe_fault

CLASSES = (2, 3, 4)   # class 1 (least-used = Pickup/Dropoff, re-converges -> impossible) dropped


# ---------------------------------------------------------------------------------------
# Profiling: one HEALTHY rollout at the instance's block-base seed -> action counts
# ---------------------------------------------------------------------------------------
def profile_seed(env, policy, env_seed, max_steps=200):
    """Roll the deterministic healthy policy from reset(seed=env_seed). Returns
    (counts, length, solved). env_seed must be the block base so the counts match the
    slip stream the real diagnosis trajectory uses for this instance."""
    counts = {a: 0 for a in range(NUM_ACTIONS)}
    obs, _ = env.reset(seed=env_seed)
    steps = 0
    term = trunc = False
    r = 0.0
    while not (term or trunc) and steps < max_steps:
        a = int(policy.predict(obs)[0])
        counts[a] += 1
        obs, r, term, trunc, _ = env.step(a)
        steps += 1
    solved = bool(term and r > 0)   # Taxi: +20 dropoff terminates successfully
    return counts, steps, solved


def used_actions(counts):
    return [a for a in range(NUM_ACTIONS) if counts[a] > 0]


# ---------------------------------------------------------------------------------------
# Per-seed class action picks (deterministic per seed, so class-4 can exclude 1/2/3)
# ---------------------------------------------------------------------------------------
def class_picks(counts, seed):
    """All four class action picks for one seed, as {class_id: action or None}.
    Deterministic per seed (seeded tie-breaks), so class 4 can exclude 1/2/3's picks."""
    used = used_actions(counts)
    if not used:
        return {c: None for c in CLASSES}
    rng = random.Random(seed * 100 + 7)

    mn = min(counts[a] for a in used)
    c1 = rng.choice([a for a in used if counts[a] == mn])                # least used

    ge2 = [a for a in used if counts[a] >= 2]
    if ge2:
        m2 = min(counts[a] for a in ge2)
        c2 = rng.choice([a for a in ge2 if counts[a] == m2])            # least used, count>=2
    else:
        c2 = None

    mx = max(counts[a] for a in used)
    c3 = rng.choice([a for a in used if counts[a] == mx])               # most used

    remaining = [a for a in used if a not in {c1, c2, c3}]
    c4 = rng.choice(remaining) if remaining else None                  # random remainder
    return {1: c1, 2: c2, 3: c3, 4: c4}


def select_b(counts, a_star):
    """2nd-rarest commanded action != a* (the tier-2 twin axis). None if no other used."""
    others = [a for a in used_actions(counts) if a != a_star]
    if not others:
        return None
    return min(others, key=lambda a: (counts[a], a))


# ---------------------------------------------------------------------------------------
# Candidate set: 6 maps of a* (E + 5) + 4 tier-2 twins on b  = 10, E first, all distinct
# ---------------------------------------------------------------------------------------
def build_candidates(a_star, b, seed):
    """Return (E, [10 candidate maps]). E corrupts a* to a random other action."""
    rng = random.Random(seed * 911 + 3)

    e_target = rng.choice([t for t in range(NUM_ACTIONS) if t != a_star])
    E = HEALTHY.copy()
    E[a_star] = e_target

    # all 6 maps of a* (a*->0..5); one is E, one is healthy-at-a*, four are other redirects
    a_maps = []
    for t in range(NUM_ACTIONS):
        m = HEALTHY.copy()
        m[a_star] = t
        a_maps.append(m)
    others = [m for m in a_maps if m != E]                              # 5 maps

    # 4 tier-2 twins: keep E's a*, redirect b to 4 distinct other actions
    b_targets = [t for t in range(NUM_ACTIONS) if t != b]
    rng.shuffle(b_targets)
    b_variants = []
    for t in b_targets[:4]:
        m = E.copy()
        m[b] = t
        b_variants.append(m)

    candidates = [E] + others + b_variants                             # 1 + 5 + 4 = 10
    return E, candidates


# ---------------------------------------------------------------------------------------
# Build the benchmark: 4 consecutive seed blocks, one class each, `seeds_per_class` valid
# ---------------------------------------------------------------------------------------
def build_benchmark(seeds_per_class=50, seed_start=1, max_steps=200, verbose=True, max_scan=8000,
                    classes=CLASSES):
    """Return a list of (main_seed, actual_seed, class, chosen_action, E_str, [cand_strs]).
    Scans seeds upward; class k takes the next `seeds_per_class` VALID seeds. A seed is
    skipped if the policy doesn't solve it, the class action is unavailable (class 4), or
    there's no 2nd action for the tier-2 twins. (Firing is checked later, at run time.)"""
    env = make_wrapped_env("Taxi_v4", "rgb_array")     # rainy 0.7 (wrapper default)
    policy = load_trained_model("Taxi_v4", "PPO")       # hardcoded 0.7 table policy

    benchmark = []
    seed = seed_start
    for class_id in classes:
        collected = 0
        scan_start = seed
        while collected < seeds_per_class:
            if seed - scan_start > max_scan:
                if verbose:
                    print(f"class {class_id}: only {collected}/{seeds_per_class} after {max_scan} scans")
                break
            base = seed * SEED_BLOCK
            counts, _length, solved = profile_seed(env, policy, base, max_steps)
            if not solved:
                seed += 1
                continue
            a_star = class_picks(counts, seed)[class_id]
            b = select_b(counts, a_star) if a_star is not None else None
            if a_star is None or b is None:
                seed += 1
                continue
            E, candidates = build_candidates(a_star, b, seed)
            benchmark.append((
                seed, base, class_id, a_star,
                map_to_str(E),
                [map_to_str(c) for c in candidates],
            ))
            collected += 1
            seed += 1
        if verbose:
            print(f"class {class_id}: {collected} instances (scanned up to seed {seed - 1})")
    return benchmark


# ---------------------------------------------------------------------------------------
# Runner: diagnose n_per_class instances across visibility x fault_rate
# ---------------------------------------------------------------------------------------
def run_benchmark(benchmark, n_per_class,
                  visibilities=(50, 80),
                  fault_rates=(0.3, 0.2, 0.1, 0.05, 0.02),
                  epsilon=0.2, unknown_fault_rate=False, debug_print=False):
    """Run the PO diagnoser over the first n_per_class instances of each class, sweeping
    visibility x fault_rate. Returns a list of per-cell result dicts."""
    from p_pipeline import run_NON_DETERMINSTIC_single_experiment_PO

    by_class = {c: [] for c in CLASSES}
    for inst in benchmark:
        by_class[inst[2]].append(inst)

    results = []
    for class_id in CLASSES:
        for main_seed, base, _cid, a_star, E_str, cands in by_class[class_id][:n_per_class]:
            for vis in visibilities:
                for fr in fault_rates:
                    out = run_NON_DETERMINSTIC_single_experiment_PO(
                        domain_name="Taxi_v4", ml_model_name="PPO", render_mode="rgb_array",
                        max_exec_len=200, debug_print=debug_print,
                        execution_fault_mode_name=E_str, instance_seed=base,
                        fault_probability=fr, percent_visible_states=vis,
                        possible_fault_mode_names=cands, num_candidate_fault_modes=10,
                        epsilon=epsilon, unknown_fault_rate=unknown_fault_rate,
                        fault_rate_candidates=None, fixed_candidate_fault_modes=cands)
                    results.append({
                        "class": class_id, "main_seed": main_seed, "a_star": a_star,
                        "vis": vis, "fault_rate": fr,
                        "rank": (out.get("real_fault_rank") if out else None),
                        "dropped": out is None,
                    })
    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2     # seeds per class for the test

    print(f"=== building benchmark: {n} seeds/class ===")
    bench = build_benchmark(seeds_per_class=n)
    print(f"built {len(bench)} instances\n")
    for main_seed, base, cid, a_star, E_str, cands in bench:
        print(f"  class {cid} seed {main_seed:<4} a*={ACTION_NAMES[a_star]:<7} "
              f"E={E_str:<16} ({describe_fault([int(x) for x in E_str[1:-1].split(',')])})")

    print(f"\n=== running: {n} seeds/class x vis{{50,80}} x fr{{0.3,0.2,0.1,0.05,0.02}} ===")
    res = run_benchmark(bench, n_per_class=n)

    # summary: avg rank per class, and per (class, fault_rate)
    print("\n--- avg rank by class (lower=better; None=dropped) ---")
    for c in CLASSES:
        rows = [r for r in res if r["class"] == c and r["rank"] is not None]
        dropped = sum(1 for r in res if r["class"] == c and r["dropped"])
        avg = (sum(r["rank"] for r in rows) / len(rows)) if rows else float("nan")
        print(f"  class {c}: avg_rank={avg:.2f} over {len(rows)} cells  (dropped {dropped})")

    print("\n--- avg rank by (class, fault_rate) ---")
    for c in CLASSES:
        line = [f"class {c}:"]
        for fr in (0.3, 0.2, 0.1, 0.05, 0.02):
            rows = [r for r in res if r["class"] == c and r["fault_rate"] == fr and r["rank"] is not None]
            avg = (sum(r["rank"] for r in rows) / len(rows)) if rows else float("nan")
            line.append(f"fr{fr}={avg:.2f}")
        print("  " + "  ".join(line))
