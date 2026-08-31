"""FrozenLake fault-mode benchmark construction — two ways of choosing a* and the 10 candidates.

A fault mode is a 4-int map [LEFT->?, DOWN->?, RIGHT->?, UP->?]; healthy = [0,1,2,3].
Both builders take a per-map healthy command-count (Counter action->count) and a seed, and
return (a_star, e_target, E_str, [10 candidate strings]). Deterministic given (counts, seed).

WAY 1 (Taxi-style, ambiguous/hard): a* = rarest commanded action (count>=2). Candidates =
  the 4 maps of a* (healthy + E + 2 redirects) + near-twins on the 2 rarest OTHER actions,
  taken to 10. Mirrors hard_taxi_benchmark_v2, completed to 10 (Taxi had 6 actions -> 10 twins;
  FrozenLake has 4, so twins span two axes to reach 10).

WAY 2 (distinguishable): a* = MOST-used commanded action. Candidates = healthy + all 3
  redirects of each of the 3 MOST-used actions. Every candidate corrupts a well-used (visible)
  action, so instances are solvable at full observability; hardness comes from hiding states.
"""
import random

HEALTHY = [0, 1, 2, 3]


def _fault(base, idx, tgt):
    m = base[:]
    m[idx] = tgt
    return m


def _to_str(m):
    return str(list(m)).replace(" ", "")


def _dedup(maps):
    out, seen = [], []
    for m in maps:
        if m not in seen:
            seen.append(m); out.append(m)
    return out


def build_way1(counts, seed):
    """Rare a* + near-twins, padded to 10 (Taxi-style)."""
    rng = random.Random(seed * 911 + 3)
    ge2 = [a for a in range(4) if counts.get(a, 0) >= 2]
    if ge2:
        astar = min(ge2, key=lambda a: (counts.get(a, 0), a))       # rarest with count>=2
    else:
        astar = min(range(4), key=lambda a: (counts.get(a, 0), a))  # fallback: rarest overall
    e_target = rng.choice([t for t in range(4) if t != astar])
    E = _fault(HEALTHY, astar, e_target)
    # other actions, rarest first -> twin axes b, c, d
    others = sorted([a for a in range(4) if a != astar], key=lambda a: (counts.get(a, 0), a))
    cands = [_fault(HEALTHY, astar, t) for t in range(4)]           # 4 a*-maps (incl healthy & E)
    for b in others:                                                # near-twins on each other axis
        for t in range(4):
            if t != b:
                cands.append(_fault(E, b, t))
    cands = _dedup(cands)[:10]
    assert E in cands and HEALTHY in cands and len(cands) == 10
    return astar, e_target, _to_str(E), [_to_str(m) for m in cands]


def build_way2(counts, seed):
    """Most-used a* + redirects of the 3 most-used actions + healthy (all distinguishable)."""
    rng = random.Random(seed * 911 + 3)
    ranked = sorted(range(4), key=lambda a: (-counts.get(a, 0), a))  # most-used first
    astar = ranked[0]
    e_target = rng.choice([t for t in range(4) if t != astar])
    E = _fault(HEALTHY, astar, e_target)
    cands = [HEALTHY[:]]
    for a in ranked[:3]:                                            # 3 most-used actions
        for t in range(4):
            if t != a:
                cands.append(_fault(HEALTHY, a, t))
    cands = _dedup(cands)[:10]
    assert E in cands and HEALTHY in cands and len(cands) == 10
    return astar, e_target, _to_str(E), [_to_str(m) for m in cands]


BUILDERS = {1: build_way1, 2: build_way2}
