"""Find HARD Taxi-v4 diagnosis instances for the Monte-Carlo diagnoser.

Goal (Ahmad's request): surface a concrete instance where the MC likelihood
diagnoser needs *many* simulations to identify the true fault against near-twin
distractors -- the regime where epsilon / #sims actually matters (unlike the
"too similar/easy" FrozenLake instances).

READ-ONLY w.r.t. existing code: reuses the real per-gap simulation
(`p_diagnosers.execute_one_trace`, identical to what `simulate_m_traces` runs)
and mirrors the executor's trajectory loop. Nothing else is modified.

KEY KNOB -- stochasticity: `rainy_probability` is the prob the intended action
succeeds (lower = MORE slips = harder). make_wrapped_env hardcodes 0.7, so we
build the env directly to sweep it. More stochasticity lowers every candidate's
match prob AND the gap delta between the true fault E and a near-twin C (slips
wash out the distinguishing action) -> #sims to separate blows up ~ 1/delta^2.

Mechanics recap (p_diagnosers.fault_identification_non_deterministic_PO):
  trajectory = [s0,a0,s1,...,sn]; observations = states, some hidden (None).
  a "gap" = span between two consecutive OBSERVED states. per gap, per fault f:
  p_hat_f = fraction of n simulated rollouts (policy + fault f at fault_rate)
  from the observed start landing on the observed end. score = sum_gaps log p_hat.
  near-twin C agrees with E on the fired action, differs only on a rare/hidden
  action -> p_hat_E ~ p_hat_C -> tiny score gap -> many sims to separate.

n* ("min sims to identify"): smallest equal-per-gap n such that for all n' in
[n*, N_max], logL(E) > logL(C) for every distractor C (stable separation).

Usage:
  python analyze_hard_instance.py --mode search --seeds 0:60 --fault_rate 0.5 \
      --visibility 100 --rainy_probability 0.5
  python analyze_hard_instance.py --mode report --seed 7 --true_fault "[0,1,0,3,4,5]" \
      --distractors "[3,1,0,3,4,5]" --fault_rate 0.5 --visibility 60 \
      --rainy_probability 0.5 --repeats 7
"""
import argparse
import math
import random

import gymnasium as gym

from p_diagnosers import execute_one_trace
from h_fault_model_generator import FaultModelGeneratorDiscrete
from h_rl_models import load_trained_model
from h_wrappers import TaxiV4SetStepWrapper
from h_raw_state_comparators import comparators
from h_state_refiners import refiners

DOMAIN = "Taxi_v4"
ML = "PPO"
N_ACTIONS = 6
ACTION_NAMES = {0: "South", 1: "North", 2: "East", 3: "West", 4: "Pickup", 5: "Dropoff"}
IDENTITY = list(range(N_ACTIONS))

_FMG = FaultModelGeneratorDiscrete()
_COMPARATOR = comparators[DOMAIN]


def fault_name(mapping):
    return "[" + ",".join(str(x) for x in mapping) + "]"


def make_env(rainy_probability):
    """Build the wrapped Taxi-v4 at a configurable stochasticity (mirrors make_wrapped_env
    but with rainy_probability as a parameter instead of the hardcoded 0.7)."""
    base = gym.make("Taxi-v4", is_rainy=True, rainy_probability=rainy_probability,
                    fickle_passenger=False, render_mode=None)
    return TaxiV4SetStepWrapper(base)


def get_policy():
    return load_trained_model(DOMAIN, ML)  # tabulated 500-state table from Taxi_v4__PPO.zip


def gen_trajectory(env, policy, seed, fault_mode, fault_rate, max_len):
    """Mirror p_executor.execute exactly, but on the given (configurable-stochasticity) env.
    Returns states [s0..sn], executed actions, faulty action indices."""
    env.reset(seed=seed)
    trajectory = []
    faulty_idx = []
    action_number = 1
    done = False
    exec_len = 1
    obs, _ = env.reset()
    rng = random.Random(seed)
    while not done and exec_len < max_len:
        trajectory.append(int(obs))
        action = int(policy.predict(refiners[DOMAIN](obs), deterministic=True)[0])
        trajectory.append(action)
        faulty_action = fault_mode(action) if rng.random() < fault_rate else action
        if faulty_action != action:
            faulty_idx.append(action_number)
        obs, reward, done, trunc, info = env.step(faulty_action)
        action_number += 1
        exec_len += 1
    states = [trajectory[i] for i in range(0, len(trajectory), 2)]
    actions = [trajectory[i] for i in range(1, len(trajectory), 2)]
    return states, actions, faulty_idx


def healthy_actions(states, policy):
    return [int(policy.predict(refiners[DOMAIN](s), deterministic=True)[0]) for s in states]


def mask_observations(states, visibility, mask_seed):
    n = len(states)
    obs = list(states)
    if visibility >= 100 or n <= 2:
        return obs
    interior = list(range(1, n - 1))
    keep_k = max(0, round((visibility / 100.0) * n) - 2)
    rng = random.Random(mask_seed)
    rng.shuffle(interior)
    for i in interior[keep_k:]:
        obs[i] = None
    return obs


def observed_gaps(obs):
    idxs = [i for i, s in enumerate(obs) if s is not None]
    return [(obs[a], obs[b], b - a) for a, b in zip(idxs[:-1], idxs[1:])]


def cum_hits(start, end, gap_len, fault_mode, fault_rate, diag_seed, simulator, policy, N):
    """Cumulative hit count after each of N faithful traces (mirrors simulate_m_traces)."""
    cum = [0] * N
    h = 0
    for i in range(N):
        rng = random.Random(diag_seed + i)
        s = execute_one_trace(start, gap_len, fault_mode, fault_rate,
                              DOMAIN, diag_seed + i, rng, simulator, policy, False)
        h += 1 if _COMPARATOR(s, end) else 0
        cum[i] = h
    return cum


def logL_at(gap_cum_hits, n):
    total = 0.0
    for cum in gap_cum_hits:
        total += math.log(max(cum[n - 1] / n, 1e-12))
    return total


def min_sims_to_identify(obs, fault_modes, true_key, fault_rate, diag_seed, simulator, policy, N):
    gaps = observed_gaps(obs)
    gap_cum = {k: [cum_hits(s0, s1, glen, fm, fault_rate, diag_seed, simulator, policy, N)
                   for (s0, s1, glen) in gaps]
               for k, fm in fault_modes.items()}
    distractors = [k for k in fault_modes if k != true_key]

    def E_top(n):
        le = logL_at(gap_cum[true_key], n)
        return all(le > logL_at(gap_cum[d], n) for d in distractors)

    nstar = None
    for n in range(1, N + 1):
        if E_top(n):
            step = max(1, (N - n) // 25)
            if all(E_top(m) for m in range(n, N + 1, step)):
                nstar = n
                break
    margin = (logL_at(gap_cum[true_key], N)
              - max(logL_at(gap_cum[d], N) for d in distractors))
    return nstar, margin, gaps


MOVE_ACTIONS = [0, 1, 2, 3]  # South/North/East/West change state; Pickup/Dropoff often no-op


def build_near_twins(hcounts, true_map, fired_action, max_twins=2):
    """Near-twins agree with E on the fired action, differ on the LEAST-used non-fired
    MOVEMENT action (count>=1), redirected to another movement action. Restricting to
    movement actions keeps the difference STATE-CHANGING -> separable-but-hard (vs
    Pickup/Dropoff redirects at wrong tiles, which leave state unchanged = unidentifiable)."""
    used = sorted(((a, hcounts.get(a, 0)) for a in MOVE_ACTIONS
                   if a != fired_action and hcounts.get(a, 0) > 0), key=lambda x: x[1])
    twins = []
    for (a, cnt) in used[:max_twins]:
        m = list(true_map)
        # redirect movement action a to a DIFFERENT movement action (state-changing)
        target = next(t for t in MOVE_ACTIONS if t != a and t != true_map[a])
        m[a] = target
        if m != true_map:
            twins.append((fault_name(m), m, a, cnt))
    return twins


def search(seed_lo, seed_hi, fault_rate, visibility, rainy_p, N, max_len, mask_seed):
    env = make_env(rainy_p)
    simulator = make_env(rainy_p); simulator.reset()
    policy = get_policy()
    print(f"[search] rainy_probability={rainy_p} fault_rate={fault_rate} "
          f"visibility={visibility}% N={N}")
    best = None
    for seed in range(seed_lo, seed_hi):
        states0, _, _ = gen_trajectory(env, policy, seed,
                                       _FMG.generate_fault_model(fault_name(IDENTITY)),
                                       fault_rate, max_len)
        if len(states0) < 6:
            continue
        hc = {}
        for a in healthy_actions(states0, policy):
            hc[a] = hc.get(a, 0) + 1
        fired = max(hc, key=hc.get)
        Emap = list(IDENTITY); Emap[fired] = (fired + 1) % N_ACTIONS
        if Emap[fired] == fired:
            Emap[fired] = (fired + 2) % N_ACTIONS
        Ename = fault_name(Emap)

        states, actions, faulty_idx = gen_trajectory(
            env, policy, seed, _FMG.generate_fault_model(Ename), fault_rate, max_len)
        if len(states) < 6 or not faulty_idx:
            continue
        hcf = {}
        for a in healthy_actions(states, policy):
            hcf[a] = hcf.get(a, 0) + 1
        twins = build_near_twins(hcf, Emap, fired, max_twins=2)
        if not twins:
            continue
        fault_modes = {Ename: _FMG.generate_fault_model(Ename)}
        for (tname, tmap, da, dc) in twins:
            fault_modes[tname] = _FMG.generate_fault_model(tname)

        obs = mask_observations(states, visibility, mask_seed)
        # ROBUST hardness: evaluate over K diagnosis seeds (avoid one-stream MC flukes).
        K = 3
        nstars, margins = [], []
        for r in range(K):
            ns, mg, gaps = min_sims_to_identify(
                obs, fault_modes, Ename, fault_rate, seed + 1000 * r, simulator, policy, N)
            nstars.append(ns); margins.append(mg)
        mean_margin = sum(margins) / K
        robust_sep = all(m > 0 for m in margins)          # E wins every stream
        ok_ns = [x for x in nstars if x is not None]
        med_ns = sorted(ok_ns)[len(ok_ns) // 2] if ok_ns else None
        # hardness = small positive margin (structurally hard but identifiable)
        flag = (f"med n*={med_ns} mean_margin={mean_margin:.3f}" if robust_sep
                else f"AMBIGUOUS (margins {[round(m,2) for m in margins]})")
        print(f"seed={seed:>3} E={Ename} fired={fired}({ACTION_NAMES[fired]}) "
              f"twins={[t[0] for t in twins]} gaps={len(gaps)} fired#={len(faulty_idx)} -> {flag}")
        if robust_sep:
            rec = dict(seed=seed, Ename=Ename, fired=fired, twins=[t[0] for t in twins],
                       med_nstar=med_ns, mean_margin=mean_margin, ngaps=len(gaps),
                       traj_len=len(states), nfired=len(faulty_idx), _hardness=-mean_margin)
            if best is None or rec["_hardness"] > best["_hardness"]:
                best = rec
    print("\n================ HARDEST FOUND ================")
    if best:
        for k, v in best.items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        print(f"\n  reproduce: python analyze_hard_instance.py --mode report "
              f"--seed {best['seed']} --true_fault \"{best['Ename']}\" "
              f"--distractors {' '.join(chr(34)+t+chr(34) for t in best['twins'])} "
              f"--fault_rate {fault_rate} --visibility {visibility} "
              f"--rainy_probability {rainy_p}")
    return best


def report(seed, true_fault, distractors, fault_rate, visibility, rainy_p, N, max_len,
           mask_seed, repeats):
    env = make_env(rainy_p)
    simulator = make_env(rainy_p); simulator.reset()
    policy = get_policy()
    states, actions, faulty_idx = gen_trajectory(
        env, policy, seed, _FMG.generate_fault_model(true_fault), fault_rate, max_len)
    obs = mask_observations(states, visibility, mask_seed)
    fault_modes = {true_fault: _FMG.generate_fault_model(true_fault)}
    for d in distractors:
        fault_modes[d] = _FMG.generate_fault_model(d)

    print("================ INSTANCE ================")
    print(f"domain=Taxi_v4  seed={seed}  rainy_probability={rainy_p} (lower=more stochastic)  "
          f"fault_rate={fault_rate}  visibility={visibility}%")
    print(f"true fault E = {true_fault}   (index->executed; identity={fault_name(IDENTITY)})")
    print(f"candidate distractors = {distractors}")
    print(f"\nfull trajectory states ({len(states)}): {states}")
    print(f"executed actions:        {actions}")
    print(f"fault fired at action indices: {faulty_idx}  ({len(faulty_idx)} times)")
    print(f"\nHIDDEN trajectory (visibility {visibility}%): "
          f"{['_' if o is None else o for o in obs]}")
    gaps = observed_gaps(obs)
    print(f"observed gaps: {len(gaps)}  (lengths {[g[2] for g in gaps]})")

    nstars = []
    for r in range(repeats):
        diag_seed = seed + 1000 * r
        nstar, margin, _ = min_sims_to_identify(
            obs, fault_modes, true_fault, fault_rate, diag_seed, simulator, policy, N)
        nstars.append(nstar)
        print(f"  repeat {r} (diag_seed={diag_seed}): n* = {nstar}  margin@N={margin:.3f}")
    ok = [x for x in nstars if x is not None]
    print("\n================ RESULT ================")
    if ok:
        print(f"min sims/gap to identify E over distractors: "
              f"median n*={sorted(ok)[len(ok)//2]}, range [{min(ok)},{max(ok)}], "
              f"separated {len(ok)}/{repeats} repeats")
        print(f"=> need at least ~{max(ok)} simulations PER GAP "
              f"(~{max(ok)*len(gaps)*len(fault_modes)} total sim-rollouts) "
              f"to reliably identify the real fault.")
    else:
        print(f"NOT separated within N={N} in any repeat -> effectively indistinguishable "
              f"at this visibility/stochasticity (near-impossible instance).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["search", "report"], default="search")
    p.add_argument("--seeds", type=str, default="0:60")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--true_fault", type=str, default=None)
    p.add_argument("--distractors", type=str, nargs="*", default=[])
    p.add_argument("--fault_rate", type=float, default=0.5)
    p.add_argument("--visibility", type=float, default=100)
    p.add_argument("--rainy_probability", type=float, default=0.7)
    p.add_argument("--N", type=int, default=2000, help="max sims/gap to sweep")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--mask_seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    if a.mode == "search":
        lo, hi = (int(x) for x in a.seeds.split(":"))
        search(lo, hi, a.fault_rate, a.visibility, a.rainy_probability, a.N, a.max_len, a.mask_seed)
    else:
        report(a.seed, a.true_fault, a.distractors, a.fault_rate, a.visibility,
               a.rainy_probability, a.N, a.max_len, a.mask_seed, a.repeats)


if __name__ == "__main__":
    main()
