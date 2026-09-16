import copy
import math
import time
import random

import numpy as np
import gym

from h_consts import DETERMINISTIC, SEED_BLOCK, SIMULATION_OFFSET, MAX_STATES
from h_raw_state_comparators import comparators
from h_rl_models import models, load_trained_model
from h_state_refiners import refiners
from h_wrappers import wrappers, make_wrapped_env


def W(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    ts0 = time.time()
    b = 0
    e = len(observations) - 1
    S = observations[0]
    for i in range(1, len(observations)):
        a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
        a = int(a)
        S, reward, done, trunc, info = simulator.step(a)
        if observations[i] is not None:
            if comparators[domain_name](observations[i], S):
                b = i
            else:
                e = i
                if debug_print:
                    print(f"i broke at {i}")
                break
    D = []
    for i in range(b + 1, e + 1):
        D.append(i)
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    # finilizing the runtime in ms
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    output = {
        "diagnoses": D,
        "init_rt_sec": 0.0,
        "init_rt_ms": 0.0,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": diagnosis_runtime_sec,
        "totl_rt_ms": diagnosis_runtime_ms,
        "G_max_size":0
    }

    return output


def SN(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        G[key_j] = [candidate_fault_modes[key_j], A_j, S_0]
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # running the diagnosis loop
    ts1 = time.time()
    for i in range(1, len(observations)):
        irrelevant_keys = []
        for key_j in G.keys():
            a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
            a_gag_i_j = G[key_j][0](a_gag_i)
            simulator.set_state(G[key_j][2])
            S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
            G[key_j][1].append(int(a_gag_i_j))
            G[key_j][2] = S_gag_i_j
            if observations[i] is not None:
                if not comparators[domain_name](observations[i], S_gag_i_j):
                    irrelevant_keys.append(key_j)

        # remove the irrelevant fault modes
        for key in irrelevant_keys:
            G.pop(key)

        if debug_print:
            print(f'STEP {i}/{len(observations)}: KICKED {len(irrelevant_keys)} ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break
    te1 = time.time()
    diagnosis_runtime_sec += te1 - ts1

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": len(candidate_fault_modes)
    }

    return raw_output


def fm_and_state_in_set(key_raw, state, FG):
    for fkey in FG.keys():
        fkey_raw = fkey.split('_')[0]
        fstate = FG[fkey][2]
        if key_raw == fkey_raw and state == fstate:
            return True
    return False


def simulate_m_tries(simulator, comparator, curr_state, current_action, observed_next_state, num_of_tries ):
    num_of_hits = 0

    for _ in range(num_of_tries):
        simulator.reset()
        simulator.set_state(curr_state)
        sim_next_state, reward, done, trunc, info = simulator.step(current_action)
        simulated_state_equals_observed = comparator(sim_next_state, observed_next_state)

        if simulated_state_equals_observed:
            num_of_hits += 1

        # Important: some envs become "stuck" after terminal unless reset

        if trunc:
            print("Error Simulate m tries:"
                  "done:", done, "trunc:", trunc, "next:", sim_next_state)
            exit(7)

    return num_of_hits


def fault_identification_non_deterministic_FO(debug_print, render_mode,
                                           instance_seed, ml_model_name,
                                           domain_name, observations,
                                           candidate_fault_modes):

    NON_DETERMINISTIC_TRIES = 200
    # load trained model as policy
    policy = load_trained_model(domain_name, ml_model_name)


    # load the environment as simulator

    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)  # instance_seed IS the block base (slot 0)
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0
    fault_hits_for_step = {}
    healthy_fault_prob_for_step = {}

    for i in range(1, len(observations)):
        fault_hits_for_step[i] = {}

        healthy_action, _ = policy.predict(refiners[domain_name](observations[i-1]), deterministic=DETERMINISTIC)
        healthy_action = int(healthy_action)
        healthy_heats = simulate_m_tries(simulator=simulator,
                                         comparator=comparators[domain_name],
                                         curr_state=observations[i-1],
                                         current_action=healthy_action,
                                         observed_next_state=observations[i],
                                         num_of_tries=NON_DETERMINISTIC_TRIES)

        fault_hits_for_step[i]["healthy_fault_key"] = healthy_heats
        healthy_fault_prob_for_step[i] = healthy_heats / NON_DETERMINISTIC_TRIES
        for curr_fault_key in candidate_fault_modes:
            curr_faulty_action = candidate_fault_modes[curr_fault_key](healthy_action)
            faulty_action_hits = simulate_m_tries(simulator=simulator,
                                         comparator=comparators[domain_name],
                                         curr_state=observations[i-1],
                                         current_action=curr_faulty_action,
                                         observed_next_state=observations[i],
                                         num_of_tries=NON_DETERMINISTIC_TRIES)

            fault_hits_for_step[i][curr_fault_key] = faulty_action_hits

    # compute faults likelihood

    # Optimisitic approach

    """
    fault_prob_for_step = {}
    for i in range(1, len(observations)):
        prob_h = healthy_fault_prob_for_step[i]

        for curr_fault_key in candidate_fault_modes:
            fault_hits = fault_hits_for_step[i][curr_fault_key]
            fault_prob_for_step[i][curr_fault_key] = max(fault_hits / NON_DETERMINISTIC_TRIES, prob_h)
    
    """

    # Dynamic - try all intermittency rates
    fault_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fault_prob_for_step_and_rate = {r: {} for r in fault_rates}

    # step 1
    for fault_rate in fault_rates:
        for i in range(1, len(observations)):
            prob_h = healthy_fault_prob_for_step[i]

            if i not in fault_prob_for_step_and_rate[fault_rate]:
                fault_prob_for_step_and_rate[fault_rate][i] = {}

            for curr_fault_key in candidate_fault_modes:
                fault_hits = fault_hits_for_step[i][curr_fault_key]
                curr_fault_prob = fault_hits / NON_DETERMINISTIC_TRIES
                fault_prob_for_step_and_rate[fault_rate][i][curr_fault_key] = fault_rate*curr_fault_prob + (1.0-fault_rate)*prob_h

    log_prob_total_per_fault_and_rate = {f: {} for f in candidate_fault_modes}

    # step 2
    for f in candidate_fault_modes:
        for r in fault_rates:
            log_prob = 0.0
            for i in range(1, len(observations)):
                p = fault_prob_for_step_and_rate[r][i][f]
                # avoid log(0) issues
                p = max(p, 1e-12)
                log_prob += math.log(p)
            log_prob_total_per_fault_and_rate[f][r] = log_prob

    best_prob_per_fault = {}
    best_rate_per_fault = {}

    # step 3 (same logic, just naming)
    T = len(observations) - 1

    # --- healthy baseline log-likelihood ---
    healthy_logL = 0.0
    for i in range(1, len(observations)):
        p_h = max(healthy_fault_prob_for_step[i], 1e-12)
        healthy_logL += math.log(p_h)

    best_logL_per_fault = {}
    best_rate_per_fault = {}

    for f in candidate_fault_modes:
        best_rate = max(
            log_prob_total_per_fault_and_rate[f],
            key=lambda r: log_prob_total_per_fault_and_rate[f][r]
        )
        best_logL = log_prob_total_per_fault_and_rate[f][best_rate]
        best_rate_per_fault[f] = best_rate
        best_logL_per_fault[f] = best_logL

    # Sort faults by log-likelihood (descending)
    sorted_faults = sorted(best_logL_per_fault.items(), key=lambda x: x[1], reverse=True)
    T = len(observations) - 1

    # sorted_faults: List[Tuple[fault_key, logL]]
    sorted_faults_geo = [(fault, math.exp(logL / T)) for fault, logL in sorted_faults]

    print("Fault scores (higher is better):")
    print(f"Healthy (no fault) logL: {healthy_logL:.6f}, per_step_geo_p: {math.exp(healthy_logL / T):.6f}")

    extra_output = ""
    for fault, logL in sorted_faults:
        rate = best_rate_per_fault[fault]
        curr_output = (f"Fault: {fault}, logL: {logL:.6f}, "
                        f"Exp of LogL: {math.exp(logL):.6f}, "
                        f"Normalized LogL: {(logL / T):.6f}, "
                        f"Exp of Normalized LogL: {math.exp(logL / T):.6f}, "
                        f"Best Rate: {rate}")

        print(curr_output)
        extra_output += curr_output
        extra_output += "\n"

    output = {}
    output["sorted_faults"] = sorted_faults
    output["sorted_faults_with_exp_val"] = sorted_faults_geo
    output["observations"] = observations
    output["observations_len"] = len(observations)
    output["extra_output"] = extra_output
    return output

"""
def simulate_m_traces(starting_state, observed_next_state, trace_length,
                      num_of_tries, fault_mode, fault_rate,
                      domain_name, instance_seed,
                      simulator, model, comparator, debug_print):

    num_of_hits = 0

    for i in range(num_of_tries):
        trace_seed = instance_seed + i * MAX_STATES
        rng = random.Random(trace_seed)
        sim_next_state = execute_one_trace(starting_state, trace_length,
                      fault_mode, fault_rate,
                      domain_name, trace_seed, rng,
                      simulator, model, debug_print)


        simulated_state_equals_observed = comparator(sim_next_state, observed_next_state)

        if simulated_state_equals_observed:
            num_of_hits += 1

    return num_of_hits
"""

def execute_one_trace(starting_state,
                      trace_length,
                      fault_mode,
                      fault_rate,
                      domain_name,
                      instance_seed,
                      rng,
                      simulator,
                      model,
                      debug_print):

    initial_obs, _ = simulator.reset(seed=instance_seed)
    simulator.set_state(starting_state)
    done = False
    trunc = False
    exec_len = 1
    obs = starting_state

    while not done and not trunc and exec_len <= trace_length:

        healthy_action, _ = model.predict(refiners[domain_name](obs),
                                  deterministic=DETERMINISTIC)
        healthy_action = int(healthy_action)

        random_val = rng.random()
        if random_val < fault_rate:
            faulty_action = fault_mode(healthy_action)
        else:
            faulty_action = healthy_action

        obs, reward, done, trunc, info = simulator.step(faulty_action)
        exec_len += 1

    return obs

def simulate_m_traces_adaptive_monte_carlo(starting_state, observed_next_state, trace_length,
                      fault_mode, fault_rate,
                      domain_name, instance_seed,
                      simulator, model, comparator, debug_print,
                      min_tries=100,
                      max_tries=2000,
                      batch_size=50,
                      epsilon=0.02):

    curr_num_of_tries = 0
    num_of_hits = 0
    margin = None
    curr_sim_seed = instance_seed

    # Allocate the fault-firing RNG ONCE; reseed it per trace instead of building a new
    # Random object each iteration (rng.seed(x) gives the same state as Random(x), so this is
    # behaviour-identical, just without the per-trace allocation).
    rng = random.Random()

    while True:
        for i in range(batch_size):
            rng.seed(curr_sim_seed)
            sim_next_state = execute_one_trace(starting_state, trace_length,
                                               fault_mode, fault_rate,
                                               domain_name, curr_sim_seed, rng,
                                               simulator, model, debug_print)

            if comparator(sim_next_state, observed_next_state):
                num_of_hits += 1

            curr_sim_seed += MAX_STATES

        curr_num_of_tries += batch_size

        p_hat = num_of_hits / curr_num_of_tries

        # --- stopping condition ---
        if curr_num_of_tries >= min_tries:
            error = math.sqrt(p_hat * (1.0 - p_hat) / curr_num_of_tries)
            margin = 1.96 * error

            if margin < epsilon:
                break

        # --- safety cap ---
        if curr_num_of_tries >= max_tries:
            break


    return {
                "min_tries": min_tries,
                "max_tries": max_tries,
                "num_of_hits": num_of_hits,
                "num_of_tries": curr_num_of_tries,
                "p_hat": p_hat,
                "margin": margin if curr_num_of_tries >= min_tries else None,
                "stop_reason": "confidence" if curr_num_of_tries < max_tries else "max_tries",
                "trace_length": trace_length
    }


def fault_identification_non_deterministic_PO_unknown_fault_rate(
        debug_print,
        render_mode,
        instance_seed,
        ml_model_name,
        domain_name,
        observations,
        candidate_fault_modes,
        fault_rate_candidates,
        epsilon,
        ):

    diagnosis_seed = instance_seed + SIMULATION_OFFSET

    policy = load_trained_model(domain_name, ml_model_name)

    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)  # instance_seed IS the block base (slot 0)
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    assert len(observations) <= MAX_STATES, (
        f"trajectory length {len(observations)} exceeds MAX_STATES {MAX_STATES}; "
        f"raise MAX_STATES in h_consts.")

    diagnosis_time_sec_start = time.time()

    # structure:
    # fault_prob_hat_for_step_and_rate[i][fault_rate][fault_key] = p_hat
    fault_prob_hat_for_step_and_rate = {}

    last_observed_index = 0
    gap_times = []
    num_gaps = 0
    num_of_observed_states = 1
    adaptive_stats = []


    for i in range(1, len(observations)):

        if observations[i] is None:
            continue

        fault_prob_hat_for_step_and_rate[i] = {}

        num_of_observed_states += 1
        current_gap_length = i - last_observed_index

        min_tries = 100 + 10 * current_gap_length
        scale = max(1.0, (0.025 / epsilon) ** 2)

        base_max = int(2500 * scale)
        gap_bonus = int(150 * current_gap_length * scale)
        max_tries = base_max + gap_bonus

        current_gap_seed = diagnosis_seed + last_observed_index
        top_seed_offset_during_iterations = (max_tries + 50) * MAX_STATES

        assert current_gap_seed + top_seed_offset_during_iterations  < instance_seed + SEED_BLOCK, (
            f"MC seeds overflow this instance's block: top offset {current_gap_seed + top_seed_offset_during_iterations} >= "
            f"SEED_BLOCK {instance_seed + SEED_BLOCK} (epsilon={epsilon}, max_tries={max_tries}, "
            f"MAX_STATES={MAX_STATES}). Increase SEED_BLOCK (e.g. 10_000_000).")

        gap_start_time = time.time()

        for curr_fault_rate in fault_rate_candidates:

            fault_prob_hat_for_step_and_rate[i][curr_fault_rate] = {}

            for curr_fault_key in candidate_fault_modes:

                res = simulate_m_traces_adaptive_monte_carlo(
                    observations[last_observed_index],
                    observations[i],
                    current_gap_length,
                    candidate_fault_modes[curr_fault_key],
                    curr_fault_rate,
                    domain_name,
                    current_gap_seed,  # gap residue class; per-trace stride = MAX_STATES
                    simulator,
                    policy,
                    comparators[domain_name],
                    debug_print,
                    min_tries=min_tries,
                    max_tries=max_tries,
                    batch_size=50,
                    epsilon=epsilon
                )

                p_hat = max(res["p_hat"], 1e-12)
                fault_prob_hat_for_step_and_rate[i][curr_fault_rate][curr_fault_key] = p_hat
                adaptive_stats.append(res)

                if res["stop_reason"] == "max_tries":
                    print(
                        f"MAX HIT | gap={res['trace_length']} "
                        f"tries={res['num_of_tries']} "
                        f"p={res['p_hat']:.4f} "
                        f"margin={res['margin']} "
                        f"fault_rate={curr_fault_rate} "
                        f"curr_fault_key={curr_fault_key}"
                    )

        gap_end_time = time.time()
        gap_times.append(gap_end_time - gap_start_time)
        num_gaps += 1
        last_observed_index = i

    # ---------------------------------------------------
    # Compute log likelihood for each fault and each rate
    # ---------------------------------------------------

    log_prob_total_per_fault_and_rate = {}

    for f in candidate_fault_modes:
        log_prob_total_per_fault_and_rate[f] = {}

        for curr_fault_rate in fault_rate_candidates:
            log_prob = 0.0

            for i in range(1, len(observations)):
                if observations[i] is None:
                    continue

                p = fault_prob_hat_for_step_and_rate[i][curr_fault_rate][f]
                log_prob += math.log(p)

            log_prob_total_per_fault_and_rate[f][curr_fault_rate] = log_prob

    # ---------------------------------------------------
    # For each fault, choose the best fault rate
    # ---------------------------------------------------

    best_logL_per_fault = {}
    best_rate_per_fault = {}

    for f in candidate_fault_modes:

        best_rate = max(
            fault_rate_candidates,
            key=lambda r: log_prob_total_per_fault_and_rate[f][r]
        )

        best_rate_per_fault[f] = best_rate
        best_logL_per_fault[f] = log_prob_total_per_fault_and_rate[f][best_rate]

    sorted_faults = sorted(
        best_logL_per_fault.items(),
        key=lambda x: x[1],
        reverse=True
    )
    # sorted_faults: List[Tuple[fault_key, logL]]

    T = num_of_observed_states - 1
    sorted_faults_geo = [
        (fault, math.exp(logL / T))
        for fault, logL in sorted_faults
    ]



    print("Fault scores using UNKNOWN fault rate method:")
    extra_output = ""

    for fault, logL in sorted_faults:
        rate = best_rate_per_fault[fault]

        curr_output = (
            f"Fault: {fault}, "
            f"logL: {logL:.6f}, "
            f"Normalized LogL: {(logL / T):.6f}, "
            f"Exp of Normalized LogL: {math.exp(logL / T):.6f}, "
            f"Best Estimated Rate: {rate}"
        )

        print(curr_output)
        extra_output += curr_output + "\n"

    diagnosis_time_sec_end = time.time()
    diagnosis_time_sec = diagnosis_time_sec_end - diagnosis_time_sec_start
    diagnosis_time_ms = diagnosis_time_sec * 1000

    avg_gap_time = sum(gap_times) / len(gap_times) if gap_times else 0.0

    output = {}
    output["diagnosis_time_sec"] = diagnosis_time_sec
    output["diagnosis_time_ms"] = diagnosis_time_ms

    output["avg_gap_time"] = avg_gap_time
    output["num_gaps"] = num_gaps

    output["sorted_faults"] = sorted_faults
    output["sorted_faults_with_exp_val"] = sorted_faults_geo

    output["best_rate_per_fault"] = best_rate_per_fault
    output["log_prob_total_per_fault_and_rate"] = log_prob_total_per_fault_and_rate
    output["fault_rate_candidates"] = fault_rate_candidates

    output["observations"] = observations
    output["observations_len"] = len(observations)
    output["extra_output"] = extra_output

    total_calls = len(adaptive_stats)
    avg_tries = sum(s["num_of_tries"] for s in adaptive_stats) / total_calls
    adaptive_max_tries_avg = sum(s["max_tries"] for s in adaptive_stats) / total_calls
    adaptive_min_tries_avg = sum(s["min_tries"] for s in adaptive_stats) / total_calls
    max_hits = sum(1 for s in adaptive_stats if s["stop_reason"] == "max_tries")
    conf_hits = total_calls - max_hits

    margins = [
        s["margin"]
        for s in adaptive_stats
        if s["margin"] is not None
    ]

    avg_margin = sum(margins) / len(margins) if margins else None

    avg_p_hat = (
        sum(s["p_hat"] for s in adaptive_stats) / len(adaptive_stats)
        if adaptive_stats else None
    )

    output["adaptive_total_calls"] = total_calls
    output["adaptive_avg_real_tries"] = avg_tries
    output["adaptive_max_tries_avg"] = adaptive_max_tries_avg
    output["adaptive_min_tries_avg"] = adaptive_min_tries_avg
    output["adaptive_max_stops"] = max_hits
    output["adaptive_conf_stops"] = conf_hits
    output["adaptive_max_stop_rate"] = max_hits / total_calls
    output["adaptive_ever_hit_max"] = bool(max_hits > 0)
    output["adaptive_avg_margin"] = avg_margin
    output["adaptive_avg_p_hat"] = avg_p_hat

    print("\n========= ADAPTIVE MC DEBUG =========")
    print(f"Total calls: {total_calls}")
    print(f"Avg tries: {avg_tries:.2f}")
    print(f"Confidence stops: {conf_hits}")
    print(f"Max stops: {max_hits}")
    print(f"Max stop rate: {max_hits / total_calls:.3f}")

    return output


def fault_identification_non_deterministic_PO_unknown_fault_rate_RACING(
        debug_print,
        render_mode,
        instance_seed,
        ml_model_name,
        domain_name,
        observations,
        candidate_fault_modes,
        fault_rate_candidates,
        epsilon,
        confidence=0.95,
        init_batch=40,
        round_batch=40,
        max_tries_cap=1200,
        max_rounds=80,
        tie_margin=1.0,
        ):
    """UNKNOWN-fault-rate diagnosis by CONFIDENCE-BOUNDED RACING (keeps all candidates; never drops
    a priori). Same inputs/outputs as fault_identification_non_deterministic_PO_unknown_fault_rate,
    so run_NON_DETERMINSTIC_single_experiment_PO can call it interchangeably.

    Idea (see references/UFR_SPEEDUP_FINDINGS.md):
      * 100 pairs = (candidate_fault_mode x candidate_fault_rate); the trajectory has GAPS.
      * For each (pair, gap) we Monte-Carlo estimate p_hat = P(reach the gap's observed end state).
        p_hat is an estimate: the true p sits in a confidence interval [p_lo, p_hi] (Hoeffding), which
        NARROWS as we add traces.
      * pair total  L(pair) = sum_gaps log p(pair, gap)   -> CI [L_lo, L_hi] (sum of the gap bounds).
      * fault score(cf) = max over its rates of L(cf, rate)  (the ufr rule) -> CI too.
      * Rank faults by score. A comparison cf_A vs cf_B is DECIDED when their score CIs don't overlap.
      Racing: seed every (pair,gap) with a small batch, then spend more traces ONLY where they can
      still change the ranking. Freeze a rate that provably can't be its fault's best rate; stop when
      every fault's position is settled. CRN: all pairs share the gap's seed base, so a given trace
      index uses the SAME env randomness across faults/rates -> comparison DIFFERENCES have far lower
      variance -> decisions come faster. Safety: we only ever stop sampling something the CIs have
      PROVED; worst case (all near-ties, e.g. Taxi) nothing is frozen and we do full work -> never
      worse than the full method.
    """
    # Budget is tunable at launch (no code edit) via env vars, so we can sweep the time<->rank dial:
    #   MG_RACING_INIT / MG_RACING_ROUND / MG_RACING_CAP / MG_RACING_ROUNDS
    import os as _os
    init_batch = int(_os.environ.get("MG_RACING_INIT", init_batch))
    round_batch = int(_os.environ.get("MG_RACING_ROUND", round_batch))
    max_tries_cap = int(_os.environ.get("MG_RACING_CAP", max_tries_cap))
    max_rounds = int(_os.environ.get("MG_RACING_ROUNDS", max_rounds))

    diagnosis_seed = instance_seed + SIMULATION_OFFSET
    policy = load_trained_model(domain_name, ml_model_name)
    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0 = initial_obs
    assert comparators[domain_name](observations[0], S_0)
    assert len(observations) <= MAX_STATES, (
        f"trajectory length {len(observations)} exceeds MAX_STATES {MAX_STATES}")
    comparator = comparators[domain_name]

    diagnosis_time_sec_start = time.time()

    faults = list(candidate_fault_modes.keys())
    rates = list(fault_rate_candidates)

    # ----- build the gap list (start state, observed end state, hidden length, shared seed base) -----
    gaps = []
    last_observed_index = 0
    for i in range(1, len(observations)):
        if observations[i] is None:
            continue
        gaps.append({
            "idx": len(gaps),
            "start": observations[last_observed_index],
            "end": observations[i],
            "length": i - last_observed_index,
            # shared across ALL (fault,rate) for this gap -> Common Random Numbers.
            # last_observed_index puts each gap on its own residue class mod MAX_STATES (no collisions).
            "seed": diagnosis_seed + last_observed_index,
        })
        last_observed_index = i
    G = len(gaps)

    # ----- per (fault, rate, gap): the HIT VECTOR (0/1 per trace), aligned by trace index so that a
    # given index uses the SAME random seed across every fault/rate (Common Random Numbers). Storing
    # the vectors (not just a count) lets us measure the CORRELATION between two faults, hence the
    # variance of their DIFFERENCE -- the key to deciding comparisons cheaply. -----
    H = {(f, r, g["idx"]): [] for f in faults for r in rates for g in gaps}
    frozen_rate = set()   # (unused in v2; kept for output-schema compatibility)
    z = 1.96 if confidence <= 0.95 else 2.576   # per-comparison normal quantile

    def sample(f, r, gidx, n):
        """Run n MORE traces for (f,r,gap), continuing the seed sequence (CRN-aligned by index)."""
        a = H[(f, r, gidx)]
        g = gaps[gidx]
        fm = candidate_fault_modes[f]
        base = g["seed"]; start = g["start"]; end = g["end"]; length = g["length"]
        rng = random.Random()
        for i in range(len(a), len(a) + n):
            s = base + i * MAX_STATES
            rng.seed(s)
            nxt = execute_one_trace(start, length, fm, r, domain_name, s, rng,
                                    simulator, policy, False)
            a.append(1 if comparator(nxt, end) else 0)

    _EPS = 1e-6

    def phat(f, r, gidx):
        a = H[(f, r, gidx)]
        return (sum(a) / len(a)) if a else _EPS

    def L_point(f, r):
        return sum(math.log(max(phat(f, r, g["idx"]), 1e-12)) for g in gaps)

    def best_rate(f):
        return max(rates, key=lambda r: L_point(f, r))

    def diff_decision(A, rA, B, rB):
        """Decide the sign of D = L(A,rA) - L(B,rB) = sum_g (log pA_g - log pB_g), using the CRN
        PAIRED variance (delta method): per gap, over the common trace prefix (same seeds),
          var_g = (1-a)/(a n) + (1-b)/(b n) - 2(pab - a b)/(a b n),
        where a,b are the two hit-rates and pab the co-hit rate. The -2*cov term is what shrinks the
        difference's variance far below the two marginals. Returns (D_hat, half_width)."""
        D = 0.0; var = 0.0
        for g in gaps:
            gi = g["idx"]
            Xa = H[(A, rA, gi)]; Yb = H[(B, rB, gi)]
            n = min(len(Xa), len(Yb))
            if n == 0:
                return 0.0, float("inf")
            xa = np.frombuffer(bytes(Xa[:n]), dtype=np.uint8)
            yb = np.frombuffer(bytes(Yb[:n]), dtype=np.uint8)
            a = min(max(xa.mean(), _EPS), 1.0 - 1e-9)
            b = min(max(yb.mean(), _EPS), 1.0 - 1e-9)
            pab = float(np.dot(xa, yb)) / n
            D += math.log(a) - math.log(b)
            var += (1 - a) / (a * n) + (1 - b) / (b * n) - 2.0 * (pab - a * b) / (a * b * n)
        var = max(var, 0.0)
        return D, z * math.sqrt(var)

    # ----- 1. seed every (pair, gap) with a small batch -----
    for f in faults:
        for r in rates:
            for g in gaps:
                sample(f, r, g["idx"], init_batch)

    # A fault is HOPELESS if, at every rate, it never reproduces a single observed transition
    # (0 hits across all gaps) -> its likelihood sits at the floor and it can't be told apart from
    # any other hopeless fault. Two hopeless faults are a genuine tie: their mutual order is
    # meaningless and must NOT block the confidence-stop (otherwise the useless bottom of the ranking
    # keeps the whole instance sampling forever).
    def hopeless(f):
        return all(all(sum(H[(f, r, g["idx"])]) == 0 for g in gaps) for r in rates)

    # ----- 2. racing rounds: decide ADJACENT comparisons in the current order via the paired diff CI.
    # Refine only the undecided adjacent pairs, on the gap contributing the most variance, sampling
    # BOTH faults (keeps the CRN pairing). A pair is auto-resolved when both faults are hopeless.
    num_rounds = 0
    stop_reason = "decided"
    while True:
        order = sorted(faults, key=lambda f: L_point(f, best_rate(f)), reverse=True)
        brate = {f: best_rate(f) for f in order}
        hop = {f: hopeless(f) for f in faults}

        undecided = []
        for k in range(len(order) - 1):
            A, B = order[k], order[k + 1]
            # both hopeless -> genuine tie at the bottom, resolved (don't waste budget).
            # A above and B hopeless (A not) -> A clearly wins, resolved.
            if hop[A] and hop[B]:
                continue
            if hop[B] and not hop[A]:
                continue
            D, half = diff_decision(A, brate[A], B, brate[B])
            # A pair is RESOLVED when either: A is proven strictly above B (D - half > 0), OR the
            # difference is pinned to within tie_margin (half < tie_margin) -> we are CONFIDENT it is a
            # (near-)tie, so more sampling is pointless. Otherwise it is genuinely uncertain -> refine.
            if (D - half > 0) or (half < tie_margin):
                continue
            undecided.append((A, brate[A], B, brate[B]))

        if not undecided:
            stop_reason = "decided"; break
        if num_rounds >= max_rounds:
            stop_reason = "max_rounds"; break

        did_sample = False
        for (A, rA, B, rB) in undecided:
            # gap with the largest (marginal) variance contribution that isn't at the cap
            best_g, best_v = None, -1.0
            for g in gaps:
                gi = g["idx"]
                nmin = min(len(H[(A, rA, gi)]), len(H[(B, rB, gi)]))
                if nmin >= max_tries_cap:
                    continue
                nn = max(nmin, 1)
                a = min(max(phat(A, rA, gi), _EPS), 1.0 - 1e-9)
                b = min(max(phat(B, rB, gi), _EPS), 1.0 - 1e-9)
                v = (1 - a) / (a * nn) + (1 - b) / (b * nn)
                if v > best_v:
                    best_v, best_g = v, gi
            if best_g is not None:
                sample(A, rA, best_g, round_batch)
                sample(B, rB, best_g, round_batch)
                did_sample = True
        if not did_sample:
            stop_reason = "budget_exhausted"; break
        num_rounds += 1

    # ----- 3. build outputs (mirror the full ufr method) -----
    log_prob_total_per_fault_and_rate = {f: {r: L_point(f, r) for r in rates} for f in faults}
    best_rate_per_fault = {}
    best_logL_per_fault = {}
    for f in faults:
        br = max(rates, key=lambda r: log_prob_total_per_fault_and_rate[f][r])
        best_rate_per_fault[f] = br
        best_logL_per_fault[f] = log_prob_total_per_fault_and_rate[f][br]

    sorted_faults = sorted(best_logL_per_fault.items(), key=lambda x: x[1], reverse=True)
    T = max(G, 1)
    sorted_faults_geo = [(fault, math.exp(logL / T)) for fault, logL in sorted_faults]

    total_traces = sum(len(v) for v in H.values())
    # what the FULL method would have sampled if every (pair,gap) ran to the cap (a conservative
    # upper baseline for the speedup, machine-independent):
    full_baseline_traces = len(H) * max_tries_cap

    extra_output = ""
    for fault, logL in sorted_faults:
        curr = (f"Fault: {fault}, logL: {logL:.6f}, Best Estimated Rate: {best_rate_per_fault[fault]}")
        extra_output += curr + "\n"

    diagnosis_time_sec = time.time() - diagnosis_time_sec_start
    output = {}
    output["diagnosis_time_sec"] = diagnosis_time_sec
    output["diagnosis_time_ms"] = diagnosis_time_sec * 1000
    output["avg_gap_time"] = 0.0
    output["num_gaps"] = G
    output["sorted_faults"] = sorted_faults
    output["sorted_faults_with_exp_val"] = sorted_faults_geo
    output["best_rate_per_fault"] = best_rate_per_fault
    output["log_prob_total_per_fault_and_rate"] = log_prob_total_per_fault_and_rate
    output["fault_rate_candidates"] = fault_rate_candidates
    output["observations"] = observations
    output["observations_len"] = len(observations)
    output["extra_output"] = extra_output
    # racing-specific stats
    output["racing_total_traces"] = total_traces
    output["racing_full_baseline_traces"] = full_baseline_traces
    output["racing_trace_speedup_vs_cap"] = full_baseline_traces / total_traces if total_traces else None
    output["racing_num_rounds"] = num_rounds
    output["racing_frozen_rate_pairs"] = len(frozen_rate)
    output["racing_stop_reason"] = stop_reason
    output["racing_confidence"] = confidence
    # keep the columns the ufr path also emits, so the xlsx schema lines up
    output["adaptive_total_calls"] = len(H)
    output["adaptive_avg_real_tries"] = total_traces / len(H) if H else 0

    print(f"\n===== RACING done: stop={stop_reason} rounds={num_rounds} "
          f"traces={total_traces} (cap-baseline {full_baseline_traces}, "
          f"~{output['racing_trace_speedup_vs_cap']:.1f}x) frozen_rates={len(frozen_rate)}/"
          f"{len(faults)*len(rates)} =====")
    return output


def _v1_run(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations,
            candidate_fault_modes, fault_rate_candidates, epsilon,
            use_freeze, init_batch=40, round_batch=40, max_rounds=80):
    """v1 unknown-fault-rate diagnoser (MARGINAL confidence intervals; each pair judged on its OWN
    interval). Two modes share this body:
      * use_freeze=False -> fault_identification_..._V1        (no rate freezing)
      * use_freeze=True  -> fault_identification_..._V1_FREEZE (freeze rates that can't be a fault's best)

    Per (candidate_fault_mode, candidate_fault_rate, gap): estimate p_hat by Monte-Carlo, with the
    normal-approx margin (1.96*sqrt(p(1-p)/n)). Pair score L = sum_gaps log p_hat, with a LOOSE CI
    [sum log p_lo, sum log p_hi]. Fault score = L at its best rate. Rank faults by the point score;
    keep simulating only the faults whose adjacent score-intervals still OVERLAP (order not yet
    confident), stop when all adjacent orders are separated (lo[A] > hi[B]) or the budget runs out.
    FREEZE additionally drops, inside each fault, any rate whose interval high-end is below another
    live rate's low-end (it can never be that fault's best rate)."""
    # budget tunable at launch (no code edit): MG_V1_INIT / MG_V1_ROUND / MG_V1_ROUNDS
    import os as _os
    init_batch = int(_os.environ.get("MG_V1_INIT", init_batch))
    round_batch = int(_os.environ.get("MG_V1_ROUND", round_batch))
    max_rounds = int(_os.environ.get("MG_V1_ROUNDS", max_rounds))
    # per-(pair,gap) sample cap: once a gap is this well estimated, stop pouring budget into it
    # (mirrors full-ufr's per-estimate max_tries). Tunable via MG_V1_CAP.
    max_tries_cap = int(_os.environ.get("MG_V1_CAP", 1200))
    # PER-ESTIMATE epsilon adaptive stop (same rule as full-ufr's adaptive Monte-Carlo): an
    # individual (pair, gap) estimate is "settled" -- and gets no more traces -- once its margin
    # 1.96*sqrt(p(1-p)/n) < eps_stop, provided it has at least min_tries samples. This is what full
    # does for EVERY estimate; v1 applies it only to the gaps it chooses to refine (contended pairs),
    # so easy gaps stop cheaply. eps_stop defaults to the run's epsilon; min_tries mirrors full's 100.
    eps_stop = float(_os.environ.get("MG_V1_EPS", epsilon))
    min_tries = int(_os.environ.get("MG_V1_MIN", 100))
    # L-CI aggregation: "loose" sums the per-gap log half-widths (conservative, over-wide -- so
    # freezing/deciding rarely fire, and budget never concentrates on the best rate); "tight" uses
    # proper error propagation sqrt(sum half_g^2) (gaps independent), which is much narrower, lets
    # freezing actually prune losing rates, and concentrates budget on the discriminative best rate.
    ci_mode = _os.environ.get("MG_V1_CI", "loose")

    _Z = 1.96
    _Z2 = _Z * _Z

    def wilson(hits, n):
        """Wilson score interval (center, lo, hi) for a binomial proportion. Unlike the raw Wald
        margin 1.96*sqrt(p(1-p)/n), it stays NON-DEGENERATE at the boundaries: a 0-hit gap in n
        tries gets hi ~ z^2/n (≈0.04 at n=100), not a false zero-width point pinned at 0. `center`
        is a shrinkage point estimate, always strictly inside (0, 1)."""
        if n <= 0:
            return 1e-12, 1e-12, 1.0
        p = hits / n
        denom = 1.0 + _Z2 / n
        center = (p + _Z2 / (2 * n)) / denom
        half = (_Z / denom) * math.sqrt(p * (1.0 - p) / n + _Z2 / (4 * n * n))
        return center, max(center - half, 0.0), min(center + half, 1.0)

    def gap_settled(a):
        """True if this (pair,gap) estimate needs no more traces: at the hard cap, or already
        pinned to within eps_stop (past min_tries) -- the per-estimate epsilon adaptive stop."""
        n = a["n"]
        if n >= max_tries_cap:
            return True
        if n >= min_tries:
            _, lo, hi = wilson(a["hits"], n)
            if (hi - lo) / 2.0 < eps_stop:
                return True
        return False

    diagnosis_seed = instance_seed + SIMULATION_OFFSET
    policy = load_trained_model(domain_name, ml_model_name)
    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)
    assert comparators[domain_name](observations[0], initial_obs)
    comparator = comparators[domain_name]
    t0 = time.time()

    faults = list(candidate_fault_modes.keys())
    rates = list(fault_rate_candidates)

    # gaps: (start observed state, end observed state, hidden length, seed base for this gap)
    gaps = []
    last = 0
    for i in range(1, len(observations)):
        if observations[i] is None:
            continue
        gaps.append({"idx": len(gaps), "start": observations[last], "end": observations[i],
                     "length": i - last, "seed": diagnosis_seed + last})
        last = i

    acc = {(f, r, g["idx"]): {"hits": 0, "n": 0} for f in faults for r in rates for g in gaps}
    frozen = set()   # (f, r) rates proven not to be their fault's best (only used if use_freeze)

    def sample(f, r, gidx, k):
        a = acc[(f, r, gidx)]; g = gaps[gidx]; fm = candidate_fault_modes[f]
        base = g["seed"]; rng = random.Random()
        for i in range(a["n"], a["n"] + k):
            s = base + i * MAX_STATES
            rng.seed(s)
            nxt = execute_one_trace(g["start"], g["length"], fm, r, domain_name, s, rng,
                                    simulator, policy, False)
            if comparator(nxt, g["end"]):
                a["hits"] += 1
            a["n"] += 1

    def interval_of_pair(f, r):
        """(L_point, L_lo, L_hi) for a pair. L_point = sum_g log(center). The CI is either LOOSE
        (sum of per-gap log half-widths -- conservative) or TIGHT (sqrt of sum of squared half-widths
        -- proper independent-gap error propagation, much narrower). Wilson endpoints keep a 0-hit /
        all-hit gap from being falsely pinned with zero width."""
        Lp = 0.0
        loose = 0.0        # sum of half-widths
        sq = 0.0           # sum of squared half-widths
        for g in gaps:
            a = acc[(f, r, g["idx"])]
            c, lo, hi = wilson(a["hits"], a["n"])
            Lp += math.log(max(c, 1e-12))
            h = 0.5 * (math.log(max(hi, 1e-12)) - math.log(max(lo, 1e-12)))
            loose += h; sq += h * h
        H = math.sqrt(sq) if ci_mode == "tight" else loose
        return Lp, Lp - H, Lp + H

    def partial_interval(f, r, upto_gidx):
        """(L_point, L_lo, L_hi) over ONLY gaps 0..upto_gidx that have been sampled -- used for
        gap-incremental freezing (compare rates on the gaps seen so far)."""
        Lp = Llo = Lhi = 0.0
        for gi in range(upto_gidx + 1):
            a = acc[(f, r, gi)]; n = a["n"]
            if n == 0:
                continue
            ph = a["hits"] / n
            margin = 1.96 * math.sqrt(ph * (1.0 - ph) / n)
            plo = max(ph - margin, 1e-12); phi = min(max(ph + margin, 1e-12), 1.0)
            Lp += math.log(max(ph, 1e-12)); Llo += math.log(plo); Lhi += math.log(max(phi, 1e-12))
        return Lp, Llo, Lhi

    def fault_score(f):
        # A fault's score is max_r L(f, r) -- a MAXIMUM over its rates. The correct CI on that max is
        # (max point, max low-end, max high-end) over the live rates, NOT the single point-best rate's
        # interval: an under-sampled live rate has a wide interval whose high-end can exceed the
        # point-best rate's high-end, and using only best_r would UNDER-state the fault's upper
        # uncertainty -> declaring it "decided" prematurely while a wider live rate could still lift it
        # into overlap. Aggregating over all live rates keeps such a fault undecided until that rate is
        # narrowed by more sampling.
        live = [r for r in rates if (f, r) not in frozen] or rates
        ivs = {r: interval_of_pair(f, r) for r in live}
        best_r = max(live, key=lambda r: ivs[r][0])
        Lp = ivs[best_r][0]                         # point = best point score
        Llo = max(ivs[r][1] for r in live)          # low   = max of the low-ends
        Lhi = max(ivs[r][2] for r in live)          # high  = max of the high-ends
        return Lp, Llo, Lhi, best_r

    # ----- 0. initial small batch for every (pair, gap) -----
    for f in faults:
        for r in rates:
            for g in gaps:
                sample(f, r, g["idx"], init_batch)

    # ----- 1. adaptive rounds -----
    num_rounds = 0
    stop_reason = "decided"
    while True:
        # (a) FREEZE losing rates inside each fault (only in the freeze variant)
        if use_freeze:
            for f in faults:
                live = [r for r in rates if (f, r) not in frozen]
                if len(live) <= 1:
                    continue
                ivs = {r: interval_of_pair(f, r) for r in live}
                best_low = max(ivs[r][1] for r in live)
                for r in live:
                    if ivs[r][2] < best_low:          # high-end below best low-end -> can't win
                        frozen.add((f, r))

        # (b) score + interval per fault, then order by point score
        sc = {f: fault_score(f) for f in faults}
        order = sorted(faults, key=lambda f: sc[f][0], reverse=True)

        # (c) which ADJACENT pairs are not yet confidently ordered? (intervals overlap)
        undecided = set()
        for k in range(len(order) - 1):
            A, B = order[k], order[k + 1]
            if not (sc[A][1] > sc[B][2]):             # NOT (lo[A] > hi[B]) -> overlap
                undecided.add(A); undecided.add(B)

        if not undecided:
            stop_reason = "decided"; break
        if num_rounds >= max_rounds:
            stop_reason = "budget"; break

        # (d) spend more only on undecided faults, only on their LIVE (non-frozen) rates -- and
        # SURGICALLY: for each such pair, refine ONLY its single widest-CI gap (the one term whose
        # uncertainty most inflates the pair's interval). Refining all gaps every round is G x too
        # much work and is what made v1 slower than brute force; one gap per pair per round matches
        # the early fast v1. A gap already at the per-gap sample cap is skipped.
        did_sample = False
        for f in undecided:
            for r in rates:
                if (f, r) in frozen:
                    continue
                widest_gidx, widest_w = None, -1.0
                for g in gaps:
                    a = acc[(f, r, g["idx"])]
                    if gap_settled(a):        # per-estimate epsilon stop: this gap needs no more
                        continue
                    _, lo, hi = wilson(a["hits"], a["n"])
                    # width in LOG space -- the pair score is sum_g log p, so a gap's contribution to
                    # the L-interval width is log(hi) - log(lo), NOT the p-space width hi - lo. These
                    # differ sharply for small p (a low-p gap has tiny p-width but huge log-width and
                    # dominates L's uncertainty), so refine by the log-space width. Floor both ends at
                    # the measurement resolution ~1/n (can't distinguish p below one expected count) so
                    # a 0-hit gap doesn't get an artificially infinite width from log(0)->log(1e-12).
                    res = 1.0 / (a["n"] + 1)
                    w = math.log(max(hi, res)) - math.log(max(lo, res))
                    if w > widest_w:
                        widest_w, widest_gidx = w, g["idx"]
                if widest_gidx is not None:
                    sample(f, r, widest_gidx, round_batch); did_sample = True
        if not did_sample:
            # every contended gap is settled (margin < eps_stop or at cap) yet the fault order still
            # overlaps -> a genuine near-tie no amount of extra sampling would break. Not budget.
            stop_reason = "settled"; break
        num_rounds += 1

    # ----- 2. build the standard output (same schema as the other ufr diagnosers) -----
    log_prob_total_per_fault_and_rate = {f: {r: interval_of_pair(f, r)[0] for r in rates} for f in faults}
    best_rate_per_fault = {}; best_logL_per_fault = {}
    for f in faults:
        br = max(rates, key=lambda r: log_prob_total_per_fault_and_rate[f][r])
        best_rate_per_fault[f] = br
        best_logL_per_fault[f] = log_prob_total_per_fault_and_rate[f][br]
    sorted_faults = sorted(best_logL_per_fault.items(), key=lambda x: x[1], reverse=True)
    T = max(len(gaps), 1)
    total_traces = sum(a["n"] for a in acc.values())

    output = {
        "diagnosis_time_sec": time.time() - t0,
        "diagnosis_time_ms": (time.time() - t0) * 1000,
        "avg_gap_time": 0.0,
        "num_gaps": len(gaps),
        "sorted_faults": sorted_faults,
        "sorted_faults_with_exp_val": [(f, math.exp(L / T)) for f, L in sorted_faults],
        "best_rate_per_fault": best_rate_per_fault,
        "log_prob_total_per_fault_and_rate": log_prob_total_per_fault_and_rate,
        "fault_rate_candidates": fault_rate_candidates,
        "observations": observations,
        "observations_len": len(observations),
        "extra_output": "",
        "v1_total_traces": total_traces,
        "v1_num_rounds": num_rounds,
        "v1_stop_reason": stop_reason,
        "v1_frozen_rate_pairs": len(frozen),
        "v1_use_freeze": use_freeze,
        "v1_eps_stop": eps_stop,
        "v1_min_tries": min_tries,
        "v1_ci_mode": ci_mode,
        "adaptive_total_calls": len(acc),
        "adaptive_avg_real_tries": total_traces / len(acc) if acc else 0,
    }
    print(f"\n===== V1{'-FREEZE' if use_freeze else ''} done: stop={stop_reason} rounds={num_rounds} "
          f"traces={total_traces} frozen_rates={len(frozen)}/{len(faults)*len(rates)} =====")
    return output


def fault_identification_non_deterministic_PO_unknown_fault_rate_V1(
        debug_print, render_mode, instance_seed, ml_model_name, domain_name,
        observations, candidate_fault_modes, fault_rate_candidates, epsilon):
    """v1, NO freezing: each fault judged by its own marginal-CI score; simulate undecided faults'
    ALL rates until adjacent orders separate (or budget). See _v1_run."""
    return _v1_run(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations,
                   candidate_fault_modes, fault_rate_candidates, epsilon, use_freeze=False)


def fault_identification_non_deterministic_PO_unknown_fault_rate_V1_FREEZE(
        debug_print, render_mode, instance_seed, ml_model_name, domain_name,
        observations, candidate_fault_modes, fault_rate_candidates, epsilon):
    """v1 WITH freezing: same as V1, but each round also freezes (stops simulating) any rate that
    provably cannot be its fault's best rate, so budget is spent only on live rates. See _v1_run."""
    return _v1_run(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations,
                   candidate_fault_modes, fault_rate_candidates, epsilon, use_freeze=True)


def fault_identification_non_deterministic_PO(
        debug_print, render_mode,

        instance_seed, ml_model_name,
        domain_name, observations,
        candidate_fault_modes, epsilon, fault_rate = None):

    diagnosis_seed = instance_seed + SIMULATION_OFFSET

    # load trained model as policy
    policy = load_trained_model(domain_name, ml_model_name)

    # load the environment as simulator
    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)  # instance_seed IS the block base (slot 0)
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # MC seeding needs every gap-start index < MAX_STATES (the seed stride) so gaps stay on
    # distinct residue classes mod MAX_STATES; guarantee the trajectory fits.
    assert len(observations) <= MAX_STATES, (
        f"trajectory length {len(observations)} exceeds MAX_STATES {MAX_STATES}; "
        f"raise MAX_STATES in h_consts.")

    # initialize time counting
    diagnosis_time_sec_start = time.time()
    fault_prob_hat_for_step = {}

    last_observed_index = 0
    gap_times = []
    num_gaps = 0
    num_of_observed_states = 1
    adaptive_stats = []


    for i in range(1, len(observations)):
        if debug_print:
            print(f"DEBUG: iteration {i}")


        if observations[i] is None:
            continue

        fault_prob_hat_for_step[i] = {}

        num_of_observed_states+=1
        current_gap_length = i - last_observed_index

        min_tries = 100 + 10 * current_gap_length
        batch_size = 50
        # min_tries = 20
        # batch_size = 10
        scale = max(1.0, (0.025 / epsilon) ** 2)

        base_max = int(2500 * scale)
        gap_bonus = int(150 * current_gap_length * scale)
        max_tries = base_max + gap_bonus

        current_gap_seed = diagnosis_seed + last_observed_index
        top_seed_offset_during_iterations = (max_tries + 50) * MAX_STATES

        assert current_gap_seed + top_seed_offset_during_iterations  < instance_seed + SEED_BLOCK, (
            f"MC seeds overflow this instance's block: top offset {current_gap_seed + top_seed_offset_during_iterations} >= "
            f"SEED_BLOCK {instance_seed + SEED_BLOCK} (epsilon={epsilon}, max_tries={max_tries}, "
            f"MAX_STATES={MAX_STATES}). Increase SEED_BLOCK (e.g. 10_000_000).")

        # print(f"epsilon1 ={epsilon}, gap length ={current_gap_length} so max_tries = {max_tries}")

        gap_start_time = time.time()
        for curr_fault_key in candidate_fault_modes:

            res = simulate_m_traces_adaptive_monte_carlo(observations[last_observed_index],
                                                   observations[i],
                                                   current_gap_length,
                                                   candidate_fault_modes[curr_fault_key],
                                                   fault_rate,
                                                   domain_name,
                                                   current_gap_seed,  # gap residue class; per-trace stride = MAX_STATES
                                                   simulator,
                                                   policy,
                                                   comparators[domain_name],
                                                   debug_print,
                                                   min_tries=min_tries,
                                                   max_tries=max_tries,
                                                   batch_size=batch_size,
                                                   epsilon=epsilon)

            p_hat = max(res["p_hat"], 1e-12)
            fault_prob_hat_for_step[i][curr_fault_key] = p_hat
            adaptive_stats.append(res)

            if res["stop_reason"] == "max_tries":
                print(f"MAX HIT | gap={res['trace_length']} "
                      f"tries={res['num_of_tries']} "
                      f"p={res['p_hat']:.4f} "
                      f"margin={res['margin']}"
                      f"curr_fault_key={curr_fault_key}")
            else:
                pass
                """
                lower = max(0.0, res["p_hat"] - res["margin"])
                upper = min(1.0, res["p_hat"] + res["margin"])

                print(f"CONF STOP | gap={res['trace_length']} "
                      f"tries={res['num_of_tries']} "
                      f"p_hat={res['p_hat']:.4f} "
                      f"95% CI=[{lower:.4f}, {upper:.4f}] "
                      f"(±{res['margin']:.4f}) "
                      f"curr_fault_key={curr_fault_key}")
                """


        gap_end_time = time.time()
        gap_times.append(gap_end_time - gap_start_time)
        num_gaps += 1

        last_observed_index = i

    log_prob_total_per_fault = {}

    # step 1
    for f in candidate_fault_modes:
        log_prob = 0.0
        for i in range(1, len(observations)):
            if observations[i] is None:
                continue
            p = fault_prob_hat_for_step[i][f]
            log_prob += math.log(p)
        log_prob_total_per_fault[f] = log_prob

    # step 3 (same logic, just naming)
    best_logL_per_fault = {}
    best_rate_per_fault = {}

    for f in candidate_fault_modes:
        best_rate_per_fault[f] = fault_rate
        best_logL_per_fault[f] = log_prob_total_per_fault[f]

    # Sort faults by log-likelihood (descending)
    sorted_faults = sorted(best_logL_per_fault.items(), key=lambda x: x[1], reverse=True)
    T = num_of_observed_states - 1

    # sorted_faults: List[Tuple[fault_key, logL]]
    sorted_faults_geo = [(fault, math.exp(logL / T)) for fault, logL in sorted_faults]

    print("Fault scores (higher is better):")
    extra_output = ""
    for fault, logL in sorted_faults:
        rate = best_rate_per_fault[fault]
        curr_output = (f"Fault: {fault}, logL: {logL:.6f}, "
                       f"Exp of LogL: {math.exp(logL):.6f}, "
                       f"Normalized LogL: {(logL / T):.6f}, "
                       f"Exp of Normalized LogL: {math.exp(logL / T):.6f}, "
                       f"Best Rate: {rate}")

        print(curr_output)
        extra_output += curr_output
        extra_output += "\n"

    diagnosis_time_sec_end = time.time()
    diagnosis_time_sec = diagnosis_time_sec_end - diagnosis_time_sec_start
    diagnosis_time_ms = diagnosis_time_sec * 1000

    avg_gap_time = sum(gap_times) / len(gap_times) if gap_times else 0.0

    output = {}
    output["diagnosis_time_sec"] = diagnosis_time_sec
    output["diagnosis_time_ms"] = diagnosis_time_ms

    output["avg_gap_time"] = avg_gap_time
    output["num_gaps"] = num_gaps

    output["sorted_faults"] = sorted_faults
    output["sorted_faults_with_exp_val"] = sorted_faults_geo
    output["observations"] = observations
    output["observations_len"] = len(observations)
    output["extra_output"] = extra_output

    total_calls = len(adaptive_stats)
    adaptive_total_real_tries = sum(
        s["num_of_tries"]
        for s in adaptive_stats
    )
    avg_tries = adaptive_total_real_tries / total_calls
    adaptive_max_tries_avg = sum(s["max_tries"] for s in adaptive_stats) / total_calls
    adaptive_min_tries_avg = sum(s["min_tries"] for s in adaptive_stats) / total_calls
    max_hits = sum(1 for s in adaptive_stats if s["stop_reason"] == "max_tries")
    conf_hits = total_calls - max_hits

    margins = [
        s["margin"]
        for s in adaptive_stats
        if s["margin"] is not None
    ]

    avg_margin = sum(margins) / len(margins) if margins else None

    avg_p_hat = (
        sum(s["p_hat"] for s in adaptive_stats) / len(adaptive_stats)
        if adaptive_stats else None
    )

    output["adaptive_total_calls"] = total_calls
    output["adaptive_avg_real_tries"] = avg_tries
    output["adaptive_total_real_tries"] = adaptive_total_real_tries
    output["adaptive_max_tries_avg"] = adaptive_max_tries_avg
    output["adaptive_min_tries_avg"] = adaptive_min_tries_avg
    output["adaptive_max_stops"] = max_hits
    output["adaptive_conf_stops"] = conf_hits
    output["adaptive_max_stop_rate"] = max_hits / total_calls
    output["adaptive_ever_hit_max"] = bool(max_hits > 0)
    output["adaptive_avg_margin"] = avg_margin
    output["adaptive_avg_p_hat"] = avg_p_hat

    print("\n========= ADAPTIVE MC DEBUG =========")
    print(f"Total calls: {total_calls}")
    print(f"Avg tries: {avg_tries:.2f}")
    print(f"Confidence stops: {conf_hits}")
    print(f"Max stops: {max_hits}")
    print(f"Max stop rate: {max_hits / total_calls:.3f}")


    return output



def SIF(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy

    policy = load_trained_model(domain_name, ml_model_name)

    # load the environment as simulator

    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], A_j, S_0]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    for i in range(1, len(observations)):
        ts1 = time.time()
        irrelevant_keys = []
        new_relevant_keys = {}
        for key_j in G.keys():
            a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
            a_gag_i = int(a_gag_i)
            a_gag_i_j = G[key_j][0](a_gag_i)

            # apply the normal and the faulty action on the reconstructed states, respectively
            simulator.set_state(G[key_j][2])
            S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
            simulator.set_state(G[key_j][2])
            S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
            if observations[i] is not None:
                # the case where there is an observation that can be checked
                S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i))
                    G[key_j][2] = S_gag_i
                elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i))
                    G[key_j][2] = S_gag_i
                elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    irrelevant_keys.append(key_j)
                elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i_j))
                    G[key_j][2] = S_gag_i_j
            else:
                # the case where there is no observation to be checked - insert the normal action and state to the original key
                if debug_print:
                    print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                G[key_j][1].append(int(a_gag_i))
                G[key_j][2] = S_gag_i
                if a_gag_i != a_gag_i_j:
                    # if the action was changed - create new trajectory and insert it as well
                    if debug_print:
                        print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    A_j_to_fault = copy.deepcopy(G[key_j][1])
                    A_j_to_fault[-1] = a_gag_i_j
                    k_j = key_j.split('_')[0]
                    new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                    I[k_j] = I[k_j] + 1
        # add new relevant fault modes
        for key in new_relevant_keys:
            G[key] = new_relevant_keys[key]
        # remove the irrelevant fault modes
        for key in irrelevant_keys:
            G.pop(key)
        te1 = time.time()
        diagnosis_runtime_sec += te1 - ts1

        # filter out similar trajectories (applies to taxi only)
        if domain_name == "Taxi_v3":
            FG = {}
            for key in G.keys():
                key_raw = key.split('_')[0]
                state = G[key][2]
                if not fm_and_state_in_set(key_raw, state, FG):
                    FG[key] = G[key]
            G = FG

        # update the maximum size of G
        G_max_size = max(G_max_size, len(G))

        if debug_print:
            if observations[i] is not None:
                print(f'STEP {i}/{len(observations)}: OBSERVED')
            else:
                print(f'STEP {i}/{len(observations)}: HIDDEN')
            print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
            print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
            print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = j - i
            i = j
    sorted_index_pairs = sorted(index_pairs.keys(), key=lambda k: (index_pairs[k], k))
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in sorted_index_pairs]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU2(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = j - i
            i = j
    useful_index_pairs = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            useful_index_pairs[pair] = index_pairs[pair]
    sorted_useful_index_pairs = sorted(useful_index_pairs.keys(), key=lambda k: (useful_index_pairs[k], k))
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in sorted_useful_index_pairs]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU3(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    index_pairs_failed = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], set()]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    index_pairs_failed_sorted_useful = {}
    action_types_combined = set()
    for pair in index_pairs_failed_sorted:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b+i}: a {a}')
            index_pairs_failed_sorted[pair][1].add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if len(index_pairs_failed_sorted[pair][1].difference(action_types_combined)) != 0:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            action_types_combined.update(index_pairs_failed_sorted[pair][1])
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU4(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    index_pairs_failed = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], set()]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    index_pairs_failed_sorted_useful = {}
    action_types_combined = set()
    for pair in index_pairs_failed_sorted:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b+i}: a {a}')
            index_pairs_failed_sorted[pair][1].add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if len(index_pairs_failed_sorted[pair][1].difference(action_types_combined)) != 0:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            action_types_combined.update(index_pairs_failed_sorted[pair][1])
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed_sorted_useful.keys():
        actions = index_pairs_failed_sorted_useful[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU5(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    # sort the pairs that failed according to length
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted.keys()]
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed_sorted.keys():
        actions = index_pairs_failed_sorted[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU6(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    # sort the pairs that failed according to length
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed_sorted.keys():
        actions = index_pairs_failed_sorted[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    # filter unuseful failed pairs (this can be done here after we filtered the fault modes)
    index_pairs_failed_sorted_useful = {}
    pairs_unique_action_sets = []
    for pair in index_pairs_failed_sorted:
        if index_pairs_failed_sorted[pair][1] not in pairs_unique_action_sets:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            pairs_unique_action_sets.append(index_pairs_failed_sorted[pair][1])
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU7(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed.keys()]
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU8(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed * SEED_BLOCK)  # trajectory block base
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    initialization_runtime_sec = 0.0
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    initialization_runtime_sec += te0 - ts0

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts1 = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j
    # compute the conflicts - that is, the index pairs that failed
    index_pairs_failed = {}
    for pair in index_pairs:
        action_types_pair = set()
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            action_types_pair.add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], action_types_pair]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
        # print(f'pair {pair}: OK\n')
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed.keys()]
    # filter fault modes that are not compatible with the healthy registered actions
    for pair in index_pairs_failed.keys():
        actions = index_pairs_failed[pair][1]
        fms_to_remove = []
        for fm in G.keys():
            fm_raw = fm.split('_')[0]
            fm_list = eval(fm_raw)
            to_remove = True
            for a in actions:
                if fm_list[a] != a:
                    to_remove = False
            if to_remove:
                fms_to_remove.append(fm)
        for fm in fms_to_remove:
            G.pop(fm)
    te1 = time.time()
    initialization_runtime_sec += te1 - ts1

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
            ts2 = time.time()
            irrelevant_keys = []
            new_relevant_keys = {}
            for key_j in G.keys():
                a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
                a_gag_i = int(a_gag_i)
                a_gag_i_j = G[key_j][0](a_gag_i)

                # apply the normal and the faulty action on the reconstructed states, respectively
                simulator.set_state(G[key_j][2])
                S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
                simulator.set_state(G[key_j][2])
                S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
                if observations[i] is not None:
                    # the case where there is an observation that can be checked
                    S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                    S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                    if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j cannot change a_gag_i
                        if debug_print:
                            print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        irrelevant_keys.append(key_j)
                    elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                        # a_gag_i     changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
                        k_j = key_j.split('_')[0]
                        new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
                        I[k_j] = I[k_j] + 1
            # add new relevant fault modes
            for key in new_relevant_keys:
                G[key] = new_relevant_keys[key]
            # remove the irrelevant fault modes
            for key in irrelevant_keys:
                G.pop(key)
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

            # filter out similar trajectories (applies to taxi only)
            if domain_name == "Taxi_v3":
                FG = {}
                for key in G.keys():
                    key_raw = key.split('_')[0]
                    state = G[key][2]
                    if not fm_and_state_in_set(key_raw, state, FG):
                        FG[key] = G[key]
                G = FG

            # update the maximum size of G
            G_max_size = max(G_max_size, len(G))

            if debug_print:
                if observations[i] is not None:
                    print(f'STEP {i}/{len(observations)}: OBSERVED')
                else:
                    print(f'STEP {i}/{len(observations)}: HIDDEN')
                print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
                print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
                print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

            if len(G) == 1:
                if debug_print:
                    print(f"i broke at {i}")
                break

    # finilizing the runtime in ms
    initialization_runtime_ms = initialization_runtime_sec * 1000
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "init_rt_sec": initialization_runtime_sec,
        "init_rt_ms": initialization_runtime_ms,
        "diag_rt_sec": diagnosis_runtime_sec,
        "diag_rt_ms": diagnosis_runtime_ms,
        "totl_rt_sec": initialization_runtime_sec + diagnosis_runtime_sec,
        "totl_rt_ms": initialization_runtime_ms + diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


diagnosers = {
    # new fault models
    "W": W,
    "SN": SN,
    "SIF": SIF,
    "SIFU": SIFU,
    "SIFU2": SIFU2,
    "SIFU3": SIFU3,
    "SIFU4": SIFU4,
    "SIFU5": SIFU5,
    "SIFU6": SIFU6,
    "SIFU7": SIFU7,
    "SIFU8": SIFU8
}
