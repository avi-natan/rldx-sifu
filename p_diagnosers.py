import copy
import math
import time

import gym

from h_consts import DETERMINISTIC, SEED_BLOCK, SIMULATION_OFFSET
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

    # Seed the env RNG per trace so rainy-slip outcomes are a pure function of this trace's
    # global seed (instance_seed already carries the +i offset from the caller). set_state
    # then overrides the sampled start; only the slip stream matters downstream.
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

import random


def simulate_m_traces(starting_state, observed_next_state, trace_length,
                      num_of_tries, fault_mode, fault_rate,
                      domain_name, instance_seed,
                      simulator, model, comparator, debug_print):

    num_of_hits = 0

    for i in range(num_of_tries):

        rng = random.Random(instance_seed + i)
        # Pass the SAME per-trace seed (instance_seed + i) used for fault-firing so the
        # simulator's env RNG (rainy slips) is reseeded per trace -> each trace is a pure
        # function of its global index, independent of candidate order and adaptive trace
        # counts. Makes runs paired across epsilon and fully reproducible.
        sim_next_state = execute_one_trace(starting_state, trace_length,
                      fault_mode, fault_rate,
                      domain_name, instance_seed + i, rng,
                      simulator, model, debug_print)

        simulated_state_equals_observed = comparator(sim_next_state, observed_next_state)

        if simulated_state_equals_observed:
            num_of_hits += 1

    return num_of_hits

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

    while True:
        num_of_hits += simulate_m_traces(starting_state, observed_next_state, trace_length,
                      batch_size, fault_mode, fault_rate,
                      domain_name, instance_seed + curr_num_of_tries,
                      simulator, model, comparator, debug_print)
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

    policy = load_trained_model(domain_name, ml_model_name)

    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)  # instance_seed IS the block base (slot 0)
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

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
                    instance_seed + SIMULATION_OFFSET,  # MC trace range, disjoint from slots 0-3
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



def fault_identification_non_deterministic_PO(
        debug_print, render_mode,

        instance_seed, ml_model_name,
        domain_name, observations,
        candidate_fault_modes, epsilon, fault_rate = None):

    # load trained model as policy
    policy = load_trained_model(domain_name, ml_model_name)

    # load the environment as simulator
    simulator = make_wrapped_env(domain_name, render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)  # instance_seed IS the block base (slot 0)
    S_0 = initial_obs  # use the seeded reset's start (no second, unseeded reset)
    assert comparators[domain_name](observations[0], S_0)

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

        # print(f"epsilon1 ={epsilon}, gap length ={current_gap_length} so max_tries = {max_tries}")

        gap_start_time = time.time()
        for curr_fault_key in candidate_fault_modes:

            res = simulate_m_traces_adaptive_monte_carlo(observations[last_observed_index],
                                                   observations[i],
                                                   current_gap_length,
                                                   candidate_fault_modes[curr_fault_key],
                                                   fault_rate,
                                                   domain_name,
                                                   instance_seed + SIMULATION_OFFSET,  # MC trace range, disjoint from slots 0-3
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
