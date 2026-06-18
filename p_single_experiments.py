import math
import random
from datetime import datetime, timedelta

import h_rl_models
from frozen_lake_random_envs import load_pairs_from_json, print_map_and_policy
from h_fault_model_generator import FaultModelGeneratorDiscrete
from h_wrappers import DOMAIN_KWARGS
from hard_taxi_data import benchmark_seeds, get_instance
from p_diagnosers import diagnosers
from p_executor import execute_manual
from p_pipeline import run_SIF_single_experiment, run_SN_single_experiment, run_W_single_experiment, \
    run_SIFU_single_experiment, run_SIFU2_single_experiment, run_SIFU3_single_experiment, run_SIFU4_single_experiment, \
    run_SIFU5_single_experiment, run_SIFU6_single_experiment, run_SIFU7_single_experiment, run_SIFU8_single_experiment, \
    separate_trajectory, calculate_largest_hidden_gap, mask_states, rank_diagnoses_WFM, rank_diagnoses_SFM, \
    prepare_record, write_records_to_excel, exper_100_write_records_to_excel, run_NON_DETERMINSTIC_single_experiment_FO, \
    exper_write_records_to_excel_ind, run_NON_DETERMINSTIC_single_experiment_PO, domain_results_dir


# =================================================================================================
# ============================================ manual =============================================
# =================================================================================================
def single_experiment_manual():
    domain_name = "CartPole_v1"
    ml_model_name = "PPO"  # "PPO", "DQN"
    render_mode = "rgb_array"  # "human", "rgb_array"
    debug_print = False
    execution_fault_mode_name = "[0,0]"
    instance_seed = 6
    fault_probability = 0.4
    percent_visible_states = 30

    # ###########################
    faulty_actions_indices = [7, 20, 21, 23, 25, 27, 31, 32, 36, 39, 40, 41]
    execution_length = 40
    observation_mask = [0, 4, 6, 7, 9, 10, 15, 17, 32, 33, 37, 39, 40]
    diagnoser_name = "SIFU3"
    candidate_fault_modes_names = [
        '[0,0]',
        '[1,0]'
    ]
    # ###########################

    fault_mode_generator = FaultModelGeneratorDiscrete()
    trajectory_execution, faulty_actions_indices = execute_manual(domain_name,
                                                                  debug_print,
                                                                  execution_fault_mode_name,
                                                                  instance_seed,
                                                                  fault_probability,
                                                                  render_mode,
                                                                  ml_model_name,
                                                                  fault_mode_generator,
                                                                  execution_length,
                                                                  faulty_actions_indices)
    registered_actions, observations = separate_trajectory(trajectory_execution)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    masked_observations = mask_states(observations, observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    candidate_fault_modes = {}
    for fmn in candidate_fault_modes_names:
        fm = fault_mode_generator.generate_fault_model(fmn)
        candidate_fault_modes[fmn] = fm

    diagnoser = diagnosers[diagnoser_name]

    raw_output = diagnoser(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    if diagnoser_name == "W":
        output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)
    else:
        output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    records = []
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, candidate_fault_modes_names, len(candidate_fault_modes_names),
                            render_mode, ml_model_name, execution_length, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, diagnoser_name, longest_hidden_state_sequence)
    records.append(record)
    write_records_to_excel(records, f"single_experiment_manual_{domain_name.split('_')[0]}_{diagnoser_name}")

    print(f'duration in ms: {raw_output["diag_rt_ms"]}')




# =================================================================================================
# ========================================== LunerLander ==========================================
# =================================================================================================
def single_experiment_LunarLander_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "LunarLander_v2"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_LunarLander_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "LunarLander_v2"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3]",  # shutting down jets
            "[0,1,0,3]",
            "[0,1,2,0]",
            "[0,0,0,3]",
            "[0,0,2,0]",
            "[0,1,0,0]",
            "[0,0,0,0]",
            "[0,3,2,1]",  # swapping jets
            "[0,2,1,3]",
            "[0,1,3,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_LunarLander_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "LunarLander_v2"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3]",  # shutting down jets
            "[0,1,0,3]",
            "[0,1,2,0]",
            "[0,0,0,3]",
            "[0,0,2,0]",
            "[0,1,0,0]",
            "[0,0,0,0]",
            "[0,3,2,1]",  # swapping jets
            "[0,2,1,3]",
            "[0,1,3,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# ============================================ Acrobot ============================================
# =================================================================================================
def single_experiment_Acrobot_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')

import math
import numpy as np
import matplotlib.pyplot as plt

def _extract_T_and_rank(records):
    """Return list of (T_steps, rank) for valid records."""
    out = []
    for r in records:
        rank = r.get("real_fault_rank", None)
        obs_len = r.get("observations_len", None)
        if rank is None or obs_len is None:
            continue
        T = int(obs_len) - 1  # steps
        if T <= 0:
            continue
        out.append((T, float(rank)))
    return out

def plot_sorted_lengths_running_avg(records, window=None, title="Rank vs trajectory length (sorted)"):
    """
    Plot 1 (your idea):
      - sort experiments by trajectory length
      - scatter (T, rank)
      - running average line:
          * cumulative mean if window=None
          * rolling mean with given window otherwise
    """

    data = _extract_T_and_rank(records)
    if not data:
        raise ValueError("No valid (observations_len, real_fault_rank) pairs found.")

    data.sort(key=lambda x: x[0])
    T = np.array([t for t, _ in data], dtype=float)
    R = np.array([r for _, r in data], dtype=float)

    plt.figure()
    plt.scatter(T, R, s=18)

    if window is None:
        # cumulative average
        running_avg = np.cumsum(R) / np.arange(1, len(R) + 1)
        plt.plot(T, running_avg)
        plt.legend(["running avg (cumulative)", "points"])
    else:
        # rolling average (centered)
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        kernel = np.ones(w) / w
        rolling = np.convolve(R, kernel, mode="valid")
        T_mid = T[w-1:]  # align to end (simple alignment)
        plt.plot(T_mid, rolling)
        plt.legend([f"running avg (window={w})", "points"])

    plt.xlabel("Trajectory length (steps)")
    plt.ylabel("Real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

def plot_binned_avg_rank(records, bin_size=20, max_steps=200, title="Average rank vs trajectory length (binned)", show_errorbars=True):
    """
    Plot 2:
      - bin by trajectory length
      - average rank in each bin
      - optionally show std error bars and counts
    """
    data = _extract_T_and_rank(records)
    if not data:
        raise ValueError("No valid (observations_len, real_fault_rank) pairs found.")

    bins = list(range(0, max_steps + bin_size, bin_size))
    xs, ys, errs, ns = [], [], [], []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        ranks = [rank for (t, rank) in data if b0 <= t < b1]
        if not ranks:
            continue
        ranks = np.array(ranks, dtype=float)
        xs.append((b0 + b1) / 2.0)
        ys.append(float(np.mean(ranks)))
        ns.append(int(len(ranks)))

        if show_errorbars:
            # standard error of the mean (SEM)
            sem = float(np.std(ranks, ddof=1) / math.sqrt(len(ranks))) if len(ranks) > 1 else 0.0
            errs.append(sem)

    plt.figure()
    if show_errorbars:
        plt.errorbar(xs, ys, yerr=errs, fmt="o-")
    else:
        plt.plot(xs, ys, "o-")

    plt.xlabel("Trajectory length bin center (steps)")
    plt.ylabel("Average real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

    # optional: annotate counts above points
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")


import matplotlib.pyplot as plt

def plot_rank_vs_fault_occurrences_scatter(records, title="Rank vs fault occurrences (scatter)"):
    xs, ys = [], []
    for r in records:
        x = r.get("faulty_actions_indices_len", None)
        y = r.get("real_fault_rank", None)
        if x is None or y is None:
            continue
        xs.append(int(x))
        ys.append(float(y))

    plt.figure()
    plt.scatter(xs, ys, s=20)
    plt.xlabel("Fault occurrences (trigger count)")
    plt.ylabel("Real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

import numpy as np
import matplotlib.pyplot as plt

def plot_avg_rank_vs_fault_rate(records, title="Average rank vs fault rate"):
    # collect ranks per fault rate
    rates_to_ranks = {}
    for r in records:
        rate = r.get("real_fault_prob", None)
        rank = r.get("real_fault_rank", None)
        if rate is None or rank is None:
            continue
        rates_to_ranks.setdefault(float(rate), []).append(float(rank))

    rates = sorted(rates_to_ranks.keys())
    avg_rank = [float(np.mean(rates_to_ranks[rt])) for rt in rates]
    counts = [len(rates_to_ranks[rt]) for rt in rates]

    plt.figure()
    plt.plot(rates, avg_rank, "o-")
    plt.xlabel("Fault rate (real_fault_prob)")
    plt.ylabel("Average real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

    # annotate sample sizes
    for x, y, n in zip(rates, avg_rank, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

import numpy as np
import matplotlib.pyplot as plt
import math


def plot_rank_vs_visibility_rate(records, title="Rank vs visibility rate"):
    xs, ys = [], []

    for r in records:
        x = r.get("percent_visible_states", None)
        y = r.get("real_fault_rank", None)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    if not xs:
        raise ValueError("No valid data for percent_visible_states vs real_fault_rank.")

    plt.figure()
    plt.scatter(xs, ys, s=20)
    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

def plot_avg_rank_vs_visibility_rate(records, title="Average rank vs visibility rate"):
    vis_to_ranks = {}

    for r in records:
        vis = r.get("percent_visible_states", None)
        rank = r.get("real_fault_rank", None)
        if vis is None or rank is None:
            continue
        vis = float(vis)
        vis_to_ranks.setdefault(vis, []).append(float(rank))

    if not vis_to_ranks:
        raise ValueError("No valid data for percent_visible_states vs real_fault_rank.")

    xs = sorted(vis_to_ranks.keys())
    ys = [float(np.mean(vis_to_ranks[x])) for x in xs]
    counts = [len(vis_to_ranks[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs)
    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

    for x, y, n in zip(xs, ys, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")


def plot_avg_time_vs_visibility_rate(records, title="Average diagnosis time vs visibility rate"):
    vis_to_times = {}

    for r in records:
        vis = r.get("percent_visible_states", None)
        t = r.get("diagnosis_time_sec", None)
        if vis is None or t is None:
            continue
        vis = float(vis)
        vis_to_times.setdefault(vis, []).append(float(t))

    if not vis_to_times:
        raise ValueError("No valid data for percent_visible_states vs diagnosis_time_sec.")

    xs = sorted(vis_to_times.keys())
    ys = [float(np.mean(vis_to_times[x])) for x in xs]
    counts = [len(vis_to_times[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs)
    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average diagnosis time (sec)")
    plt.title(title)
    plt.grid(True)

    for x, y, n in zip(xs, ys, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

def plot_rank_vs_num_observed_states(records, title="Rank vs number of observed states"):
    xs, ys = [], []

    for r in records:
        x = r.get("num_observed_states", None)
        y = r.get("real_fault_rank", None)
        if x is None or y is None:
            continue
        xs.append(int(x))
        ys.append(float(y))

    if not xs:
        raise ValueError("No valid data for num_observed_states vs real_fault_rank.")

    plt.figure()
    plt.scatter(xs, ys, s=20)
    plt.xlabel("Number of observed states")
    plt.ylabel("Real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

def plot_rank_vs_largest_gap(records, title="Rank vs largest hidden gap"):
    xs, ys = [], []

    for r in records:
        x = r.get("largest_gap", None)
        y = r.get("real_fault_rank", None)
        if x is None or y is None:
            continue
        xs.append(int(x))          # ensure integer
        ys.append(float(y))

    if not xs:
        raise ValueError("No valid data for largest_gap vs real_fault_rank.")

    plt.figure()
    plt.scatter(xs, ys, s=20)

    # force integer ticks on x-axis
    min_x, max_x = min(xs), max(xs)
    plt.xticks(range(min_x, max_x + 1))

    plt.xlabel("Largest hidden gap")
    plt.ylabel("Real fault rank (lower is better)")
    plt.title(title)
    plt.grid(True)

def save_all_plots(prefix="rank_plots"):
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        fig.savefig(f"{prefix}_{i}.png", dpi=200, bbox_inches="tight")

# ---- Example usage ----
# records = ... (your list of dicts)

# Plot 1: sorted lengths + cumulative average
# plot_sorted_lengths_running_avg(records, window=None)

# Plot 1 alternative: sorted lengths + rolling avg window=30
# plot_sorted_lengths_running_avg(records, window=30)

# Plot 2: binned (bin size 20)
# plot_binned_avg_rank(records, bin_size=20, max_steps=200, show_errorbars=True)

# plt.show()
# save_all_plots("frozenlake_rank")


def multiple_experiment_FrozenLake_NON_DETERMINSTIC_FO():

    global HARD_CODED_POLICY
    loaded = load_pairs_from_json("frozenlake_100_pairs_risk_averse_slippery.json")
    diagnosis_runtimes_ms = []
    records = []
    NUM_TRIES = 49

    for i in range(NUM_TRIES):

        map_desc, hardcoded_policy = loaded[i]
        DOMAIN_KWARGS["FrozenLake_v1"]["desc"] = map_desc
        DOMAIN_KWARGS["FrozenLake_v1"]["is_slippery"] = True
        h_rl_models.HARD_CODED_POLICY = hardcoded_policy
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')

        domain_name = "FrozenLake_v1"
        ml_model_name = "PPO"                               # "PPO", "DQN"
        render_mode = "rgb_array"                           # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False

        instance_seed = 10+i
        num_candidate_fault_modes = 10

        rng = random.Random(instance_seed)
        percent_visible_states = 100
        possible_fault_mode_names = [
                "[0,0,2,3]",
                "[0,3,2,3]",
                "[0,1,0,3]",
                "[0,1,3,3]",
                "[0,2,2,3]",

                "[0,3,0,3]",
                "[0,3,3,3]",
                "[0,0,0,3]",
                "[0,0,3,3]",
                "[0,2,1,3]",

                "[0,0,0,0]",
                "[1,1,1,1]",
                "[2,2,2,2]",
                "[3,3,3,3]"
              ]

        execution_fault_mode_name = rng.choice(possible_fault_mode_names)

        # 1.0 mean Non-intermittent fault - faults always occur
        fault_rate_list = [0.5, 0.8]
        for fault_rate in fault_rate_list:
            output = run_NON_DETERMINSTIC_single_experiment_FO( domain_name=domain_name,
                                                             ml_model_name=ml_model_name,
                                                             render_mode=render_mode,
                                                             max_exec_len=max_exec_len,
                                                             debug_print=debug_print,
                                                             execution_fault_mode_name=execution_fault_mode_name,
                                                             instance_seed=instance_seed,
                                                             fault_probability=fault_rate,
                                                             percent_visible_states=percent_visible_states,
                                                             possible_fault_mode_names=possible_fault_mode_names,
                                                             num_candidate_fault_modes=num_candidate_fault_modes)
            if not output:
                continue

            output["experiment_num"] = i + 1
            output["real_fault_prob"] = fault_rate
            output["map_desc"] = map_desc
            output["hardcoded_policy"] = hardcoded_policy
            output["domain_name"] = domain_name

            records.append(output)

        """
            output["sorted_faults"] = sorted_faults
            output["sorted_faults_with_exp_val"] = sorted_faults_geo
            output["observations"] = observations
            output["observations_len"] = len(observations)
            output["extra_output"] = extra_output
            output["execution_fault_in_top1"] = execution_fault_in_top1
            output["execution_fault_in_top2"] = execution_fault_in_top2
            output["execution_fault_in_top3"] = execution_fault_in_top3
            output["faulty_actions_indices"] = faulty_actions_indices
            output["faulty_actions_indices_len"] = len(faulty_actions_indices)
            output["faulty_actions_indices_not_zero"] = bool(len(faulty_actions_indices) > 0)
        """
    exper_write_records_to_excel_ind(records, "frozen_lake_non_deterministic")


    plot_sorted_lengths_running_avg(records, window=None)
    plot_binned_avg_rank(records, bin_size=20, max_steps=200, show_errorbars=True)
    plot_rank_vs_fault_occurrences_scatter(records)
    plot_avg_rank_vs_fault_rate(records)

    save_all_plots("frozenlake_rank")
    plt.show()

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))



def plot_avg_gap_time_vs_visibility_rate(records, title="Average gap time vs visibility rate"):
    vis_to_times = {}

    for r in records:
        vis = r.get("percent_visible_states", None)
        t = r.get("avg_gap_time", None)
        if vis is None or t is None:
            continue
        vis = float(vis)
        vis_to_times.setdefault(vis, []).append(float(t))

    if not vis_to_times:
        raise ValueError("No valid data for percent_visible_states vs avg_gap_time.")

    xs = sorted(vis_to_times.keys())
    ys = [float(np.mean(vis_to_times[x])) for x in xs]
    counts = [len(vis_to_times[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs)
    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average time per gap (sec)")
    plt.title(title)
    plt.grid(True)

    for x, y, n in zip(xs, ys, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

def plot_num_gaps_vs_visibility_rate(records, title="Number of gaps vs visibility rate"):
    vis_to_num_gaps = {}

    for r in records:
        vis = r.get("percent_visible_states", None)
        num_gaps = r.get("num_gaps", None)
        if vis is None or num_gaps is None:
            continue
        vis = float(vis)
        vis_to_num_gaps.setdefault(vis, []).append(float(num_gaps))

    if not vis_to_num_gaps:
        raise ValueError("No valid data for percent_visible_states vs num_gaps.")

    xs = sorted(vis_to_num_gaps.keys())
    ys = [float(np.mean(vis_to_num_gaps[x])) for x in xs]
    counts = [len(vis_to_num_gaps[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs)
    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average number of gaps")
    plt.title(title)
    plt.grid(True)

    for x, y, n in zip(xs, ys, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

def multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO(epsilon=0.03, unknown_fault_rate=False, maps_num=49, run_folder=None):

    global HARD_CODED_POLICY
    loaded = load_pairs_from_json("frozenlake_100_pairs_risk_averse_slippery.json")
    diagnosis_runtimes_ms = []
    records = []

    if unknown_fault_rate:
        fault_rate_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fault_rate_candidates = [0.1, 0.3, 0.5, 0.7, 0.8]
    else:
        fault_rate_candidates = None



    NUM_MAPS = maps_num

    msg = (
        f"Running PO diagnosis | "
        f"epsilon={epsilon} | "
        f"unknown_fault_rate={unknown_fault_rate} | "
        f"num_maps={NUM_MAPS}"
    )

    if fault_rate_candidates is not None:
        msg += f" | fault_rate_candidates={fault_rate_candidates}"

    print(msg+ "\n\n")

    for i in range(NUM_MAPS):

        map_desc, hardcoded_policy = loaded[i]
        DOMAIN_KWARGS["FrozenLake_v1"]["desc"] = map_desc
        DOMAIN_KWARGS["FrozenLake_v1"]["is_slippery"] = True
        h_rl_models.HARD_CODED_POLICY = hardcoded_policy
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'================================= {dt_string}: MAP {i+1}/{NUM_MAPS} START =================================')

        domain_name = "FrozenLake_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = True

        instance_seed = 10+i
        num_candidate_fault_modes = 10

        rng = random.Random(instance_seed)
        possible_fault_mode_names = [
                "[0,0,2,3]",
                "[0,3,2,3]",
                "[0,1,0,3]",
                "[0,1,3,3]",
                "[0,2,2,3]",

                "[0,3,0,3]",
                "[0,3,3,3]",
                "[0,0,0,3]",
                "[0,0,3,3]",
                "[0,2,1,3]",

                "[0,0,0,0]",
                "[1,1,1,1]",
                "[2,2,2,2]",
                "[3,3,3,3]"
              ]

        execution_fault_mode_name = rng.choice(possible_fault_mode_names)

        # 1.0 mean Non-intermittent fault - faults always occur
        fault_rate_list = [0.5, 0.8]
        percent_visible_states_list = [20, 40, 60, 80, 100]

        for percent_visible_states in percent_visible_states_list:
            for fault_rate in fault_rate_list:

                print(f'======================= START SINGLE EXPERIMENT MAP {i + 1}/{NUM_MAPS} with FR={fault_rate}, VR={percent_visible_states} =======================')

                output = run_NON_DETERMINSTIC_single_experiment_PO( domain_name=domain_name,
                                                                 ml_model_name=ml_model_name,
                                                                 render_mode=render_mode,
                                                                 max_exec_len=max_exec_len,
                                                                 debug_print=debug_print,
                                                                 execution_fault_mode_name=execution_fault_mode_name,
                                                                 instance_seed=instance_seed,
                                                                 fault_probability=fault_rate,
                                                                 percent_visible_states=percent_visible_states,
                                                                 possible_fault_mode_names=possible_fault_mode_names,
                                                                 num_candidate_fault_modes=num_candidate_fault_modes,
                                                                 epsilon = epsilon,
                                                                 unknown_fault_rate=unknown_fault_rate,
                                                                 fault_rate_candidates=fault_rate_candidates)
                if not output:
                    continue

                output["epsilon"] = epsilon
                output["experiment_num"] = i + 1
                output["real_fault_prob"] = fault_rate
                output["map_desc"] = map_desc
                output["hardcoded_policy"] = hardcoded_policy
                output["domain_name"] = domain_name

                records.append(output)


                print(f'=========================== END SINGLE EXPERIMENT MAP {i + 1}/{NUM_MAPS} with FR={fault_rate}, VR{percent_visible_states} ===========================')

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'======================================= {dt_string}: MAP {i+1}/{NUM_MAPS} END =======================================')


    from datetime import timedelta

    total_diagnosis_time_sec = sum(
        r["diagnosis_time_sec"]
        for r in records
    )

    avg_diagnosis_time_sec = (
            total_diagnosis_time_sec / len(records)
    )

    print(f"\nNumber of diagnosis runs: {len(records)}")

    # timedelta prints 5:42:17 -> 5h, 42 min, 17 sec
    print(
        f"Total diagnosis time: "
        f"{timedelta(seconds=int(total_diagnosis_time_sec))} "
        f"({total_diagnosis_time_sec:.2f} sec)"
    )

    print(
        f"Average diagnosis time: "
        f"{timedelta(seconds=int(avg_diagnosis_time_sec))} "
        f"({avg_diagnosis_time_sec:.2f} sec)"
    )

    file_suffix = str(epsilon).replace(".", "_")

    if unknown_fault_rate:
        method_suffix = "UN_known_fr"
    else:
        method_suffix = "known_fr"

    file_path = f"frozen_lake_non_deterministic_PO_{method_suffix}_epsilon_{file_suffix}_MAPS_{maps_num}"

    output_dir = domain_results_dir("FrozenLake_v1", run_folder)
    exper_write_records_to_excel_ind(
        records,
        file_path,
        output_dir=output_dir
    )
    print(f"file was written at: {output_dir}/{file_path}.xlsx")

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))

"""
def single_experiment_Taxi_SIFU():
    # changable test settings - strong fault model intermittent faults (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')

"""



def single_experiment_stochastic_FrozenLake(run_folder=None):


    global HARD_CODED_POLICY
    loaded = load_pairs_from_json("frozenlake_100_pairs_risk_averse_slippery.json")

    epsilon = 0.03
    unknown_fault_rate = False
    fault_rate_candidates = None
    records = []

    print("Running single_experiment_stochastic_FrozenLake"+ "\n\n")
    i = 1


    map_desc, hardcoded_policy = loaded[i]
    DOMAIN_KWARGS["FrozenLake_v1"]["desc"] = map_desc
    DOMAIN_KWARGS["FrozenLake_v1"]["is_slippery"] = True
    h_rl_models.HARD_CODED_POLICY = hardcoded_policy
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'==================== {dt_string}: START ========================')

    domain_name = "FrozenLake_v1"
    ml_model_name = "PPO"                         # "PPO", "DQN"
    render_mode = "rgb_array"                     # "human", "rgb_array"
    max_exec_len = 200
    debug_print = True

    instance_seed = 42
    num_candidate_fault_modes = 10
    possible_fault_mode_names = [
        "[0,0,2,3]",
        "[0,3,2,3]",
        "[0,1,0,3]",
        "[0,1,3,3]",
        "[0,2,2,3]",

        "[0,3,0,3]",
        "[0,3,3,3]",
        "[0,0,0,3]",
        "[0,0,3,3]",
        "[0,2,1,3]",

        "[0,0,0,0]",
        "[1,1,1,1]",
        "[2,2,2,2]",
        "[3,3,3,3]"
        ]

    fault_rate = 0.5
    percent_visible_states = 100

    execution_fault_mode_name = "[0,0,2,3]"
    output = run_NON_DETERMINSTIC_single_experiment_PO(
        domain_name=domain_name,
        ml_model_name=ml_model_name,
        render_mode=render_mode,
        max_exec_len=max_exec_len,
        debug_print=debug_print,
        execution_fault_mode_name=execution_fault_mode_name,
        instance_seed=instance_seed,
        fault_probability=fault_rate,
        percent_visible_states=percent_visible_states,
        possible_fault_mode_names=possible_fault_mode_names,
        num_candidate_fault_modes=num_candidate_fault_modes,
        epsilon=epsilon,
        unknown_fault_rate=unknown_fault_rate,
        fault_rate_candidates=fault_rate_candidates)
    if not output:
        print("output is empty")
        exit(9)

    output["epsilon"] = epsilon
    output["real_fault_prob"] = fault_rate
    output["domain_name"] = domain_name
    records.append(output)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'=========================== {dt_string}: END ===========================')

    total_diagnosis_time_sec = output["diagnosis_time_sec"]
    # timedelta prints 5:42:17 -> 5h, 42 min, 17 sec
    print(f"Total diagnosis time: "
          f"{timedelta(seconds=int(total_diagnosis_time_sec))} "
          f"({total_diagnosis_time_sec:.2f} sec)"
          )

    file_suffix = str(epsilon).replace(".", "_")
    method_suffix = "known_fr"
    file_path = f"frozen_lake_single_exper_{method_suffix}_epsilon_{file_suffix}"

    output_dir = domain_results_dir(domain_name, run_folder)
    exper_write_records_to_excel_ind(
        records,
        file_path,
        output_dir=output_dir
    )
    print(f"file was written at: {output_dir}/{file_path}.xlsx")

def single_experiment_stochastic_Taxi_v4(run_folder=None):

    epsilon = 0.03
    unknown_fault_rate = False
    fault_rate_candidates = None
    records = []

    print("Running single_experiment_stochastic_Taxi_v4\n\n")
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'====================== {dt_string}: START ======================')

    domain_name = "Taxi_v4"
    ml_model_name = "PPO"   # "PPO", "DQN"
    render_mode = "rgb_array"   # "human", "rgb_array"
    max_exec_len = 200
    debug_print = True
    instance_seed = 42
    # Taxi action ids (fixed by Gymnasium). On the grid:
    #   0 = South  -> move DOWN
    #   1 = North  -> move UP
    #   2 = East   -> move RIGHT
    #   3 = West   -> move LEFT
    #   4 = Pickup passenger
    #   5 = Dropoff passenger
    #
    # A fault mode is a 6-length action map: position = action the policy COMMANDS,
    # value = action that actually EXECUTES. The healthy (no-fault) map is [0,1,2,3,4,5].
    # Values may repeat (an action can collapse onto another), so it need not be a permutation.
    # Example: "[0,0,2,3,4,5]" -> position 1 holds 0, so a commanded 1 (UP) executes as 0 (DOWN).
    execution_fault_mode_name = "[0,0,2,3,4,5]"
    fault_rate = 0.5
    percent_visible_states = 100
    possible_fault_mode_names = [
        "[0,0,2,3,4,5]",   # UP    -> DOWN
        "[0,1,0,3,4,5]",   # RIGHT -> DOWN
        "[0,1,2,0,4,5]",   # LEFT  -> DOWN
        "[0,1,2,3,0,5]",   # Pickup  -> DOWN
        "[0,1,2,3,4,0]",   # Dropoff -> DOWN
        "[0,2,1,3,4,5]",   # swap UP<->RIGHT
        "[0,3,2,1,4,5]",   # swap UP<->LEFT
        "[0,4,2,3,1,5]",   # swap UP<->Pickup
        "[0,5,2,3,4,1]",   # swap UP<->Dropoff
        "[1,0,2,3,4,5]"    # swap DOWN<->UP
    ]
    num_candidate_fault_modes = 10


    output = run_NON_DETERMINSTIC_single_experiment_PO(
            domain_name=domain_name,
            ml_model_name=ml_model_name,
            render_mode=render_mode,
            max_exec_len=max_exec_len,
            debug_print=debug_print,
            execution_fault_mode_name=execution_fault_mode_name,
            instance_seed=instance_seed,
            fault_probability=fault_rate,
            percent_visible_states=percent_visible_states,
            possible_fault_mode_names=possible_fault_mode_names,
            num_candidate_fault_modes=num_candidate_fault_modes,
            epsilon = epsilon,
            unknown_fault_rate=unknown_fault_rate,
            fault_rate_candidates=fault_rate_candidates)
    if not output:
        print("output is empty")
        exit(9)

    output["epsilon"] = epsilon
    output["real_fault_prob"] = fault_rate
    output["domain_name"] = domain_name
    records.append(output)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'=========================== {dt_string}: END ===========================')


    total_diagnosis_time_sec = output["diagnosis_time_sec"]
    # timedelta prints 5:42:17 -> 5h, 42 min, 17 sec
    print(f"Total diagnosis time: "
    f"{timedelta(seconds=int(total_diagnosis_time_sec))} "
    f"({total_diagnosis_time_sec:.2f} sec)"
    )

    file_suffix = str(epsilon).replace(".", "_")
    method_suffix = "known_fr"
    file_path = f"taxi_v4_single_exper_non_deterministic_PO_{method_suffix}_epsilon_{file_suffix}"

    output_dir = domain_results_dir(domain_name, run_folder)
    exper_write_records_to_excel_ind(
        records,
        file_path,
        output_dir=output_dir
    )
    print(f"file was written at: {output_dir}/{file_path}.xlsx")


def multiple_experiment_Taxi_v4_NON_DETERMINSTIC_PO(epsilon=0.03, unknown_fault_rate=False, num_seeds=5, run_folder=None):
    """Taxi-v4 sweep, mirroring multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO.

    Taxi has ONE fixed map and ONE policy, so the variation source is the SEED:
    each seed yields a different start (taxi cell, passenger, destination) -> a
    different trajectory. For each seed we sweep fault_rate x visibility, with one
    random execution fault per seed. Trajectory creation can fail under a fault
    (taxi never completes) -> we skip and count those.
    """
    records = []
    skipped = 0

    if unknown_fault_rate:
        fault_rate_candidates = [0.1, 0.3, 0.5, 0.7, 0.8]
    else:
        fault_rate_candidates = None

    domain_name = "Taxi_v4"
    ml_model_name = "PPO"
    render_mode = "rgb_array"
    max_exec_len = 200
    debug_print = False
    num_candidate_fault_modes = 10

    # Each seed's execution fault and 10-candidate set come from the frozen hard benchmark
    # (hard_taxi_data). The first num_seeds good seeds are used.
    seeds = benchmark_seeds()[:num_seeds]

    fault_rate_list = [0.5, 0.8]
    percent_visible_states_list = [20, 40, 60, 80, 100]

    msg = (
        f"Running Taxi-v4 PO diagnosis (HARD benchmark) | epsilon={epsilon} | "
        f"unknown_fault_rate={unknown_fault_rate} | num_seeds={len(seeds)}"
    )
    if fault_rate_candidates is not None:
        msg += f" | fault_rate_candidates={fault_rate_candidates}"
    print(msg + "\n\n")

    for i, instance_seed in enumerate(seeds):
        execution_fault_mode_name, candidate_fault_modes = get_instance(instance_seed)

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'================= {dt_string}: SEED {instance_seed} ({i+1}/{len(seeds)}), '
              f'fault={execution_fault_mode_name} START =================')

        for percent_visible_states in percent_visible_states_list:
            for fault_rate in fault_rate_list:
                print(f'===== SEED {instance_seed} | FR={fault_rate} | VR={percent_visible_states} =====')

                output = run_NON_DETERMINSTIC_single_experiment_PO(
                    domain_name=domain_name,
                    ml_model_name=ml_model_name,
                    render_mode=render_mode,
                    max_exec_len=max_exec_len,
                    debug_print=debug_print,
                    execution_fault_mode_name=execution_fault_mode_name,
                    instance_seed=instance_seed,
                    fault_probability=fault_rate,
                    percent_visible_states=percent_visible_states,
                    possible_fault_mode_names=candidate_fault_modes,
                    num_candidate_fault_modes=num_candidate_fault_modes,
                    epsilon=epsilon,
                    unknown_fault_rate=unknown_fault_rate,
                    fault_rate_candidates=fault_rate_candidates,
                    fixed_candidate_fault_modes=candidate_fault_modes)

                if not output:
                    skipped += 1
                    continue

                output["epsilon"] = epsilon
                output["experiment_num"] = i + 1
                output["real_fault_prob"] = fault_rate
                output["map_desc"] = f"seed_{instance_seed}"
                output["hardcoded_policy"] = f"{domain_name}_{ml_model_name}"
                output["domain_name"] = domain_name

                records.append(output)

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'================= {dt_string}: SEED {instance_seed} END =================')

    if not records:
        print("No successful Taxi experiments produced (all trajectories failed).")
        return

    total_diagnosis_time_sec = sum(r["diagnosis_time_sec"] for r in records)
    avg_diagnosis_time_sec = total_diagnosis_time_sec / len(records)
    print(f"\nNumber of diagnosis runs: {len(records)} (skipped {skipped})")
    print(f"Total diagnosis time: {timedelta(seconds=int(total_diagnosis_time_sec))} "
          f"({total_diagnosis_time_sec:.2f} sec)")
    print(f"Average diagnosis time: {timedelta(seconds=int(avg_diagnosis_time_sec))} "
          f"({avg_diagnosis_time_sec:.2f} sec)")

    file_suffix = str(epsilon).replace(".", "_")
    method_suffix = "UN_known_fr" if unknown_fault_rate else "known_fr"
    file_path = f"taxi_v4_non_deterministic_PO_{method_suffix}_epsilon_{file_suffix}_SEEDS_{num_seeds}"

    output_dir = domain_results_dir(domain_name, run_folder)
    exper_write_records_to_excel_ind(records, file_path, output_dir=output_dir)
    print(f"file was written at: {output_dir}/{file_path}.xlsx")


def multiple_experiments_FrozenLake_SIF():

    global HARD_CODED_POLICY
    diagnosis_runtimes_ms = []
    NUM_TRIES = 100
    loaded = load_pairs_from_json("frozenlake_100_pairs.json")

    skip_list = [12, 25, 36, 42, 59, 73, 83, 86, 91, 100]
    records = []
    for i in range(NUM_TRIES):
        print(f"experiment {i+1} starts:")
        if (i+1) in skip_list:
            print("here skippp")
            continue
        map_desc, hardcoded_policy = loaded[i]

        DOMAIN_KWARGS["FrozenLake_v1"]["desc"] = map_desc
        h_rl_models.HARD_CODED_POLICY = hardcoded_policy

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i+1}/{NUM_TRIES}')
        domain_name = "FrozenLake_v1"
        ml_model_name = "PPO"
        render_mode = "rgb_array"
        max_exec_len = 200
        debug_print = False
        instance_seed = 10 + i
        num_candidate_fault_modes = 10
        rng = random.Random(instance_seed)

        fault_probability = 0.3
        possible_fault_mode_names = [
                "[0,0,2,3]",
                "[0,3,2,3]",
                "[0,1,0,3]",
                "[0,1,3,3]",
                "[0,2,2,3]",

                "[0,3,0,3]",
                "[0,3,3,3]",
                "[0,0,0,3]",
                "[0,0,3,3]",
                "[0,2,1,3]",

                "[0,0,0,0]",
                "[1,1,1,1]",
                "[2,2,2,2]",
                "[3,3,3,3]"
              ]
        execution_fault_mode_name = rng.choice(possible_fault_mode_names)
        percent_visible_states = 30
        time_always_increasing = True
        time_increases_between_start_and_end = True
        visible_states_percent_list = [100, 90, 75, 50, 35]
        visible_states_percent_start = visible_states_percent_list[0]
        visible_states_percent_end = visible_states_percent_list[-1]

        for percent_visible_states in visible_states_percent_list:
            record = run_SIF_single_experiment(domain_name=domain_name,
                                                             ml_model_name=ml_model_name,
                                                             render_mode=render_mode,
                                                             max_exec_len=max_exec_len,
                                                             debug_print=debug_print,
                                                             execution_fault_mode_name=execution_fault_mode_name,
                                                             instance_seed=instance_seed,
                                                             fault_probability=fault_probability,
                                                             percent_visible_states=percent_visible_states,
                                                             possible_fault_mode_names=possible_fault_mode_names,
                                                             num_candidate_fault_modes=num_candidate_fault_modes,
                                                             multi_experiment=True)

            record["experiment_num"] = i+1
            record["map_desc"] = map_desc
            record["hardcoded_policy"] = hardcoded_policy
            record["contains_real_diagnosis"] = str(execution_fault_mode_name in record["output"]["diagnoses"])
            record["diagnosis"] = str(record["output"]["diagnoses"])
            record["diagnosis_run_time_ms"] = record["output"]["diag_rt_ms"]
            record["diagnosis_run_time_sec"] = record["output"]["diag_rt_sec"]
            record["fault_prob"] = fault_probability
            record["domain_name"] = domain_name

            if not percent_visible_states == visible_states_percent_start:
                if records[-1]["diagnosis_run_time_ms"] > record["diagnosis_run_time_ms"]:
                    time_always_increasing = False

            if percent_visible_states == visible_states_percent_end:
                compare_index = len(visible_states_percent_list)-1
                if records[-compare_index]["diagnosis_run_time_ms"] > record["diagnosis_run_time_ms"]:
                    time_increases_between_start_and_end = False

            record["time_always_increasing"] = time_always_increasing
            record["time_increases_between_start_and_end"] = time_increases_between_start_and_end

            records.append(record)

    exper_100_write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIF")
    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    # print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')



def single_experiment_FrozenLake_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)

    """
    frozen lake:
    0 = Left
    1 = Down
    2 = Right
    3 = Up
    [2,1,2,3] means
    [0,1,2,3] -> [2,1,2,3]
    fault mode is:
    f(Left) = Right
    all other actions behave good
    """

    global HARD_CODED_POLICY
    loaded = load_pairs_from_json("frozenlake_100_pairs.json")
    NUM_OF_MAPS_AND_POLICIES = 12
    choosen_num = 1
    map_desc, hardcoded_policy = loaded[choosen_num]
    DOMAIN_KWARGS["FrozenLake_v1"]["desc"] = map_desc
    h_rl_models.HARD_CODED_POLICY = hardcoded_policy

    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')

        domain_name = "FrozenLake_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        # execution_fault_mode_name = "[0,1,0,3]"
        # execution_fault_mode_name = "[0,3,0,3]"
        # execution_fault_mode_name = "[0,3,0,3]"
        execution_fault_mode_name = "[0,3,2,3]"
        instance_seed = 10
        fault_probability = 0.3 # 1.0 mean Non-intermittent fault - faults always occur
        percent_visible_states = 90
        possible_fault_mode_names = [
                "[0,0,2,3]",
                "[2,1,2,3]",
                "[0,3,2,1]",
                "[0,0,0,3]",
                "[0,0,2,0]",
                "[0,1,0,0]",
                "[0,0,0,0]",
                "[0,2,2,1]",
                "[2,1,0,3]",
                "[1,0,2,3]",
                "[0,3,2,3]"
              ]

        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')




def single_experiment_FrozenLake_NON_DETERMINSTIC():
    # changable test settings - strong fault model intermittent faults (SIF)

    """
    frozen lake:
    0 = Left
    1 = Down
    2 = Right
    3 = Up
    [2,1,2,3] means
    [0,1,2,3] -> [2,1,2,3]
    fault mode is:
    f(Left) = Right
    all other actions behave good
    """

    global HARD_CODED_POLICY
    loaded = load_pairs_from_json("frozenlake_100_pairs_risk_averse_slippery.json")
    NUM_OF_MAPS_AND_POLICIES = 12
    choosen_num = 1
    map_desc, hardcoded_policy = loaded[choosen_num]

    print_map_and_policy(map_desc, hardcoded_policy)
    print(f"map desc: {map_desc}")
    print(f"hardcoded_policy: {hardcoded_policy}")
    DOMAIN_KWARGS["FrozenLake_v1"]["desc"] = map_desc
    DOMAIN_KWARGS["FrozenLake_v1"]["is_slippery"] = True

    h_rl_models.HARD_CODED_POLICY = hardcoded_policy

    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')

        domain_name = "FrozenLake_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        # execution_fault_mode_name = "[0,1,0,3]"
        # execution_fault_mode_name = "[0,3,0,3]"
        # execution_fault_mode_name = "[0,3,0,3]"
        execution_fault_mode_name = "[0,3,2,3]"

        instance_seed = 10
        fault_probability = 0.6 # 1.0 mean Non-intermittent fault - faults always occur
        percent_visible_states = 100
        possible_fault_mode_names = [
                "[0,0,2,3]",
                "[2,1,2,3]",
                "[0,3,2,1]",
                "[0,0,0,3]",
                "[0,0,2,0]",
                "[0,1,0,0]",
                "[0,0,0,0]",
                "[0,2,2,1]",
                "[2,1,0,3]",
                "[1,0,2,3]",
                "[0,3,2,3]"
              ]

        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_NON_DETERMINSTIC_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')

def single_experiment_Acrobot_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,2,1]"
        instance_seed = 42
        fault_probability = 0.5
        percent_visible_states = 30
        possible_fault_mode_names = [
                            "[1,1,2]",
                            "[0,1,1]",
                            "[0,2,1]",
                            "[1,0,2]",
                            "[1,2,0]",
                            "[2,0,1]",
                            "[2,1,0]",
                            "[0,0,0]",
                            "[1,1,1]",
                            "[0,2,2]"
                          ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU():
    # changable test settings - strong fault model intermittent faults smart (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU5():
    # changable test settings - strong fault model intermittent faults upgraded 5 (SIFU5)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU5_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU6():
    # changable test settings - strong fault model intermittent faults upgraded 6 (SIFU6)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU6_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU7():
    # changable test settings - strong fault model intermittent faults upgraded 7 (SIFU7)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU7_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU8():
    # changable test settings - strong fault model intermittent faults upgraded 8 (SIFU8)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU8_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# =========================================== CartPole ============================================
# =================================================================================================
def single_experiment_CartPole_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU():
    # changable test settings - strong fault model intermittent faults smart (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU5():
    # changable test settings - strong fault model intermittent faults upgraded 5 (SIFU5)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU5_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU6():
    # changable test settings - strong fault model intermittent faults upgraded 6 (SIFU6)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU6_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU7():
    # changable test settings - strong fault model intermittent faults upgraded 7 (SIFU7)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU7_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU8():
    # changable test settings - strong fault model intermittent faults upgraded 8 (SIFU8)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU8_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# ========================================== MountainCar ==========================================
# =================================================================================================
def single_experiment_MountainCar_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU():
    # changable test settings - strong fault model intermittent faults (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU5():
    # changable test settings - strong fault model intermittent faults upgraded 5 (SIFU5)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU5_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU6():
    # changable test settings - strong fault model intermittent faults upgraded 6 (SIFU6)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU6_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU7():
    # changable test settings - strong fault model intermittent faults upgraded 7 (SIFU7)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU7_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU8():
    # changable test settings - strong fault model intermittent faults upgraded 8 (SIFU8)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
            "[0,1,1]",
            "[0,2,1]",
            "[1,0,2]",
            "[1,2,0]",
            "[2,0,1]",
            "[2,1,0]",
            "[0,0,0]",
            "[1,1,1]",
            "[2,2,2]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU8_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# ============================================= Taxi ==============================================
# =================================================================================================
def single_experiment_Taxi_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU():
    # changable test settings - strong fault model intermittent faults (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU5():
    # changable test settings - strong fault model intermittent faults upgraded 5 (SIFU5)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU5_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU6():
    # changable test settings - strong fault model intermittent faults upgraded 6 (SIFU6)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU6_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU7():
    # changable test settings - strong fault model intermittent faults upgraded 7 (SIFU7)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU7_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU8():
    # changable test settings - strong fault model intermittent faults upgraded 8 (SIFU8)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3,4,5]",
            "[0,1,0,3,4,5]",
            "[0,1,2,0,4,5]",
            "[0,1,2,3,0,5]",
            "[0,1,2,3,4,0]",
            "[0,2,1,3,4,5]",
            "[0,3,2,1,4,5]",
            "[0,4,2,3,1,5]",
            "[0,5,2,3,4,1]",
            "[1,0,2,3,4,5]"
        ]
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU8_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')

