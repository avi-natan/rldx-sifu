import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


try_num = 2
INPUT_DIR = f"C:/Users/ahmad/Downloads/experiments_outputs/several_epsilons_run/try{try_num}/xl_results"
PLOTS_DIR = "aggregated_plots"

def save_current_plot(filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {path}")


def load_all_records(input_dir):
    records = []

    xlsx_files = glob.glob(os.path.join(input_dir, "*.xlsx"))

    for path in xlsx_files:

        # extract epsilon from filename
        match = re.search(r"epsilon_(\d+_\d+)", os.path.basename(path))
        if match:
            epsilon = float(match.group(1).replace("_", "."))
        else:
            epsilon = None

        df = pd.read_excel(path)

        curr_records = df.to_dict(orient="records")

        for r in curr_records:
            r["epsilon"] = epsilon

        records.extend(curr_records)

    return records

def plot_avg_rank_vs_epsilon(records):

    eps_to_ranks = {}

    for r in records:
        eps = r.get("epsilon")
        rank = r.get("real_fault_rank")

        if eps is None or rank is None:
            continue

        eps_to_ranks.setdefault(float(eps), []).append(float(rank))

    xs = sorted(eps_to_ranks.keys())
    ys = [np.mean(eps_to_ranks[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs, rotation=90)
    plt.xlabel("Epsilon")
    plt.ylabel("Average real fault rank")
    plt.title("Epsilon vs average real fault rank")
    plt.grid(True)

    save_current_plot("epsilon_vs_avg_real_fault_rank.png")



def plot_runtime_vs_epsilon(records):

    eps_to_times = {}

    for r in records:
        eps = r.get("epsilon")
        t = r.get("diagnosis_time_sec")

        if eps is None or t is None:
            continue

        eps_to_times.setdefault(float(eps), []).append(float(t))

    xs = sorted(eps_to_times.keys())

    avg_times = [np.mean(eps_to_times[x]) for x in xs]
    total_times = [np.sum(eps_to_times[x]) for x in xs]

    plt.figure()
    plt.plot(xs, avg_times, "o-", label="avg runtime")
    # plt.plot(xs, total_times, "s--", label="total runtime")

    plt.xlabel("Epsilon")
    plt.ylabel("Runtime (sec)")
    plt.title("Epsilon vs runtime")
    plt.legend()
    plt.grid(True)

    save_current_plot("epsilon_vs_avg_diagnosis_time.png")


def plot_avg_rank_vs_visibility(records):

    vis_to_ranks = {}

    for r in records:
        vis = r.get("percent_visible_states")
        rank = r.get("real_fault_rank")

        if vis is None or rank is None:
            continue

        vis_to_ranks.setdefault(float(vis), []).append(float(rank))

    xs = sorted(vis_to_ranks.keys())
    ys = [np.mean(vis_to_ranks[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average real fault rank")
    plt.title("Visibility vs average real fault rank")
    plt.grid(True)
    save_current_plot("visibility_vs_avg_real_fault_rank.png")



def plot_runtime_vs_visibility(records):

    vis_to_times = {}

    for r in records:
        vis = r.get("percent_visible_states")
        t = r.get("diagnosis_time_sec")

        if vis is None or t is None:
            continue

        vis_to_times.setdefault(float(vis), []).append(float(t))

    xs = sorted(vis_to_times.keys())

    avg_times = [np.mean(vis_to_times[x]) for x in xs]
    total_times = [np.sum(vis_to_times[x]) for x in xs]

    plt.figure()
    plt.plot(xs, avg_times, "o-", label="avg runtime")
    # plt.plot(xs, total_times, "s--", label="total runtime")

    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Runtime (sec)")
    plt.title("Visibility vs runtime")
    plt.legend()
    plt.grid(True)
    save_current_plot("visibility_vs_avg_diagnosis_time.png")

def plot_avg_real_tries_vs_epsilon(records):
    eps_to_vals = {}

    for r in records:
        eps = r.get("epsilon")
        v = r.get("adaptive_avg_real_tries")

        if eps is None or v is None or pd.isna(v):
            continue

        eps_to_vals.setdefault(float(eps), []).append(float(v))

    xs = sorted(eps_to_vals.keys())
    ys = [np.mean(eps_to_vals[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs, rotation=90)
    plt.xlabel("Epsilon")
    plt.ylabel("Average real tries per MC call")
    plt.title("Epsilon vs average real tries")
    plt.grid(True)

    save_current_plot("epsilon_vs_avg_real_tries.png")

def plot_total_real_tries_vs_epsilon(records):
    eps_to_vals = {}

    for r in records:
        eps = r.get("epsilon")
        v = r.get("adaptive_total_real_tries")

        if eps is None or v is None or pd.isna(v):
            continue

        eps_to_vals.setdefault(float(eps), []).append(float(v))

    xs = sorted(eps_to_vals.keys())
    ys = [np.mean(eps_to_vals[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs, rotation=90)
    plt.xlabel("Epsilon")
    plt.ylabel("Average total real tries per diagnosis")
    plt.title("Epsilon vs total real tries")
    plt.grid(True)

    save_current_plot("epsilon_vs_total_real_tries.png")

def plot_max_stop_rate_vs_epsilon(records):
    eps_to_vals = {}

    for r in records:
        eps = r.get("epsilon")
        v = r.get("adaptive_max_stop_rate")

        if eps is None or v is None or pd.isna(v):
            continue

        eps_to_vals.setdefault(float(eps), []).append(float(v))

    xs = sorted(eps_to_vals.keys())
    ys = [np.mean(eps_to_vals[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-")
    plt.xticks(xs, rotation=90)
    plt.xlabel("Epsilon")
    plt.ylabel("Average max stop rate")
    plt.title("Epsilon vs max stop rate")
    plt.grid(True)

    save_current_plot("epsilon_vs_max_stop_rate.png")

def plot_avg_margin_vs_epsilon(records):
    eps_to_vals = {}

    for r in records:
        eps = r.get("epsilon")
        v = r.get("adaptive_avg_margin")

        if eps is None or v is None or pd.isna(v):
            continue

        eps_to_vals.setdefault(float(eps), []).append(float(v))

    xs = sorted(eps_to_vals.keys())
    ys = [np.mean(eps_to_vals[x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, "o-", label="avg margin")
    plt.plot(xs, xs, "--", label="epsilon target")
    plt.xticks(xs, rotation=90)
    plt.xlabel("Epsilon")
    plt.ylabel("Margin")
    plt.title("Epsilon vs average achieved margin")
    plt.legend()
    plt.grid(True)

    save_current_plot("epsilon_vs_avg_margin.png")


if __name__ == "__main__":

    records = load_all_records(INPUT_DIR)

    print(f"Loaded {len(records)} records")

    plot_avg_rank_vs_epsilon(records)
    plot_runtime_vs_epsilon(records)

    plot_avg_rank_vs_visibility(records)
    plot_runtime_vs_visibility(records)

    plot_avg_real_tries_vs_epsilon(records)
    plot_total_real_tries_vs_epsilon(records)
    plot_max_stop_rate_vs_epsilon(records)
    plot_avg_margin_vs_epsilon(records)

    plt.show()