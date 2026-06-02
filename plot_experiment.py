import os
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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


def load_records_from_excel(xls_path, sheet_name=0):
    df = pd.read_excel(xls_path)
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")

def save_current_plot(output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def create_and_save_all_plots(records, output_dir):
    plots = [
        ("sorted_lengths_running_avg.png", lambda: plot_sorted_lengths_running_avg(records)),
        ("binned_avg_rank.png", lambda: plot_binned_avg_rank(records)),
        ("rank_vs_fault_occurrences.png", lambda: plot_rank_vs_fault_occurrences_scatter(records)),
        ("avg_rank_vs_fault_rate.png", lambda: plot_avg_rank_vs_fault_rate(records)),
        ("rank_vs_visibility_rate.png", lambda: plot_rank_vs_visibility_rate(records)),
        ("avg_rank_vs_visibility_rate.png", lambda: plot_avg_rank_vs_visibility_rate(records)),
        ("avg_time_vs_visibility_rate.png", lambda: plot_avg_time_vs_visibility_rate(records)),
        ("rank_vs_num_observed_states.png", lambda: plot_rank_vs_num_observed_states(records)),
        ("rank_vs_largest_gap.png", lambda: plot_rank_vs_largest_gap(records)),
        ("avg_gap_time_vs_visibility_rate.png", lambda: plot_avg_gap_time_vs_visibility_rate(records)),
        ("num_gaps_vs_visibility_rate.png", lambda: plot_num_gaps_vs_visibility_rate(records)),
    ]

    for filename, plot_func in plots:
        try:
            plot_func()
            save_current_plot(output_dir, filename)
        except Exception as e:
            plt.close()
            print(f"Skipped {filename}: {e}")


def main():
    epsilon_number = "0_03"
    xls_path = f"C:/Users/ahmad/Downloads/xl_results/frozen_lake_non_deterministic_PO_epsilon_{epsilon_number}.xlsx"
    output_dir = f"single_experiment_plots/epsilon_{epsilon_number}_experiment"

    records = load_records_from_excel(xls_path)
    print(f"Number of records read: {len(records)}")

    os.makedirs(output_dir, exist_ok=True)
    create_and_save_all_plots(records, output_dir)


if __name__ == "__main__":
    main()