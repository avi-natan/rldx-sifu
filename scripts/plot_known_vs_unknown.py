"""Taxi-v4 known- vs unknown-fault-rate comparison, epsilon-driven.

For each shared epsilon, compare the two runs on:
  - avg real-fault rank   -> ..._rank_known_vs_unknown.png
  - avg diagnosis time    -> ..._time_known_vs_unknown.png

X = epsilon; two lines (known, unknown), each pooled over all visibility
levels / seeds at that epsilon (SEM error bars). Only epsilons present in
BOTH runs are plotted, so the comparison is apples-to-apples.

Run from repo root:
  ./.venv_domains/Scripts/python.exe scripts/plot_known_vs_unknown.py
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANK_COL = "real_fault_rank"
TIME_COL = "diagnosis_time_sec"
EPS_COL = "epsilon"

KNOWN_COLOR = "#1f77b4"
UNKNOWN_COLOR = "#d62728"


def load_files(paths, label):
    if not paths:
        raise SystemExit(f"No {label} xlsx found.")
    print(f"  {label}: {len(paths)} files")
    return pd.concat([pd.read_excel(p) for p in sorted(paths)], ignore_index=True)


def find_taxi_xlsx(tx_root):
    """Rename-proof: locate known vs unknown Taxi xlsx by filename, anywhere under tx_root."""
    all_xlsx = glob.glob(os.path.join(tx_root, "**", "*.xlsx"), recursive=True)
    all_xlsx = [p for p in all_xlsx if os.sep + "old" + os.sep not in p]  # skip any old/ archive
    unknown = [p for p in all_xlsx if "unknown_fr" in os.path.basename(p)]
    known = [p for p in all_xlsx if "known_fr" in os.path.basename(p)
             and "unknown_fr" not in os.path.basename(p)]
    return known, unknown


def agg(df, value_col, epsilons):
    xs, ys, sems = [], [], []
    for e in epsilons:
        s = df[df[EPS_COL] == e][value_col]
        xs.append(e)
        ys.append(s.mean())
        sems.append(s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
    return xs, ys, sems


def compare_plot(known, unknown, epsilons, value_col, ylabel, title, out_path,
                 annotate_ratio=False):
    kx, ky, ks = agg(known, value_col, epsilons)
    ux, uy, us = agg(unknown, value_col, epsilons)
    plt.figure(figsize=(7.5, 4.8))
    plt.errorbar(kx, ky, yerr=ks, fmt="o-", capsize=4, markersize=7, linewidth=2,
                 color=KNOWN_COLOR, label="known fault rate")
    plt.errorbar(ux, uy, yerr=us, fmt="s--", capsize=4, markersize=7, linewidth=2,
                 color=UNKNOWN_COLOR, label="unknown fault rate")
    if annotate_ratio:
        for e, ku, uu in zip(epsilons, ky, uy):
            if ku:
                plt.annotate(f"{uu / ku:.1f}x", (e, uu), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9, color=UNKNOWN_COLOR)
    plt.xticks(epsilons)
    plt.xlabel("Epsilon (MC CI half-width)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tx = os.path.join(repo_root, "experimental results", "Taxi_v4")
    known_paths, unknown_paths = find_taxi_xlsx(tx)
    known = load_files(known_paths, "known")
    unknown = load_files(unknown_paths, "unknown")

    shared = sorted(set(known[EPS_COL].unique()) & set(unknown[EPS_COL].unique()))
    if not shared:
        raise SystemExit("No epsilon overlap between known and unknown runs.")
    out_dir = os.path.join(tx, "known_vs_unknown_comparison")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Shared epsilons: {shared}  |  -> {out_dir}\n")

    compare_plot(known, unknown, shared, RANK_COL, "Avg real-fault rank",
                 "Taxi-v4 known vs unknown fr: rank by epsilon",
                 os.path.join(out_dir, "taxi_v4_rank_known_vs_unknown.png"))
    compare_plot(known, unknown, shared, TIME_COL, "Avg diagnosis time (sec)",
                 "Taxi-v4 known vs unknown fr: time by epsilon",
                 os.path.join(out_dir, "taxi_v4_time_known_vs_unknown.png"),
                 annotate_ratio=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
