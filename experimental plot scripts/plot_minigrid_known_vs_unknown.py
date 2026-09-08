"""MiniGrid Empty known- vs unknown-fault-rate comparison at noise 0.5, along fault rate.

Both runs are at noise 0.5 and epsilon 0.04, over the same 26 faults x 3 seeds x 5 visibilities
x 3 injected fault rates. X = injected fault rate (0.3 / 0.5 / 0.8); two lines (known, unknown),
each POOLED over all visibility levels (and seeds/faults) at that fault rate (SEM error bars) --
i.e. every instance at a fault rate contributes, exactly like the FrozenLake known-vs-unknown plot.

Produces, into MiniGrid_Empty_16x16_v0/known_vs_unknown_comparison/:
  - avg real-fault rank  -> minigrid_noise0_5_rank_known_vs_unknown.png
  - avg diagnosis time   -> minigrid_noise0_5_time_known_vs_unknown.png   (unknown/known ratio annotated)

Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_minigrid_known_vs_unknown.py"
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_provenance import write_plot_provenance

RANK_COL = "real_fault_rank"
TIME_COL = "diagnosis_time_sec"
FR_COL = "real_fault_prob"

KNOWN_COLOR = "#1f77b4"
UNKNOWN_COLOR = "#d62728"

DOMAIN = "MiniGrid_Empty_16x16_v0"
KNOWN_RUN = os.path.join("known", "minigrid_bench_noise0_5")
UNKNOWN_RUN = os.path.join("unknown", "minigrid_bench_noise0_5_ufr")


def _load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "xlsx", "*.xlsx")))
    if not files:
        raise SystemExit(f"no result xlsx found in {run_dir}")
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    df = df[df[RANK_COL].notna()].copy()
    df[FR_COL] = df[FR_COL].round(2)
    return df


def _agg_by_fr(df, value_col, frs):
    ys, sems = [], []
    for fr in frs:
        s = df[df[FR_COL] == fr][value_col]
        ys.append(s.mean())
        sems.append(s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
    return ys, sems


def compare_plot(known, unknown, frs, value_col, ylabel, title, out_path, annotate_ratio=False):
    ky, ks = _agg_by_fr(known, value_col, frs)
    uy, us = _agg_by_fr(unknown, value_col, frs)
    plt.figure(figsize=(7.5, 4.8))
    plt.errorbar(frs, ky, yerr=ks, fmt="o-", capsize=4, markersize=7, linewidth=2,
                 color=KNOWN_COLOR, label="known fault rate")
    plt.errorbar(frs, uy, yerr=us, fmt="s--", capsize=4, markersize=7, linewidth=2,
                 color=UNKNOWN_COLOR, label="unknown fault rate")
    if annotate_ratio:
        for fr, kk, uu in zip(frs, ky, uy):
            if kk:
                plt.annotate(f"{uu / kk:.1f}x", (fr, uu), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9, color=UNKNOWN_COLOR)
    plt.xticks(frs)
    plt.xlabel("Injected fault rate")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")
    return out_path


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    domain_root = os.path.join(repo_root, "experimental results", DOMAIN)

    known_dir = os.path.join(domain_root, KNOWN_RUN)
    unknown_dir = os.path.join(domain_root, UNKNOWN_RUN)
    known = _load_run(known_dir)
    unknown = _load_run(unknown_dir)
    print(f"known: {len(known)} rows | unknown: {len(unknown)} rows (all visibilities pooled)")

    frs = sorted(set(known[FR_COL].unique()) & set(unknown[FR_COL].unique()))
    if not frs:
        raise SystemExit("No shared fault rates between known and unknown.")
    print(f"Shared fault rates: {frs}")

    out_dir = os.path.join(domain_root, "known_vs_unknown_comparison")
    os.makedirs(out_dir, exist_ok=True)
    created = []

    created.append(compare_plot(known, unknown, frs, RANK_COL, "Avg real-fault rank",
        "MiniGrid Empty (noise 0.5): rank by fault rate, known vs unknown fr",
        os.path.join(out_dir, "minigrid_noise0_5_rank_known_vs_unknown.png")))
    created.append(compare_plot(known, unknown, frs, TIME_COL, "Avg diagnosis time (sec)",
        "MiniGrid Empty (noise 0.5): time by fault rate, known vs unknown fr",
        os.path.join(out_dir, "minigrid_noise0_5_time_known_vs_unknown.png"),
        annotate_ratio=True))

    write_plot_provenance(out_dir, created,
                          input_sources=[known_dir, unknown_dir],
                          script_path=os.path.abspath(__file__))
    print("\nDone.")


if __name__ == "__main__":
    main()
