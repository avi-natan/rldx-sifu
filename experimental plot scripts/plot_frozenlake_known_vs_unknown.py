"""FrozenLake way2 known- vs unknown-fault-rate comparison, along fault rate.

The unknown experiments are all at epsilon=0.04; the known experiments are epsilon sweeps
that include 0.04. To compare apples-to-apples we RESTRICT the known side to epsilon=0.04
(essential for the time plot, since epsilon drives MC compute). X = injected fault rate
(0.3 / 0.5 / 0.8); two lines (known, unknown), each pooled over all visibility levels and
maps at that fault rate (SEM error bars). Produces, into
FrozenLake_v1/known_vs_unknown_comparison/:

  - avg real-fault rank -> fl_way2_rank_known_vs_unknown.png
  - avg diagnosis time  -> fl_way2_time_known_vs_unknown.png

Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plots/plot_frozenlake_known_vs_unknown.py"
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
EPS_COL = "epsilon"
FR_COL = "real_fault_prob"

MATCH_EPSILON = 0.04
KNOWN_COLOR = "#1f77b4"
UNKNOWN_COLOR = "#d62728"

UNKNOWN_EXPERIMENTS = ["exper1_fr05_eps_004", "exper2_fr03_eps_004", "exper3_fr08_eps_004"]
KNOWN_SWEEPS = ["known_fr_03_eps_sweep", "known_fr_05_08_eps_sweep"]


def _load_glob(paths):
    frames = [pd.read_excel(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def load_known(fl_root):
    base = os.path.join(fl_root, "known_fr_experiments")
    files = []
    for sub in KNOWN_SWEEPS:
        files += sorted(glob.glob(os.path.join(base, sub, "xlsx", "*.xlsx")))
    df = _load_glob(files)
    df[FR_COL] = df[FR_COL].round(2)
    df = df[df[EPS_COL].round(4) == MATCH_EPSILON]  # apples-to-apples on epsilon
    return df, base


def load_unknown(fl_root):
    base = os.path.join(fl_root, "UNknown_fr_experiments")
    frames = []
    for folder in UNKNOWN_EXPERIMENTS:
        merged = os.path.join(base, folder, f"{folder}_MERGED.xlsx")
        if os.path.exists(merged):
            frames.append(pd.read_excel(merged))
        else:
            frames += [pd.read_excel(f)
                       for f in glob.glob(os.path.join(base, folder, "xlsx", "*.xlsx"))]
    df = pd.concat(frames, ignore_index=True)
    df[FR_COL] = df[FR_COL].round(2)
    return df, base


def _agg_by_fr(df, value_col, frs):
    ys, sems = [], []
    for fr in frs:
        s = df[df[FR_COL] == fr][value_col]
        ys.append(s.mean())
        sems.append(s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
    return ys, sems


def compare_plot(known, unknown, frs, value_col, ylabel, title, out_path,
                 annotate_ratio=False):
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
    fl_root = os.path.join(repo_root, "experimental results", "FrozenLake_v1")

    known, known_base = load_known(fl_root)
    unknown, unknown_base = load_unknown(fl_root)
    print(f"known (eps={MATCH_EPSILON}): {len(known)} rows | unknown: {len(unknown)} rows")

    frs = sorted(set(known[FR_COL].unique()) & set(unknown[FR_COL].unique()))
    if not frs:
        raise SystemExit("No shared fault rates between known and unknown.")
    print(f"Shared fault rates: {frs}")

    out_dir = os.path.join(fl_root, "known_vs_unknown_comparison")
    os.makedirs(out_dir, exist_ok=True)
    created = []

    created.append(compare_plot(known, unknown, frs, RANK_COL, "Avg real-fault rank",
        f"FrozenLake way2 known vs unknown fr: rank by fault rate (eps={MATCH_EPSILON})",
        os.path.join(out_dir, "fl_way2_rank_known_vs_unknown.png")))
    created.append(compare_plot(known, unknown, frs, TIME_COL, "Avg diagnosis time (sec)",
        f"FrozenLake way2 known vs unknown fr: time by fault rate (eps={MATCH_EPSILON})",
        os.path.join(out_dir, "fl_way2_time_known_vs_unknown.png"),
        annotate_ratio=True))

    write_plot_provenance(out_dir, created,
                          input_sources=[known_base, unknown_base],
                          script_path=os.path.abspath(__file__))
    print("\nDone.")


if __name__ == "__main__":
    main()
