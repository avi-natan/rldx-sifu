"""Fault-rate view across the FrozenLake way2 UNKNOWN-fault-rate experiments.

The three unknown experiments (exper1_fr05 / exper2_fr03 / exper3_fr08) differ only by
injected fault rate (0.5 / 0.3 / 0.8), all at epsilon=0.04. Pooling them gives a fault-rate
axis. Produces, into UNknown_fr_experiments/combined_fault_rate_view/:

  1. rank vs fault rate (pooled over visibility)   -> ..._rank_vs_faultrate.png
  2. time vs fault rate (pooled over visibility)   -> ..._time_vs_faultrate.png
  3. rank vs fault rate, one line per visibility    -> ..._rank_vs_faultrate_by_visibility.png
  4. time vs fault rate, one line per visibility     -> ..._time_vs_faultrate_by_visibility.png

Reads each experiment's merged xlsx (produced by plot_frozenlake_unknown.py); falls back to
globbing the per-map xlsx/ folder if a merged file is missing.

Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plots/plot_frozenlake_unknown_faultrate.py"
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_epsilon_sweep import _agg, _line_plot, RANK_COL, TIME_COL
from plot_provenance import write_plot_provenance

FR_COL = "real_fault_prob"
VIS_COL = "percent_visible_states"

EXPERIMENTS = ["exper1_fr05_eps_004", "exper2_fr03_eps_004", "exper3_fr08_eps_004"]


def load_unknown(base):
    frames = []
    for folder in EXPERIMENTS:
        exper_dir = os.path.join(base, folder)
        merged = os.path.join(exper_dir, f"{folder}_MERGED.xlsx")
        if os.path.exists(merged):
            frames.append(pd.read_excel(merged))
        else:
            files = sorted(glob.glob(os.path.join(exper_dir, "xlsx", "*.xlsx")))
            frames += [pd.read_excel(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df[FR_COL] = df[FR_COL].round(2)
    return df


def _multiline_by_visibility(df, y_col, ylabel, title, out_path):
    """One line per visibility level; x = fault rate, y = mean(y_col) with SEM."""
    plt.figure(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("viridis")
    vis_vals = sorted(df[VIS_COL].dropna().unique())
    frs = sorted(df[FR_COL].dropna().unique())
    for i, v in enumerate(vis_vals):
        sub = df[df[VIS_COL] == v]
        ys, sems = [], []
        for fr in frs:
            s = sub[sub[FR_COL] == fr][y_col]
            ys.append(s.mean())
            sems.append(s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
        color = cmap(i / max(1, len(vis_vals) - 1))
        plt.errorbar(frs, ys, yerr=sems, fmt="o-", capsize=3, markersize=5,
                     linewidth=1.6, color=color, label=f"vis={int(v)}%")
    plt.xticks(frs)
    plt.xlabel("Injected fault rate")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="visibility", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")
    return out_path


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(repo_root, "experimental results", "FrozenLake_v1",
                        "UNknown_fr_experiments")
    df = load_unknown(base)
    print(f"Unknown pooled: {len(df)} rows  fault rates: "
          f"{sorted(df[FR_COL].unique())}  (all epsilon=0.04)")

    out_dir = os.path.join(base, "combined_fault_rate_view")
    os.makedirs(out_dir, exist_ok=True)
    tag = "fl_way2_unknown"
    created = []

    # 1. rank vs fault rate (pooled over visibility)
    xs, ys, se, ns = _agg(df, FR_COL, RANK_COL)
    created.append(_line_plot(xs, ys, se, ns, "Injected fault rate", "Avg real-fault rank",
        "FrozenLake way2 (unknown fr): rank vs fault rate",
        os.path.join(out_dir, f"{tag}_rank_vs_faultrate.png")))

    # 2. time vs fault rate (pooled over visibility)
    xs, ys, se, ns = _agg(df, FR_COL, TIME_COL)
    created.append(_line_plot(xs, ys, se, ns, "Injected fault rate", "Avg diagnosis time (sec)",
        "FrozenLake way2 (unknown fr): time vs fault rate",
        os.path.join(out_dir, f"{tag}_time_vs_faultrate.png")))

    # 3-4. by visibility
    created.append(_multiline_by_visibility(df, RANK_COL, "Avg real-fault rank",
        "FrozenLake way2 (unknown fr): rank vs fault rate, by visibility",
        os.path.join(out_dir, f"{tag}_rank_vs_faultrate_by_visibility.png")))
    created.append(_multiline_by_visibility(df, TIME_COL, "Avg diagnosis time (sec)",
        "FrozenLake way2 (unknown fr): time vs fault rate, by visibility",
        os.path.join(out_dir, f"{tag}_time_vs_faultrate_by_visibility.png")))

    write_plot_provenance(out_dir, created, input_sources=[base],
                          script_path=os.path.abspath(__file__))
    print("\nDone.")


if __name__ == "__main__":
    main()
