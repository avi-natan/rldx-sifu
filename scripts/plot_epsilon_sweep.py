"""Plot an epsilon-sweep experiment (one xlsx per epsilon) into 4 summary figures.

Produces, for a folder of result xlsx files:
  1. avg real-fault rank   vs epsilon
  2. avg diagnosis time    vs epsilon
  3. avg real-fault rank   vs visibility (%)
  4. avg diagnosis time    vs visibility (%)

Rank plots aggregate over every row in the folder; "vs epsilon" groups by the
`epsilon` column, "vs visibility" groups by `percent_visible_states` (pooled
over all epsilons, which is fair here because epsilon is an inert compute knob).
Points carry SEM error bars and an `n=` annotation.

Default target is the Taxi-v4 full known-fr epsilon sweep; override with --input/--out.

Run from repo root, e.g.:
  ./.venv_domains/Scripts/python.exe scripts/plot_epsilon_sweep.py
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANK_COL = "real_fault_rank"
TIME_COL = "diagnosis_time_sec"
EPS_COL = "epsilon"
VIS_COL = "percent_visible_states"


def load_folder(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    if not files:
        raise SystemExit(f"No .xlsx files found in {input_dir}")
    frames = []
    for f in files:
        df = pd.read_excel(f)
        frames.append(df)
        print(f"  loaded {os.path.basename(f)}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def _agg(df, group_col, value_col):
    """Return sorted x, mean(y), sem(y), n per group."""
    g = df.dropna(subset=[group_col, value_col]).groupby(group_col)[value_col]
    xs = sorted(g.groups.keys())
    means = [g.get_group(x).mean() for x in xs]
    sems = [(g.get_group(x).std(ddof=1) / np.sqrt(len(g.get_group(x)))
             if len(g.get_group(x)) > 1 else 0.0) for x in xs]
    ns = [len(g.get_group(x)) for x in xs]
    return xs, means, sems, ns


def _line_plot(xs, ys, sems, ns, xlabel, ylabel, title, out_path, xticks=True):
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(xs, ys, yerr=sems, fmt="o-", capsize=4, color="#1f77b4",
                 ecolor="#888888", markersize=6, linewidth=1.8)
    if xticks:
        plt.xticks(xs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(f"n={n}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color="#555555")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def _multiline_plot(df, x_col, y_col, series_col, xlabel, ylabel, title, out_path):
    """One line per value of series_col (e.g. epsilon), x=x_col, y=mean(y_col)."""
    plt.figure(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("viridis")
    series_vals = sorted(df[series_col].dropna().unique())
    for i, sv in enumerate(series_vals):
        sub = df[df[series_col] == sv]
        xs, ys, sems, _ = _agg(sub, x_col, y_col)
        color = cmap(i / max(1, len(series_vals) - 1))
        plt.errorbar(xs, ys, yerr=sems, fmt="o-", capsize=3, markersize=5,
                     linewidth=1.6, color=color, label=f"eps={sv:g}")
    xticks = sorted(df[x_col].dropna().unique())
    plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="epsilon", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_in = os.path.join(
        repo_root, "experimental results", "Taxi_v4",
        "experiment1-known_fr_epsilon_sweep")
    default_out = os.path.join(repo_root, "experimental results", "Taxi_v4", "plots")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=default_in, help="folder of epsilon xlsx files")
    ap.add_argument("--out", default=default_out, help="folder to write plots into")
    ap.add_argument("--tag", default="taxi_v4_known_fr",
                    help="filename prefix for the output plots")
    ap.add_argument("--title", default="Taxi-v4 (known fr) epsilon sweep",
                    help="title stem shown on each figure")
    args = ap.parse_args()

    print(f"Loading from: {args.input}")
    df = load_folder(args.input)
    os.makedirs(args.out, exist_ok=True)
    print(f"Total rows: {len(df)}  |  writing plots to: {args.out}\n")

    # 1. rank vs epsilon
    xs, ys, se, ns = _agg(df, EPS_COL, RANK_COL)
    _line_plot(xs, ys, se, ns, "Epsilon (MC CI half-width)", "Avg real-fault rank",
               f"{args.title}: rank vs epsilon",
               os.path.join(args.out, f"{args.tag}_rank_vs_epsilon.png"))

    # 2. time vs epsilon
    xs, ys, se, ns = _agg(df, EPS_COL, TIME_COL)
    _line_plot(xs, ys, se, ns, "Epsilon (MC CI half-width)", "Avg diagnosis time (sec)",
               f"{args.title}: time vs epsilon",
               os.path.join(args.out, f"{args.tag}_time_vs_epsilon.png"))

    # 3. rank vs visibility
    xs, ys, se, ns = _agg(df, VIS_COL, RANK_COL)
    _line_plot(xs, ys, se, ns, "Visibility (% observed states)", "Avg real-fault rank",
               f"{args.title}: rank vs visibility",
               os.path.join(args.out, f"{args.tag}_rank_vs_visibility.png"))

    # 4. time vs visibility
    xs, ys, se, ns = _agg(df, VIS_COL, TIME_COL)
    _line_plot(xs, ys, se, ns, "Visibility (% observed states)", "Avg diagnosis time (sec)",
               f"{args.title}: time vs visibility",
               os.path.join(args.out, f"{args.tag}_time_vs_visibility.png"))

    # 5. rank vs visibility, one line per epsilon (proves epsilon-invariance at every vis)
    _multiline_plot(df, VIS_COL, RANK_COL, EPS_COL,
                    "Visibility (% observed states)", "Avg real-fault rank",
                    f"{args.title}: rank vs visibility, by epsilon",
                    os.path.join(args.out, f"{args.tag}_rank_vs_visibility_by_epsilon.png"))

    # 6. time vs visibility, one line per epsilon (lines fan out -> epsilon costs time)
    _multiline_plot(df, VIS_COL, TIME_COL, EPS_COL,
                    "Visibility (% observed states)", "Avg diagnosis time (sec)",
                    f"{args.title}: time vs visibility, by epsilon",
                    os.path.join(args.out, f"{args.tag}_time_vs_visibility_by_epsilon.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
