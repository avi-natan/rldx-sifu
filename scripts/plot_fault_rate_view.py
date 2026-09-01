"""Cross-experiment 'fault-rate view' plots for the FrozenLake way2 known-fr sweeps.

Merges both known-fr experiments (fr 0.3 from one folder, fr 0.5 & 0.8 from the
other) into one frame, pools over epsilon (epsilon is an inert compute knob), and
produces:

  A. rank vs visibility, one line per fault rate  -> ..._rank_vs_visibility_by_fr.png
  C. rank vs fault rate,  one line per visibility  -> ..._rank_vs_faultrate_by_visibility.png

Run from repo root:
  ./.venv_domains/Scripts/python.exe scripts/plot_fault_rate_view.py
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANK_COL = "real_fault_rank"
FR_COL = "real_fault_prob"
VIS_COL = "percent_visible_states"


def load_folder(folder):
    frames = [pd.read_excel(f) for f in sorted(glob.glob(os.path.join(folder, "*.xlsx")))]
    if not frames:
        raise SystemExit(f"No xlsx in {folder}")
    return pd.concat(frames, ignore_index=True)


def _mean_sem(series):
    m = series.mean()
    sem = series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0
    return m, sem


def plot_by_series(df, x_col, series_col, x_label, series_label, title, out_path,
                   series_fmt=lambda v: f"{v:g}"):
    """Generic: y = mean(rank); one line per value of series_col; x = x_col."""
    plt.figure(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("viridis")
    series_vals = sorted(df[series_col].dropna().unique())
    for i, sv in enumerate(series_vals):
        sub = df[df[series_col] == sv]
        xs = sorted(sub[x_col].dropna().unique())
        ys, sems = [], []
        for x in xs:
            m, se = _mean_sem(sub[sub[x_col] == x][RANK_COL])
            ys.append(m); sems.append(se)
        color = cmap(i / max(1, len(series_vals) - 1))
        plt.errorbar(xs, ys, yerr=sems, fmt="o-", capsize=3, markersize=6,
                     linewidth=1.8, color=color, label=f"{series_label}={series_fmt(sv)}")
    plt.xticks(sorted(df[x_col].dropna().unique()))
    plt.xlabel(x_label)
    plt.ylabel("Avg real-fault rank")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title=series_label, fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(repo_root, "experimental results", "FrozenLake_v1",
                        "known_fr_experiments")
    fr03 = load_folder(os.path.join(base, "known_fr_03_eps_sweep", "xlsx"))
    fr0508 = load_folder(os.path.join(base, "known_fr_05_08_eps_sweep", "xlsx"))
    df = pd.concat([fr03, fr0508], ignore_index=True)
    df[FR_COL] = df[FR_COL].round(2)

    out_dir = os.path.join(base, "combined_fault_rate_view")
    os.makedirs(out_dir, exist_ok=True)
    frs = sorted(df[FR_COL].unique())
    print(f"Merged rows: {len(df)}  fault rates: {frs}  "
          f"(pooled over epsilon)\n  -> {out_dir}\n")

    tag = "fl_way2_known_fr"

    # A. rank vs visibility, one line per fault rate
    plot_by_series(
        df, x_col=VIS_COL, series_col=FR_COL,
        x_label="Visibility (% observed states)", series_label="fault rate",
        title="FrozenLake way2 (known fr): rank vs visibility, by fault rate",
        out_path=os.path.join(out_dir, f"{tag}_rank_vs_visibility_by_fr.png"))

    # C. rank vs fault rate, one line per visibility
    plot_by_series(
        df, x_col=FR_COL, series_col=VIS_COL,
        x_label="Fault rate", series_label="visibility",
        title="FrozenLake way2 (known fr): rank vs fault rate, by visibility",
        out_path=os.path.join(out_dir, f"{tag}_rank_vs_faultrate_by_visibility.png"),
        series_fmt=lambda v: f"{int(v)}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
