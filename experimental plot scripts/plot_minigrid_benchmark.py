"""Curves for the MiniGrid PO fault-diagnosis benchmark (item 5), one noise level.

Matches the project convention (see plot_fault_rate_view.py): the metric is the AVG REAL-FAULT
RANK (mean +/- SEM), lower = better (1 = true fault ranked first, 10 = last of the 10 candidates).

Reads all per-task result xlsx in
    experimental results/MiniGrid_Empty_16x16_v0/minigrid_bench_noise{TAG}/xlsx/
(default TAG = 0_7; falls back to loose xlsx at the run-folder top) and writes into a plots/
subfolder there, plus a PLOT_PROVENANCE.txt.

Figures:
  A. rank vs VISIBILITY, one line per fault rate         -> ..._rank_vs_visibility_by_fr.png
  C. rank vs FAULT RATE, one line per visibility         -> ..._rank_vs_faultrate_by_visibility.png
  F. per-FAULT avg rank (sorted)                         -> ..._rank_per_fault.png

Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_minigrid_benchmark.py" [noise_tag]
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_provenance import write_plot_provenance

NOISE_TAG = sys.argv[1] if len(sys.argv) > 1 else "0_7"
N_CANDIDATES = 10
RANDOM_RANK = (N_CANDIDATES + 1) / 2.0   # 5.5 = expected rank under random ranking

RANK_COL = "real_fault_rank"
FR_COL = "real_fault_prob"
VIS_COL = "percent_visible_states"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "experimental results", "MiniGrid_Empty_16x16_v0",
                           f"minigrid_bench_noise{NOISE_TAG}")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

_NA = {0: "L", 1: "R", 2: "F"}
def fault_name(spec):
    d = {int(a): int(b) for a, b in re.findall(r"([012]):([012])", str(spec).replace(" ", ""))}
    return _NA[d[0]] + _NA[d[1]] + _NA[d[2]]


def load():
    files = glob.glob(os.path.join(RESULTS_DIR, "xlsx", "*.xlsx"))
    if not files:
        files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "*.xlsx")) if "MERGED" not in f]
    if not files:
        raise SystemExit(f"no result xlsx found in {RESULTS_DIR}")
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    df["fault"] = df["execution_fault"].map(fault_name)
    return df


def _mean_sem(series):
    m = series.mean()
    sem = series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0
    return m, sem


def plot_by_series(df, x_col, series_col, x_label, series_label, title, out_path,
                   series_fmt=lambda v: f"{v:g}"):
    """y = mean real-fault rank (+/- SEM); one line per value of series_col; x = x_col."""
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
    plt.axhline(RANDOM_RANK, ls="--", color="grey", lw=1, label=f"random ({RANDOM_RANK:g})")
    plt.xticks(sorted(df[x_col].dropna().unique()))
    plt.xlabel(x_label)
    plt.ylabel("Avg real-fault rank  (1 = best, 10 = worst)")
    plt.ylim(1, N_CANDIDATES)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title=series_label, fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")
    return out_path


def plot_pooled(df, x_col, x_label, title, out_path, y_col=RANK_COL,
                y_label="Avg real-fault rank  (1 = best, 10 = worst)",
                ylim=(1, N_CANDIDATES), baseline=RANDOM_RANK, label="all fault rates pooled"):
    """Single aggregate line: mean(y_col) +/- SEM vs x_col, POOLED over everything else."""
    xs = sorted(df[x_col].dropna().unique())
    ys, sems = [], []
    for x in xs:
        m, se = _mean_sem(df[df[x_col] == x][y_col]); ys.append(m); sems.append(se)
    plt.figure(figsize=(7.5, 4.8))
    plt.errorbar(xs, ys, yerr=sems, fmt="o-", capsize=3, markersize=7, linewidth=2.2,
                 color="#1f77b4", label=label)
    if baseline is not None:
        plt.axhline(baseline, ls="--", color="grey", lw=1, label=f"random ({baseline:g})")
    plt.xticks(xs); plt.xlabel(x_label); plt.ylabel(y_label)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path}")
    return out_path


def main():
    df = load()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    noise = df["minigrid_noise"].iloc[0]
    suffix = f"MiniGrid Empty-16x16, noise {noise} (N={len(df)}, {N_CANDIDATES} candidates)"
    tag = f"minigrid_noise{NOISE_TAG}"
    created = []

    created.append(plot_by_series(
        df, x_col=VIS_COL, series_col=FR_COL,
        x_label="Visibility (% observed states)", series_label="fault rate",
        title=f"MiniGrid PO: rank vs visibility, by fault rate\n{suffix}",
        out_path=os.path.join(PLOTS_DIR, f"{tag}_rank_vs_visibility_by_fr.png")))

    created.append(plot_by_series(
        df, x_col=FR_COL, series_col=VIS_COL,
        x_label="Fault rate", series_label="visibility",
        title=f"MiniGrid PO: rank vs fault rate, by visibility\n{suffix}",
        out_path=os.path.join(PLOTS_DIR, f"{tag}_rank_vs_faultrate_by_visibility.png"),
        series_fmt=lambda v: f"{int(v)}%"))

    # --- pooled/aggregate rank curves: all fault rates combined into a single line ---
    created.append(plot_pooled(
        df, x_col=VIS_COL, x_label="Visibility (% observed states)",
        title=f"MiniGrid PO: rank vs visibility (ALL fault rates combined)\n{suffix}",
        out_path=os.path.join(PLOTS_DIR, f"{tag}_rank_vs_visibility_pooled.png")))

    created.append(plot_pooled(
        df, x_col=FR_COL, x_label="Fault rate",
        title=f"MiniGrid PO: rank vs fault rate (ALL visibilities combined)\n{suffix}",
        out_path=os.path.join(PLOTS_DIR, f"{tag}_rank_vs_faultrate_pooled.png"),
        label="all visibilities pooled"))

    # --- diagnosis TIME curves (pooled) ---
    created.append(plot_pooled(
        df, x_col=VIS_COL, x_label="Visibility (% observed states)",
        title=f"MiniGrid PO: avg diagnosis time vs visibility\n{suffix}",
        out_path=os.path.join(PLOTS_DIR, f"{tag}_time_vs_visibility.png"),
        y_col="diagnosis_time_sec", y_label="Avg diagnosis time (s)", ylim=None,
        baseline=None, label="all conditions pooled"))

    created.append(plot_pooled(
        df, x_col=FR_COL, x_label="Fault rate",
        title=f"MiniGrid PO: avg diagnosis time vs fault rate\n{suffix}",
        out_path=os.path.join(PLOTS_DIR, f"{tag}_time_vs_faultrate.png"),
        y_col="diagnosis_time_sec", y_label="Avg diagnosis time (s)", ylim=None,
        baseline=None, label="all conditions pooled"))

    write_plot_provenance(PLOTS_DIR, created, input_sources=[os.path.join(RESULTS_DIR, "xlsx")])
    print("wrote:", *[os.path.basename(c) for c in created])


if __name__ == "__main__":
    main()
