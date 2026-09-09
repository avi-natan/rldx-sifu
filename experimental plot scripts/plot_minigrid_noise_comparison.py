"""Cross-noise comparison for the MiniGrid PO benchmark: one line per noise level (0.3/0.5/0.7).

Pools each noise level's per-task xlsx and plots the project-standard metric (avg real-fault rank,
mean +/- SEM; 1=best, 10=worst; random=5.5):
  A. rank vs VISIBILITY, one line per noise level   (fault rates pooled)
  B. rank vs FAULT RATE, one line per noise level    (visibilities pooled)
Output + PLOT_PROVENANCE.txt go to
  experimental results/MiniGrid_Empty_16x16_v0/noise_comparison/plots/.

Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_minigrid_noise_comparison.py"
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_provenance import write_plot_provenance

RANK_COL = "real_fault_rank"
FR_COL = "real_fault_prob"
VIS_COL = "percent_visible_states"
N_CAND = 10
RANDOM_RANK = (N_CAND + 1) / 2.0
# Optional args: 1=comma-separated noise tags (hard->easy), 2=domain. Defaults reproduce the
# original Empty 3-noise comparison; SimpleCrossing has only 0_5/0_7 trained.
NOISES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0_3", "0_5", "0_7"]
DOMAIN = sys.argv[2] if len(sys.argv) > 2 else "MiniGrid_Empty_16x16_v0"
COLORS = {"0_3": "#d62728", "0_5": "#ff7f0e", "0_7": "#1f77b4"}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The known-fr runs and this comparison live under the known/ subfolder.
DDIR = os.path.join(REPO, "experimental results", DOMAIN, "known")
OUT = os.path.join(DDIR, "noise_comparison", "plots")


def load(tag):
    files = glob.glob(os.path.join(DDIR, f"minigrid_bench_noise{tag}", "xlsx", "*.xlsx"))
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    return df[df[RANK_COL].notna()]


def _ms(s):
    return s.mean(), (s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)


def cmp_plot(dfs, x_col, x_label, title, out_path):
    plt.figure(figsize=(7.5, 4.8))
    for tag in NOISES:
        df = dfs[tag]
        xs = sorted(df[x_col].unique())
        ys, es = zip(*[_ms(df[df[x_col] == x][RANK_COL]) for x in xs])
        plt.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, markersize=6, linewidth=1.9,
                     color=COLORS[tag], label=f"noise {tag.replace('_', '.')}")
    plt.axhline(RANDOM_RANK, ls="--", color="grey", lw=1, label=f"random ({RANDOM_RANK:g})")
    plt.xticks(sorted(next(iter(dfs.values()))[x_col].unique()))
    plt.xlabel(x_label); plt.ylabel("Avg real-fault rank  (1 = best, 10 = worst)")
    plt.ylim(1, N_CAND); plt.title(title); plt.grid(True, alpha=0.3)
    plt.legend(title="env noise (action-success prob)", fontsize=8, title_fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path}")
    return out_path


def main():
    dfs = {t: load(t) for t in NOISES}
    os.makedirs(OUT, exist_ok=True)
    ns = {t: len(dfs[t]) for t in NOISES}
    _pretty = {"MiniGrid_Empty_16x16_v0": "MiniGrid Empty-16x16",
               "MiniGrid_SimpleCrossing_S11N2_v0": "MiniGrid SimpleCrossing-S11N2"}.get(DOMAIN, DOMAIN)
    suffix = f"{_pretty} (N per noise: {ns}, {N_CAND} candidates)"
    created = [
        cmp_plot(dfs, VIS_COL, "Visibility (% observed states)",
                 f"MiniGrid PO: rank vs visibility, by env noise\n{suffix}",
                 os.path.join(OUT, "minigrid_rank_vs_visibility_by_noise.png")),
        cmp_plot(dfs, FR_COL, "Fault rate",
                 f"MiniGrid PO: rank vs fault rate, by env noise\n{suffix}",
                 os.path.join(OUT, "minigrid_rank_vs_faultrate_by_noise.png")),
    ]
    write_plot_provenance(OUT, created,
                          input_sources=[os.path.join(DDIR, f"minigrid_bench_noise{t}", "xlsx") for t in NOISES])
    print("wrote:", *[os.path.basename(c) for c in created])


if __name__ == "__main__":
    main()
