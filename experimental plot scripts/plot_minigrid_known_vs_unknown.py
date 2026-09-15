"""MiniGrid known- vs unknown-fault-rate comparison, along VISIBILITY, per (domain, noise).

Both the known-fr and ufr runs at a given noise share the same benchmark (26 faults x 3 seeds x 5
visibilities) at one injected fault rate. X = percent visible states (20..100); two lines (known,
unknown), each pooled over seeds/faults at that visibility (SEM error bars). Produces, into
<domain>/known_vs_unknown_comparison/:
  - avg real-fault rank  -> minigrid_noise{tag}_rank_known_vs_unknown.png
  - avg diagnosis time   -> minigrid_noise{tag}_time_known_vs_unknown.png (unknown/known ratio annotated)

Args: [noise_tag] [domain]. Defaults: 0_5, MiniGrid_SimpleCrossing_S11N2_v0.
Run from repo root, e.g.:
  ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_minigrid_known_vs_unknown.py" 0_5 MiniGrid_SimpleCrossing_S11N2_v0
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_provenance import write_plot_provenance

RANK_COL = "real_fault_rank"
TIME_COL = "diagnosis_time_sec"
VIS_COL = "percent_visible_states"

KNOWN_COLOR = "#1f77b4"
UNKNOWN_COLOR = "#d62728"

NOISE_TAG = sys.argv[1] if len(sys.argv) > 1 else "0_5"
DOMAIN = sys.argv[2] if len(sys.argv) > 2 else "MiniGrid_SimpleCrossing_S11N2_v0"
KNOWN_RUN = os.path.join("known", f"minigrid_bench_noise{NOISE_TAG}")
UNKNOWN_RUN = os.path.join("unknown", f"minigrid_bench_noise{NOISE_TAG}_ufr")
_PRETTY = {"MiniGrid_Empty_16x16_v0": "MiniGrid Empty-16x16",
           "MiniGrid_SimpleCrossing_S11N2_v0": "MiniGrid SimpleCrossing-S11N2"}.get(DOMAIN, DOMAIN)


def _load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "xlsx", "*.xlsx")))
    if not files:
        raise SystemExit(f"no result xlsx found in {run_dir}")
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    return df[df[RANK_COL].notna()].copy()


def _agg_by_vis(df, value_col, viss):
    ys, sems = [], []
    for v in viss:
        s = df[df[VIS_COL] == v][value_col]
        ys.append(s.mean())
        sems.append(s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
    return ys, sems


def compare_plot(known, unknown, viss, value_col, ylabel, title, out_path, annotate_ratio=False):
    ky, ks = _agg_by_vis(known, value_col, viss)
    uy, us = _agg_by_vis(unknown, value_col, viss)
    plt.figure(figsize=(7.5, 4.8))
    plt.errorbar(viss, ky, yerr=ks, fmt="o-", capsize=4, markersize=7, linewidth=2,
                 color=KNOWN_COLOR, label="known fault rate")
    plt.errorbar(viss, uy, yerr=us, fmt="s--", capsize=4, markersize=7, linewidth=2,
                 color=UNKNOWN_COLOR, label="unknown fault rate")
    if annotate_ratio:
        for v, kk, uu in zip(viss, ky, uy):
            if kk:
                plt.annotate(f"{uu / kk:.1f}x", (v, uu), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9, color=UNKNOWN_COLOR)
    plt.xticks(viss)
    plt.xlabel("Percent visible states")
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
    noise = known["minigrid_noise"].iloc[0]
    print(f"[{DOMAIN} noise {noise}] known: {len(known)} rows | unknown: {len(unknown)} rows")

    viss = sorted(set(known[VIS_COL].unique()) & set(unknown[VIS_COL].unique()))
    out_dir = os.path.join(domain_root, "known_vs_unknown_comparison")
    os.makedirs(out_dir, exist_ok=True)
    created = []

    created.append(compare_plot(known, unknown, viss, RANK_COL, "Avg real-fault rank",
        f"{_PRETTY} (noise {noise}): rank vs visibility, known vs unknown fr",
        os.path.join(out_dir, f"minigrid_noise{NOISE_TAG}_rank_known_vs_unknown.png")))
    created.append(compare_plot(known, unknown, viss, TIME_COL, "Avg diagnosis time (sec)",
        f"{_PRETTY} (noise {noise}): time vs visibility, known vs unknown fr",
        os.path.join(out_dir, f"minigrid_noise{NOISE_TAG}_time_known_vs_unknown.png"),
        annotate_ratio=True))

    write_plot_provenance(out_dir, created,
                          input_sources=[known_dir, unknown_dir],
                          script_path=os.path.abspath(__file__))
    print("Done.")


if __name__ == "__main__":
    main()
