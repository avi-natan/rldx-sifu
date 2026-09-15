"""Per-domain comparison plots: brute-force (full ufr) vs v1 (rounds=15) vs known-fault-rate.

For each domain (Cross0.7, Cross0.5, Empty0.5, Taxi) two figures are produced:
  * visibility vs avg real_fault_rank
  * visibility vs avg diagnosis_time_sec
each with THREE lines (full / v1 / known).

Input: plot_data.csv (tidy: domain,variant,visibility,n,mean_rank,mean_time) sitting next to the
output folder -- produced by aggregating the cluster xlsx (MiniGrid rows filtered to fault rate 0.5
for fairness; Taxi is the fixed hard-class2 benchmark). Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_v1_vs_full_vs_known.py"
Plots + PLOT_PROVENANCE.txt land in ufr_experiments/comparison_plots_v1_full_known/ (a NEW folder;
the existing experimental-results plot folders are left untouched).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_provenance import write_plot_provenance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "ufr_experiments", "comparison_plots_v1_full_known")
CSV = os.path.join(OUT_DIR, "plot_data.csv")

# consistent look per variant across every figure
STYLE = {
    "full":  dict(label="brute-force (full ufr)", color="#1f77b4", marker="o"),
    "v1":    dict(label="v1 (rounds=15)",         color="#d62728", marker="s"),
    "known": dict(label="known fault rate",       color="#2ca02c", marker="^"),
}
VAR_ORDER = ["full", "v1", "known"]
DOMAIN_ORDER = ["Cross0.7", "Cross0.5", "Empty0.5", "Taxi"]


def one_plot(df_dom, domain, ycol, ylabel, title_metric, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    for var in VAR_ORDER:
        sub = df_dom[df_dom["variant"] == var].sort_values("visibility")
        if sub.empty:
            continue
        ax.plot(sub["visibility"], sub[ycol], **STYLE[var], linewidth=2, markersize=7)
    ax.set_xlabel("visibility (% observed states)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{domain}: {title_metric} vs visibility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ycol == "mean_rank":
        ax.set_ylim(bottom=1.0)   # rank 1 is the best possible
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    df = pd.read_csv(CSV)
    created = []
    domains = [d for d in DOMAIN_ORDER if d in set(df["domain"])]
    for domain in domains:
        dd = df[df["domain"] == domain]
        tag = domain.replace(".", "_")
        created.append(one_plot(dd, domain, "mean_rank", "avg real-fault rank (lower is better)",
                                "rank", f"{tag}__visibility_vs_rank.png"))
        created.append(one_plot(dd, domain, "mean_time", "avg diagnosis time (s)",
                                "time", f"{tag}__visibility_vs_time.png"))
    write_plot_provenance(OUT_DIR, created, input_sources=[CSV])
    print(f"wrote {len(created)} plots to {OUT_DIR}")
    for c in created:
        print("  ", os.path.basename(c))


if __name__ == "__main__":
    main()
