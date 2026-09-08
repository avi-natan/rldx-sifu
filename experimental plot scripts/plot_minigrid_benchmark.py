"""Curves for the MiniGrid PO fault-diagnosis benchmark (item 5), for one noise level.

Reads all per-task result xlsx in
    experimental results/MiniGrid_Empty_16x16_v0/minigrid_bench_noise{TAG}/
(default TAG = 0_7) and writes accuracy curves into a plots/ subfolder there, plus a
PLOT_PROVENANCE.txt (standing project rule).

Curves produced:
  1. accuracy vs VISIBILITY (top-1, top-3) + mean-rank            -> the partial-observability curve
  2. top-1 vs VISIBILITY, one line per fault rate                 -> both difficulty axes at once
  3. accuracy vs FAULT RATE (top-1, top-3)                        -> signal-strength curve
  4. per-FAULT top-1 (sorted)                                     -> the detectability gradient

Run from repo root:
  ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_minigrid_benchmark.py" [noise_tag]
"""
import os, sys, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_provenance import write_plot_provenance

NOISE_TAG = sys.argv[1] if len(sys.argv) > 1 else "0_7"
N_CANDIDATES = 10
RANDOM_TOP1 = 1.0 / N_CANDIDATES   # 0.10 baseline reference

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
    # per-task xlsx live in run_folder/xlsx/ (current layout); fall back to loose at the top
    # (older runs) for backward compatibility.
    files = glob.glob(os.path.join(RESULTS_DIR, "xlsx", "*.xlsx"))
    if not files:
        files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "*.xlsx")) if "MERGED" not in f]
    if not files:
        raise SystemExit(f"no result xlsx found in {RESULTS_DIR}")
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    df["fault"] = df["execution_fault"].map(fault_name)
    return df, files


def main():
    df, files = load()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    noise = df["minigrid_noise"].iloc[0]
    title_suffix = f"MiniGrid Empty-16x16, noise {noise}  (N={len(df)}, {N_CANDIDATES} candidates)"
    created = []

    # ---- 1. accuracy vs visibility (+ mean rank) ----
    g = df.groupby("percent_visible_states")
    vis = sorted(df["percent_visible_states"].unique())
    top1 = [g.get_group(v)["execution_fault_in_top1"].mean() for v in vis]
    top3 = [g.get_group(v)["execution_fault_in_top3"].mean() for v in vis]
    rank = [g.get_group(v)["real_fault_rank"].mean() for v in vis]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(vis, top1, "o-", color="#1f77b4", label="top-1 accuracy")
    ax.plot(vis, top3, "s-", color="#2ca02c", label="top-3 accuracy")
    ax.axhline(RANDOM_TOP1, ls="--", color="grey", lw=1, label=f"random top-1 ({RANDOM_TOP1:.2f})")
    ax.set_xlabel("% of trajectory observed"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax2 = ax.twinx(); ax2.plot(vis, rank, "^:", color="#d62728", label="mean rank")
    ax2.set_ylabel("mean true-fault rank (1=best, 10=worst)"); ax2.set_ylim(1, N_CANDIDATES)
    ax.set_title("Diagnosis accuracy vs visibility\n" + title_suffix)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(PLOTS_DIR, "1_accuracy_vs_visibility.png")
    fig.savefig(p, dpi=150); plt.close(fig); created.append(p)

    # ---- 2. top-1 vs visibility, one line per fault rate ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fr, color in zip(sorted(df["real_fault_prob"].unique()), ["#9467bd", "#1f77b4", "#ff7f0e"]):
        sub = df[df["real_fault_prob"] == fr].groupby("percent_visible_states")["execution_fault_in_top1"].mean()
        ax.plot(sub.index, sub.values, "o-", color=color, label=f"fault rate {fr}")
    ax.axhline(RANDOM_TOP1, ls="--", color="grey", lw=1, label=f"random ({RANDOM_TOP1:.2f})")
    ax.set_xlabel("% of trajectory observed"); ax.set_ylabel("top-1 accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Top-1 vs visibility, by fault rate\n" + title_suffix)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(PLOTS_DIR, "2_top1_vs_visibility_by_faultrate.png")
    fig.savefig(p, dpi=150); plt.close(fig); created.append(p)

    # ---- 3. accuracy vs fault rate ----
    g = df.groupby("real_fault_prob"); frs = sorted(df["real_fault_prob"].unique())
    top1 = [g.get_group(fr)["execution_fault_in_top1"].mean() for fr in frs]
    top3 = [g.get_group(fr)["execution_fault_in_top3"].mean() for fr in frs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frs, top1, "o-", color="#1f77b4", label="top-1 accuracy")
    ax.plot(frs, top3, "s-", color="#2ca02c", label="top-3 accuracy")
    ax.axhline(RANDOM_TOP1, ls="--", color="grey", lw=1, label=f"random top-1 ({RANDOM_TOP1:.2f})")
    ax.set_xlabel("injected fault rate"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_xticks(frs); ax.set_title("Diagnosis accuracy vs fault rate\n" + title_suffix)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(PLOTS_DIR, "3_accuracy_vs_faultrate.png")
    fig.savefig(p, dpi=150); plt.close(fig); created.append(p)

    # ---- 4. per-fault top-1 (detectability gradient) ----
    pf = df.groupby("fault")["execution_fault_in_top1"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if v < RANDOM_TOP1 * 2 else "#1f77b4" for v in pf.values]
    ax.bar(range(len(pf)), pf.values, color=colors)
    ax.axhline(RANDOM_TOP1, ls="--", color="grey", lw=1, label=f"random ({RANDOM_TOP1:.2f})")
    ax.set_xticks(range(len(pf))); ax.set_xticklabels(pf.index, rotation=90, fontsize=7)
    ax.set_ylabel("top-1 accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Per-fault top-1 (detectability gradient; label = L,R,F -> mapping)\n" + title_suffix)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); p = os.path.join(PLOTS_DIR, "4_per_fault_top1.png")
    fig.savefig(p, dpi=150); plt.close(fig); created.append(p)

    write_plot_provenance(PLOTS_DIR, created, input_sources=[RESULTS_DIR + " (per-task xlsx)"])
    print("wrote:", *[os.path.basename(c) for c in created])


if __name__ == "__main__":
    main()
