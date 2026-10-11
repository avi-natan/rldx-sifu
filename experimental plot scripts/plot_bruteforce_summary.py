"""Plots for the brute-force / v2b sims-study summary tables.

Reads the five CSVs in
    bruteforce_unknown_eps_experiments/summary_tables/
(produced from the per-instance xlsx) and renders one or more figures per table
into that folder's `plots/` subdir. Run from the repo root:

    ./.venv_domains/Scripts/python.exe "experimental plot scripts/plot_bruteforce_summary.py"

Design rules (dataviz skill): NO dual-axis (rank and sims always live in separate
panels); colorblind-safe Okabe-Ito palette; COLOR follows the entity (domain / config)
while LINESTYLE carries the method (plain vs Jeffreys, brute-force vs v2b); legend for
>=2 series; recessive grid.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# shared provenance helper (same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_provenance import write_plot_provenance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(REPO, "bruteforce_unknown_eps_experiments", "summary_tables")
OUT = os.path.join(TAB, "plots")
os.makedirs(OUT, exist_ok=True)

# ---- Okabe-Ito colorblind-safe palette; color = entity ----
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
DOM_COLOR = {"FL": BLUE, "Taxi": ORANGE, "Cross": GREEN}
DOM_LABEL = {"FL": "FrozenLake", "Taxi": "Taxi", "Cross": "Cross (SimpleCrossing)"}
DOMS = ["FL", "Taxi", "Cross"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.autolayout": False,
})

created = []


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    created.append(p)
    print("  wrote", name)


def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")


# ======================================================================
# TABLE 1 & 2 : FrozenLake epsilon sweep (two configs) + comparison
# ======================================================================
t1 = pd.read_csv(os.path.join(TAB, "table1_FL_epsilon_sweep.csv"))
t2 = pd.read_csv(os.path.join(TAB, "table2_FL_epsilon_sweep_min20_batch10.csv"))


def eps_fig(df, title_cfg, color, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(df.epsilon, df.avg_real_fault_rank, "-o", color=color, lw=2, ms=7)
    style_ax(ax[0], "epsilon", "avg real-fault rank (lower better)",
             "Rank vs epsilon")
    ax[1].plot(df.epsilon, df.avg_total_sims, "-o", color=color, lw=2, ms=7)
    ax[1].set_yscale("log")
    style_ax(ax[1], "epsilon", "avg total simulations / diagnosis (log)",
             "Simulations vs epsilon")
    fig.suptitle(title_cfg, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, fname)


eps_fig(t1, "FrozenLake epsilon sweep - brute-force adaptive (original min/batch)",
        BLUE, "table1_FL_eps_sweep.png")
eps_fig(t2, "FrozenLake epsilon sweep - min=20, batch=10",
        VERM, "table2_FL_eps_sweep_min20_b10.png")

# comparison overlay: config = entity -> color; both solid, markers differ
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t1.epsilon, t1.avg_real_fault_rank, "-o", color=BLUE, lw=2, ms=7, label="original min/batch")
ax[0].plot(t2.epsilon, t2.avg_real_fault_rank, "-s", color=VERM, lw=2, ms=7, label="min=20, batch=10")
style_ax(ax[0], "epsilon", "avg real-fault rank (lower better)", "Rank vs epsilon")
ax[0].legend()
ax[1].plot(t1.epsilon, t1.avg_total_sims, "-o", color=BLUE, lw=2, ms=7, label="original min/batch")
ax[1].plot(t2.epsilon, t2.avg_total_sims, "-s", color=VERM, lw=2, ms=7, label="min=20, batch=10")
ax[1].set_yscale("log")
style_ax(ax[1], "epsilon", "avg total simulations / diagnosis (log)", "Simulations vs epsilon")
ax[1].legend()
fig.suptitle("FrozenLake epsilon sweep - effect of the min-tries / batch-size change",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "table1_2_FL_eps_compare.png")

# ======================================================================
# TABLE 3 : fixed tries, all three domains
# ======================================================================
t3 = pd.read_csv(os.path.join(TAB, "table3_fixedtries_all_domains.csv"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for d in DOMS:
    ax[0].plot(t3.fixed_tries, t3[f"{d}_rank"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label=DOM_LABEL[d])
    ax[1].plot(t3.fixed_tries, t3[f"{d}_sims"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label=DOM_LABEL[d])
ax[0].set_xscale("log"); ax[0].set_xticks(t3.fixed_tries); ax[0].set_xticklabels(t3.fixed_tries)
style_ax(ax[0], "fixed tries per (pair, gap)  [log]", "avg real-fault rank (lower better)", "Rank vs fixed tries")
ax[0].legend()
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xticks(t3.fixed_tries); ax[1].set_xticklabels(t3.fixed_tries)
style_ax(ax[1], "fixed tries per (pair, gap)  [log]", "avg total simulations / diagnosis (log)", "Simulations vs fixed tries")
ax[1].legend()
fig.suptitle("Fixed tries 5->800 (all domains)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "table3_fixedtries.png")

# efficiency view: rank vs sims (the accuracy-per-sim tradeoff)
fig, ax = plt.subplots(figsize=(7, 5))
for d in DOMS:
    ax.plot(t3[f"{d}_sims"], t3[f"{d}_rank"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label=DOM_LABEL[d])
ax.set_xscale("log")
style_ax(ax, "avg total simulations / diagnosis (log)", "avg real-fault rank (lower better)",
         "Accuracy vs cost - fixed tries")
ax.legend()
fig.tight_layout()
save(fig, "table3_efficiency_rank_vs_sims.png")

# ======================================================================
# TABLE 4 : Jeffreys fixed tries vs plain fixed tries (comparison)
# ======================================================================
t4 = pd.read_csv(os.path.join(TAB, "table4_jeffreys_fixedtries_vs_fixedtries.csv"))

# small multiples: per domain, jeffreys (solid) vs plain (dashed)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)
for ax, d in zip(axes, DOMS):
    ax.plot(t4.fixed_tries, t4[f"{d}_rank"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label="Jeffreys")
    ax.plot(t3.fixed_tries, t3[f"{d}_rank"], "--s", color=DOM_COLOR[d], lw=2, ms=6, mfc="white", label="plain (1e-12 floor)")
    ax.set_xscale("log"); ax.set_xticks(t4.fixed_tries); ax.set_xticklabels(t4.fixed_tries, rotation=45)
    style_ax(ax, "fixed tries [log]", "avg real-fault rank" if d == "FL" else "", DOM_LABEL[d])
    ax.legend()
fig.suptitle("Jeffreys vs plain floor at equal budget (solid = Jeffreys, dashed = plain)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "table4_jeffreys_vs_plain.png")

# delta plot: jeffreys - plain (negative = jeffreys better)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.axhline(0, color="#555555", lw=1)
for d in DOMS:
    ax.plot(t4.fixed_tries, t4[f"{d}_vs_fixed_rank"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label=DOM_LABEL[d])
ax.set_xscale("log"); ax.set_xticks(t4.fixed_tries); ax.set_xticklabels(t4.fixed_tries)
style_ax(ax, "fixed tries per (pair, gap)  [log]", "rank(Jeffreys) - rank(plain)",
         "Jeffreys effect on rank  (below 0 = Jeffreys better)")
ax.legend()
fig.tight_layout()
save(fig, "table4_jeffreys_delta.png")

# ======================================================================
# TABLE 5 : v2b vs Jeffreys fixed tries at EQUAL budget (comparison)
# ======================================================================
t5 = pd.read_csv(os.path.join(TAB, "table5_v2b_vs_jeffreys_fixedtries.csv"))
# jeffreys reference restricted to the N in v2b
t4i = t4.set_index("fixed_tries")

# small multiples: per domain, v2b (solid) vs jeffreys-fixed (dashed)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)
for ax, d in zip(axes, DOMS):
    ax.plot(t5.N, t5[f"{d}_rank"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label="v2b (smart alloc)")
    jr = [t4i.loc[n, f"{d}_rank"] for n in t5.N]
    ax.plot(t5.N, jr, "--s", color=DOM_COLOR[d], lw=2, ms=6, mfc="white", label="brute-force (Jeffreys fixed)")
    ax.set_xscale("log"); ax.set_xticks(t5.N); ax.set_xticklabels(t5.N)
    style_ax(ax, "budget N [log]", "avg real-fault rank" if d == "FL" else "", DOM_LABEL[d])
    ax.legend()
fig.suptitle("v2b vs brute-force at EQUAL sim budget (solid = v2b, dashed = brute-force)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "table5_v2b_vs_bruteforce.png")

# delta plot: v2b - jeffreys (negative = v2b better)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.axhline(0, color="#555555", lw=1)
for d in DOMS:
    ax.plot(t5.N, t5[f"{d}_vs_jeff_rank"], "-o", color=DOM_COLOR[d], lw=2, ms=6, label=DOM_LABEL[d])
ax.set_xscale("log"); ax.set_xticks(t5.N); ax.set_xticklabels(t5.N)
style_ax(ax, "budget N [log]", "rank(v2b) - rank(brute-force)",
         "v2b effect on rank at equal budget  (below 0 = v2b better)")
ax.legend()
fig.tight_layout()
save(fig, "table5_v2b_delta.png")

# money plot: efficiency rank vs sims, v2b vs brute-force (Jeffreys fixed) per domain
fig, ax = plt.subplots(figsize=(8, 5.5))
for d in DOMS:
    # brute-force curve across ALL fixed-tries N (Jeffreys), full range
    ax.plot(t4[f"{d}_sims"], t4[f"{d}_rank"], "--s", color=DOM_COLOR[d], lw=1.8, ms=5,
            mfc="white", label=f"{DOM_LABEL[d]} - brute-force")
    ax.plot(t5[f"{d}_sims"], t5[f"{d}_rank"], "-o", color=DOM_COLOR[d], lw=2.2, ms=7,
            label=f"{DOM_LABEL[d]} - v2b")
ax.set_xscale("log")
style_ax(ax, "avg total simulations / diagnosis (log)", "avg real-fault rank (lower better)",
         "Accuracy vs cost: v2b (solid) vs brute-force (dashed)")
ax.legend(fontsize=9, ncol=1)
fig.tight_layout()
save(fig, "table5_efficiency_rank_vs_sims.png")

# ======================================================================
write_plot_provenance(OUT, created, input_sources=[
    os.path.join(TAB, "table1_FL_epsilon_sweep.csv"),
    os.path.join(TAB, "table2_FL_epsilon_sweep_min20_batch10.csv"),
    os.path.join(TAB, "table3_fixedtries_all_domains.csv"),
    os.path.join(TAB, "table4_jeffreys_fixedtries_vs_fixedtries.csv"),
    os.path.join(TAB, "table5_v2b_vs_jeffreys_fixedtries.csv"),
])
print(f"\nDone. {len(created)} figures in {OUT}")
