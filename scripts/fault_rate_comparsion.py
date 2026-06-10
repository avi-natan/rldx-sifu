import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER INPUTS
# =========================

PATHH = "C:/Users/ahmad/Downloads/experiments_outputs/UNknow_vs_known"
KNOWN_FILE = f"{PATHH}/frozen_lake_non_deterministic_PO_known_fault_rate_epsilon_0_03.xlsx"
UNKNOWN_FILE = f"{PATHH}/frozen_lake_non_deterministic_PO_UN_known_fault_rate_epsilon_0_03.xlsx"

OUTPUT_DIR = "known_vs_unknown_comparison_plots1"


# =========================
# HELPERS
# =========================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_current_plot(filename):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def safe_mean(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.mean(values)) if len(values) > 0 else None


def safe_sum(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.sum(values)) if len(values) > 0 else None


def fmt(v, digits=3):
    if v is None or pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_runs():
    known_df = pd.read_excel(KNOWN_FILE)
    unknown_df = pd.read_excel(UNKNOWN_FILE)

    known_df["method"] = "Known fault rate"
    unknown_df["method"] = "Unknown fault rate"

    df = pd.concat([known_df, unknown_df], ignore_index=True)

    return known_df, unknown_df, df


# =========================
# PRINT TABLES
# =========================

def print_overall_summary(df):
    print_section("OVERALL SUMMARY")

    rows = []

    for method, g in df.groupby("method"):
        rows.append({
            "method": method,
            "n": len(g),
            "avg_rank": safe_mean(g["real_fault_rank"]),
            "avg_runtime_sec": safe_mean(g["diagnosis_time_sec"]),
            "total_runtime_sec": safe_sum(g["diagnosis_time_sec"]),
        })

    out = pd.DataFrame(rows)

    print(out.to_string(index=False))

    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, "overall_summary.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved table: {path}")

    if len(out) == 2:
        known_time = out.loc[out["method"] == "Known fault rate", "avg_runtime_sec"].values
        unknown_time = out.loc[out["method"] == "Unknown fault rate", "avg_runtime_sec"].values

        known_rank = out.loc[out["method"] == "Known fault rate", "avg_rank"].values
        unknown_rank = out.loc[out["method"] == "Unknown fault rate", "avg_rank"].values

        if len(known_time) > 0 and len(unknown_time) > 0 and known_time[0] != 0:
            print()
            print(f"Runtime ratio Unknown/Known: {unknown_time[0] / known_time[0]:.3f}x")

        if len(known_rank) > 0 and len(unknown_rank) > 0:
            print(f"Rank difference Unknown-Known: {unknown_rank[0] - known_rank[0]:.3f}")


def print_summary_by_visibility(df):
    print_section("SUMMARY BY VISIBILITY")

    rows = []

    for (vis, method), g in df.groupby(["percent_visible_states", "method"]):
        rows.append({
            "visibility": vis,
            "method": method,
            "n": len(g),
            "avg_rank": safe_mean(g["real_fault_rank"]),
            "avg_runtime_sec": safe_mean(g["diagnosis_time_sec"]),
        })

    out = pd.DataFrame(rows).sort_values(["visibility", "method"])

    print(out.to_string(index=False))

    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, "summary_by_visibility.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved table: {path}")


def print_summary_by_fault_rate(df):
    print_section("SUMMARY BY REAL FAULT RATE")

    rows = []

    for (fr, method), g in df.groupby(["real_fault_prob", "method"]):
        rows.append({
            "real_fault_rate": fr,
            "method": method,
            "n": len(g),
            "avg_rank": safe_mean(g["real_fault_rank"]),
            "avg_runtime_sec": safe_mean(g["diagnosis_time_sec"]),
        })

    out = pd.DataFrame(rows).sort_values(["real_fault_rate", "method"])

    print(out.to_string(index=False))

    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, "summary_by_fault_rate.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved table: {path}")


# =========================
# PLOTS
# =========================

def plot_overall_avg_rank(df):
    means = df.groupby("method")["real_fault_rank"].mean()

    plt.figure()
    means.plot(kind="bar")
    plt.ylabel("Average real fault rank")
    plt.title("Known vs unknown fault rate: average rank")
    plt.grid(axis="y")

    for i, v in enumerate(means.values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    save_current_plot("overall_avg_rank.png")


def plot_overall_avg_runtime(df):
    means = df.groupby("method")["diagnosis_time_sec"].mean()

    plt.figure()
    means.plot(kind="bar")
    plt.ylabel("Average diagnosis runtime (sec)")
    plt.title("Known vs unknown fault rate: average runtime")
    plt.grid(axis="y")

    for i, v in enumerate(means.values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    save_current_plot("overall_avg_runtime.png")


def plot_avg_rank_vs_visibility(df):
    grouped = (
        df.groupby(["percent_visible_states", "method"])["real_fault_rank"]
        .mean()
        .reset_index()
    )

    plt.figure()

    for method in grouped["method"].unique():
        curr = grouped[grouped["method"] == method].sort_values("percent_visible_states")
        plt.plot(
            curr["percent_visible_states"],
            curr["real_fault_rank"],
            "o-",
            label=method
        )

    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average real fault rank")
    plt.title("Visibility vs average rank")
    plt.xticks(sorted(df["percent_visible_states"].dropna().unique()))
    plt.legend()
    plt.grid(True)

    save_current_plot("visibility_vs_avg_rank.png")


def plot_avg_runtime_vs_visibility(df):
    grouped = (
        df.groupby(["percent_visible_states", "method"])["diagnosis_time_sec"]
        .mean()
        .reset_index()
    )

    plt.figure()

    for method in grouped["method"].unique():
        curr = grouped[grouped["method"] == method].sort_values("percent_visible_states")
        plt.plot(
            curr["percent_visible_states"],
            curr["diagnosis_time_sec"],
            "o-",
            label=method
        )

    plt.xlabel("Visibility rate (%)")
    plt.ylabel("Average diagnosis runtime (sec)")
    plt.title("Visibility vs average runtime")
    plt.xticks(sorted(df["percent_visible_states"].dropna().unique()))
    plt.legend()
    plt.grid(True)

    save_current_plot("visibility_vs_avg_runtime.png")


def plot_avg_rank_vs_fault_rate(df):
    grouped = (
        df.groupby(["real_fault_prob", "method"])["real_fault_rank"]
        .mean()
        .reset_index()
    )

    plt.figure()

    for method in grouped["method"].unique():
        curr = grouped[grouped["method"] == method].sort_values("real_fault_prob")
        plt.plot(
            curr["real_fault_prob"],
            curr["real_fault_rank"],
            "o-",
            label=method
        )

    plt.xlabel("Real fault rate")
    plt.ylabel("Average real fault rank")
    plt.title("Real fault rate vs average rank")
    plt.legend()
    plt.grid(True)

    save_current_plot("fault_rate_vs_avg_rank.png")


def plot_avg_runtime_vs_fault_rate(df):
    grouped = (
        df.groupby(["real_fault_prob", "method"])["diagnosis_time_sec"]
        .mean()
        .reset_index()
    )

    plt.figure()

    for method in grouped["method"].unique():
        curr = grouped[grouped["method"] == method].sort_values("real_fault_prob")
        plt.plot(
            curr["real_fault_prob"],
            curr["diagnosis_time_sec"],
            "o-",
            label=method
        )

    plt.xlabel("Real fault rate")
    plt.ylabel("Average diagnosis runtime (sec)")
    plt.title("Real fault rate vs average runtime")
    plt.legend()
    plt.grid(True)

    save_current_plot("fault_rate_vs_avg_runtime.png")


def create_all_plots(df):
    plot_overall_avg_rank(df)
    plot_overall_avg_runtime(df)

    plot_avg_rank_vs_visibility(df)
    plot_avg_runtime_vs_visibility(df)

    plot_avg_rank_vs_fault_rate(df)
    plot_avg_runtime_vs_fault_rate(df)


# =========================
# MAIN
# =========================

def main():
    known_df, unknown_df, df = load_runs()

    print(f"Known records: {len(known_df)}")
    print(f"Unknown records: {len(unknown_df)}")
    print(f"Total records: {len(df)}")

    print_overall_summary(df)
    print_summary_by_visibility(df)
    print_summary_by_fault_rate(df)

    create_all_plots(df)

    print()
    print("Done.")


if __name__ == "__main__":
    main()