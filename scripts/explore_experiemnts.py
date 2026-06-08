import os
import re
import glob
import pandas as pd
import numpy as np


INPUT_DIR = "C:/Users/ahmad/Downloads/experiments_outputs/several_epsilons_run/try2/xl_results"


def extract_epsilon_from_filename(path):
    match = re.search(r"epsilon_(\d+_\d+)", os.path.basename(path))
    if match:
        return float(match.group(1).replace("_", "."))
    return None


def safe_mean(values):
    values = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else None


def load_all_records(input_dir):
    records = []
    xlsx_files = glob.glob(os.path.join(input_dir, "*.xlsx"))

    for path in xlsx_files:
        epsilon = extract_epsilon_from_filename(path)

        df = pd.read_excel(path)
        curr_records = df.to_dict(orient="records")

        for r in curr_records:
            r["epsilon"] = epsilon
            r["source_file"] = os.path.basename(path)

        records.extend(curr_records)

    return records, xlsx_files


def print_columns_from_one_file(xlsx_files):
    if not xlsx_files:
        print("No xlsx files found.")
        return

    path = xlsx_files[0]
    df = pd.read_excel(path)

    print("\nColumns from file:")
    print(path)
    print("--------------------------------")

    for col in df.columns:
        print(col)


def print_avg_statistics_per_epsilon(records):
    eps_stats = {}

    columns_to_collect = [
        "real_fault_rank",
        "diagnosis_time_sec",
        "adaptive_avg_real_tries",
        "adaptive_total_real_tries",
        "adaptive_max_stop_rate",
        "adaptive_avg_margin",
        "adaptive_total_calls",
        "adaptive_max_stops",
        "adaptive_conf_stops",
        "adaptive_avg_p_hat",
        "num_gaps",
        "avg_gap_time",
    ]

    for r in records:
        eps = r.get("epsilon")

        if eps is None or pd.isna(eps):
            continue

        eps = float(eps)

        if eps not in eps_stats:
            eps_stats[eps] = {col: [] for col in columns_to_collect}
            eps_stats[eps]["n"] = 0

        eps_stats[eps]["n"] += 1

        for col in columns_to_collect:
            v = r.get(col)

            if v is not None and not pd.isna(v):
                eps_stats[eps][col].append(float(v))

    print("\nAverage statistics per epsilon")
    print("-" * 180)

    print(
        "epsilon\t"
        "n\t"
        "avg_rank\t"
        "avg_time_sec\t"
        "avg_real_tries\t"
        "avg_total_real_tries\t"
        "avg_max_stop_rate\t"
        "avg_margin\t"
        "avg_total_calls\t"
        "avg_max_stops\t"
        "avg_conf_stops\t"
        "avg_p_hat\t"
        "avg_num_gaps\t"
        "avg_gap_time"
    )

    for eps in sorted(eps_stats.keys()):
        s = eps_stats[eps]

        avg_rank = safe_mean(s["real_fault_rank"])
        avg_time = safe_mean(s["diagnosis_time_sec"])
        avg_real_tries = safe_mean(s["adaptive_avg_real_tries"])
        avg_total_real_tries = safe_mean(s["adaptive_total_real_tries"])
        avg_max_stop_rate = safe_mean(s["adaptive_max_stop_rate"])
        avg_margin = safe_mean(s["adaptive_avg_margin"])
        avg_total_calls = safe_mean(s["adaptive_total_calls"])
        avg_max_stops = safe_mean(s["adaptive_max_stops"])
        avg_conf_stops = safe_mean(s["adaptive_conf_stops"])
        avg_p_hat = safe_mean(s["adaptive_avg_p_hat"])
        avg_num_gaps = safe_mean(s["num_gaps"])
        avg_gap_time = safe_mean(s["avg_gap_time"])

        print(
            f"{eps}\t"
            f"{s['n']}\t"
            f"{avg_rank:.6f}\t"
            f"{avg_time:.6f}\t"
            f"{avg_real_tries:.2f}\t"
            f"{avg_total_real_tries:.2f}\t"
            f"{avg_max_stop_rate:.6f}\t"
            f"{avg_margin:.6f}\t"
            f"{avg_total_calls:.2f}\t"
            f"{avg_max_stops:.2f}\t"
            f"{avg_conf_stops:.2f}\t"
            f"{avg_p_hat:.6f}\t"
            f"{avg_num_gaps:.2f}\t"
            f"{avg_gap_time:.6f}"
        )


def print_top_accuracy_per_epsilon(records):
    eps_stats = {}

    for r in records:
        eps = r.get("epsilon")

        if eps is None or pd.isna(eps):
            continue

        eps = float(eps)

        eps_stats.setdefault(eps, {
            "top1": [],
            "top2": [],
            "top3": [],
        })

        if pd.notna(r.get("execution_fault_in_top1")):
            eps_stats[eps]["top1"].append(float(r["execution_fault_in_top1"]))

        if pd.notna(r.get("execution_fault_in_top2")):
            eps_stats[eps]["top2"].append(float(r["execution_fault_in_top2"]))

        if pd.notna(r.get("execution_fault_in_top3")):
            eps_stats[eps]["top3"].append(float(r["execution_fault_in_top3"]))

    print("\nTop-k accuracy per epsilon")
    print("-" * 80)

    print(
        "epsilon\t"
        "top1_acc\t"
        "top2_acc\t"
        "top3_acc"
    )

    for eps in sorted(eps_stats.keys()):
        s = eps_stats[eps]

        top1 = safe_mean(s["top1"])
        top2 = safe_mean(s["top2"])
        top3 = safe_mean(s["top3"])

        print(
            f"{eps}\t"
            f"{top1:.6f}\t"
            f"{top2:.6f}\t"
            f"{top3:.6f}"
        )


def print_statistics_per_visibility_and_epsilon(records):
    stats = {}

    for r in records:
        eps = r.get("epsilon")
        vis = r.get("percent_visible_states")

        if eps is None or vis is None or pd.isna(eps) or pd.isna(vis):
            continue

        eps = float(eps)
        vis = float(vis)

        key = (vis, eps)

        stats.setdefault(key, {
            "rank": [],
            "time": [],
            "total_tries": [],
            "avg_tries": [],
            "max_stop_rate": [],
            "margin": [],
        })

        if pd.notna(r.get("real_fault_rank")):
            stats[key]["rank"].append(float(r["real_fault_rank"]))

        if pd.notna(r.get("diagnosis_time_sec")):
            stats[key]["time"].append(float(r["diagnosis_time_sec"]))

        if pd.notna(r.get("adaptive_total_real_tries")):
            stats[key]["total_tries"].append(float(r["adaptive_total_real_tries"]))

        if pd.notna(r.get("adaptive_avg_real_tries")):
            stats[key]["avg_tries"].append(float(r["adaptive_avg_real_tries"]))

        if pd.notna(r.get("adaptive_max_stop_rate")):
            stats[key]["max_stop_rate"].append(float(r["adaptive_max_stop_rate"]))

        if pd.notna(r.get("adaptive_avg_margin")):
            stats[key]["margin"].append(float(r["adaptive_avg_margin"]))

    print("\nStatistics per visibility and epsilon")
    print("-" * 140)

    print(
        "visibility\t"
        "epsilon\t"
        "n\t"
        "avg_rank\t"
        "avg_time_sec\t"
        "avg_total_real_tries\t"
        "avg_real_tries\t"
        "avg_max_stop_rate\t"
        "avg_margin"
    )

    for key in sorted(stats.keys()):
        vis, eps = key
        s = stats[key]

        n = len(s["rank"])

        print(
            f"{vis}\t"
            f"{eps}\t"
            f"{n}\t"
            f"{safe_mean(s['rank']):.6f}\t"
            f"{safe_mean(s['time']):.6f}\t"
            f"{safe_mean(s['total_tries']):.2f}\t"
            f"{safe_mean(s['avg_tries']):.2f}\t"
            f"{safe_mean(s['max_stop_rate']):.6f}\t"
            f"{safe_mean(s['margin']):.6f}"
        )


def print_statistics_per_fault_rate_and_epsilon(records):
    stats = {}

    for r in records:
        eps = r.get("epsilon")
        fr = r.get("real_fault_prob")

        if eps is None or fr is None or pd.isna(eps) or pd.isna(fr):
            continue

        eps = float(eps)
        fr = float(fr)

        key = (fr, eps)

        stats.setdefault(key, {
            "rank": [],
            "time": [],
            "total_tries": [],
            "avg_tries": [],
            "max_stop_rate": [],
            "margin": [],
        })

        if pd.notna(r.get("real_fault_rank")):
            stats[key]["rank"].append(float(r["real_fault_rank"]))

        if pd.notna(r.get("diagnosis_time_sec")):
            stats[key]["time"].append(float(r["diagnosis_time_sec"]))

        if pd.notna(r.get("adaptive_total_real_tries")):
            stats[key]["total_tries"].append(float(r["adaptive_total_real_tries"]))

        if pd.notna(r.get("adaptive_avg_real_tries")):
            stats[key]["avg_tries"].append(float(r["adaptive_avg_real_tries"]))

        if pd.notna(r.get("adaptive_max_stop_rate")):
            stats[key]["max_stop_rate"].append(float(r["adaptive_max_stop_rate"]))

        if pd.notna(r.get("adaptive_avg_margin")):
            stats[key]["margin"].append(float(r["adaptive_avg_margin"]))

    print("\nStatistics per fault rate and epsilon")
    print("-" * 140)

    print(
        "fault_rate\t"
        "epsilon\t"
        "n\t"
        "avg_rank\t"
        "avg_time_sec\t"
        "avg_total_real_tries\t"
        "avg_real_tries\t"
        "avg_max_stop_rate\t"
        "avg_margin"
    )

    for key in sorted(stats.keys()):
        fr, eps = key
        s = stats[key]

        n = len(s["rank"])

        print(
            f"{fr}\t"
            f"{eps}\t"
            f"{n}\t"
            f"{safe_mean(s['rank']):.6f}\t"
            f"{safe_mean(s['time']):.6f}\t"
            f"{safe_mean(s['total_tries']):.2f}\t"
            f"{safe_mean(s['avg_tries']):.2f}\t"
            f"{safe_mean(s['max_stop_rate']):.6f}\t"
            f"{safe_mean(s['margin']):.6f}"
        )


if __name__ == "__main__":
    records, xlsx_files = load_all_records(INPUT_DIR)

    print(f"Loaded files: {len(xlsx_files)}")
    print(f"Loaded records: {len(records)}")

    print_columns_from_one_file(xlsx_files)

    print_avg_statistics_per_epsilon(records)
    print_top_accuracy_per_epsilon(records)

    print_statistics_per_visibility_and_epsilon(records)
    print_statistics_per_fault_rate_and_epsilon(records)