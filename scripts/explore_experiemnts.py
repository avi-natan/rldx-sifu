import os
import re
import glob
import pandas as pd
import numpy as np

try_num = 2
INPUT_DIR = f"C:/Users/ahmad/Downloads/experiments_outputs/several_epsilons_run/try{try_num}/xl_results"


def extract_epsilon_from_filename(path):
    match = re.search(r"epsilon_(\d+_\d+)", os.path.basename(path))
    if match:
        return float(match.group(1).replace("_", "."))
    return None


def safe_mean(values):
    values = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else None


def fmt(v, digits=3):
    if v is None or pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


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


def print_section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def print_columns_from_one_file(xlsx_files):
    if not xlsx_files:
        print("No xlsx files found.")
        return

    path = xlsx_files[0]
    df = pd.read_excel(path)

    print_section("COLUMNS FROM ONE FILE")
    print(path)
    print()

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
        "execution_fault_in_top1",
        "execution_fault_in_top2",
        "execution_fault_in_top3",
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

    print_section("EPSILON SUMMARY")

    header = (
        f"{'eps':>7} "
        f"{'rank':>7} "
        f"{'time(s)':>10} "
        f"{'tries':>9} "
        f"{'tot_tries':>12} "
        f"{'margin':>8} "
        f"{'max_stop':>9} "
        f"{'gaps':>7} "
        f"{'gap_time':>10} "
        f"{'calls':>8} "
        f"{'p_hat':>8} "
        f"{'n':>5}"
    )

    print(header)
    print("-" * len(header))

    for eps in sorted(eps_stats.keys()):
        s = eps_stats[eps]

        print(
            f"{eps:7.4f} "
            f"{fmt(safe_mean(s['real_fault_rank']), 3):>7} "
            f"{fmt(safe_mean(s['diagnosis_time_sec']), 2):>10} "
            f"{fmt(safe_mean(s['adaptive_avg_real_tries']), 0):>9} "
            f"{fmt(safe_mean(s['adaptive_total_real_tries']), 0):>12} "
            f"{fmt(safe_mean(s['adaptive_avg_margin']), 4):>8} "
            f"{fmt(safe_mean(s['adaptive_max_stop_rate']), 3):>9} "
            f"{fmt(safe_mean(s['num_gaps']), 1):>7} "
            f"{fmt(safe_mean(s['avg_gap_time']), 3):>10} "
            f"{fmt(safe_mean(s['adaptive_total_calls']), 1):>8} "
            f"{fmt(safe_mean(s['adaptive_avg_p_hat']), 3):>8} "
            f"{s['n']:5d}"
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

    print_section("TOP-K ACCURACY PER EPSILON")

    header = (
        f"{'eps':>7} "
        f"{'top1':>7} "
        f"{'top2':>7} "
        f"{'top3':>7}"
    )

    print(header)
    print("-" * len(header))

    for eps in sorted(eps_stats.keys()):
        s = eps_stats[eps]

        print(
            f"{eps:7.4f} "
            f"{fmt(safe_mean(s['top1']), 3):>7} "
            f"{fmt(safe_mean(s['top2']), 3):>7} "
            f"{fmt(safe_mean(s['top3']), 3):>7}"
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

    print_section("VISIBILITY × EPSILON")

    header = (
        f"{'vis':>5} "
        f"{'eps':>7} "
        f"{'rank':>7} "
        f"{'time(s)':>10} "
        f"{'tries':>9} "
        f"{'tot_tries':>12} "
        f"{'margin':>8} "
        f"{'max_stop':>9} "
        f"{'n':>5}"
    )

    print(header)
    print("-" * len(header))

    for key in sorted(stats.keys()):
        vis, eps = key
        s = stats[key]
        n = len(s["rank"])

        print(
            f"{vis:5.0f} "
            f"{eps:7.4f} "
            f"{fmt(safe_mean(s['rank']), 3):>7} "
            f"{fmt(safe_mean(s['time']), 2):>10} "
            f"{fmt(safe_mean(s['avg_tries']), 0):>9} "
            f"{fmt(safe_mean(s['total_tries']), 0):>12} "
            f"{fmt(safe_mean(s['margin']), 4):>8} "
            f"{fmt(safe_mean(s['max_stop_rate']), 3):>9} "
            f"{n:5d}"
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

    print_section("FAULT RATE × EPSILON")

    header = (
        f"{'fr':>5} "
        f"{'eps':>7} "
        f"{'rank':>7} "
        f"{'time(s)':>10} "
        f"{'tries':>9} "
        f"{'tot_tries':>12} "
        f"{'margin':>8} "
        f"{'max_stop':>9} "
        f"{'n':>5}"
    )

    print(header)
    print("-" * len(header))

    for key in sorted(stats.keys()):
        fr, eps = key
        s = stats[key]
        n = len(s["rank"])

        print(
            f"{fr:5.1f} "
            f"{eps:7.4f} "
            f"{fmt(safe_mean(s['rank']), 3):>7} "
            f"{fmt(safe_mean(s['time']), 2):>10} "
            f"{fmt(safe_mean(s['avg_tries']), 0):>9} "
            f"{fmt(safe_mean(s['total_tries']), 0):>12} "
            f"{fmt(safe_mean(s['margin']), 4):>8} "
            f"{fmt(safe_mean(s['max_stop_rate']), 3):>9} "
            f"{n:5d}"
        )


if __name__ == "__main__":
    records, xlsx_files = load_all_records(INPUT_DIR)

    print(f"Loaded files: {len(xlsx_files)}")
    print(f"Loaded records: {len(records)}")

    # print_columns_from_one_file(xlsx_files)

    print_avg_statistics_per_epsilon(records)
    # print_top_accuracy_per_epsilon(records)

    print_statistics_per_visibility_and_epsilon(records)
    print_statistics_per_fault_rate_and_epsilon(records)