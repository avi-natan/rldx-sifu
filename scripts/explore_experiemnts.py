import os
import re
import glob
import pandas as pd
import numpy as np


INPUT_DIR = "C:/Users/ahmad/Downloads/xl_results"


def extract_epsilon_from_filename(path):
    match = re.search(r"epsilon_(\d+_\d+)", os.path.basename(path))
    if match:
        return float(match.group(1).replace("_", "."))
    return None


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


def print_avg_rank_per_epsilon(records):
    eps_to_ranks = {}

    for r in records:
        eps = r.get("epsilon")
        rank = r.get("real_fault_rank")

        if eps is None or rank is None or pd.isna(rank):
            continue

        eps_to_ranks.setdefault(float(eps), []).append(float(rank))

    print("\nAverage real fault rank per epsilon")
    print("-----------------------------------")
    print("epsilon\tavg_rank\tn")

    for eps in sorted(eps_to_ranks.keys()):
        ranks = eps_to_ranks[eps]
        avg_rank = float(np.mean(ranks))
        n = len(ranks)

        print(f"{eps}\t{avg_rank:.6f}\t{n}")


def print_avg_rank_per_epsilon_with_runtime(records):
    eps_to_ranks = {}
    eps_to_times = {}

    for r in records:
        eps = r.get("epsilon")
        rank = r.get("real_fault_rank")
        t = r.get("diagnosis_time_sec")

        if eps is None or pd.isna(eps):
            continue

        if rank is not None and not pd.isna(rank):
            eps_to_ranks.setdefault(float(eps), []).append(float(rank))

        if t is not None and not pd.isna(t):
            eps_to_times.setdefault(float(eps), []).append(float(t))

    print("\nAverage rank and runtime per epsilon")
    print("------------------------------------")
    print("epsilon\tavg_rank\tavg_time_sec\tn_rank\tn_time")

    all_eps = sorted(set(eps_to_ranks.keys()) | set(eps_to_times.keys()))

    for eps in all_eps:
        ranks = eps_to_ranks.get(eps, [])
        times = eps_to_times.get(eps, [])

        avg_rank = float(np.mean(ranks)) if ranks else None
        avg_time = float(np.mean(times)) if times else None

        avg_rank_str = f"{avg_rank:.6f}" if avg_rank is not None else "NA"
        avg_time_str = f"{avg_time:.6f}" if avg_time is not None else "NA"

        print(f"{eps}\t{avg_rank_str}\t{avg_time_str}\t{len(ranks)}\t{len(times)}")


if __name__ == "__main__":
    records, xlsx_files = load_all_records(INPUT_DIR)

    print(f"Loaded files: {len(xlsx_files)}")
    print(f"Loaded records: {len(records)}")

    print_columns_from_one_file(xlsx_files)
    print_avg_rank_per_epsilon(records)
    print_avg_rank_per_epsilon_with_runtime(records)