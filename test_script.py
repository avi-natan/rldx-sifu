import os
import re
import glob


LOG_DIR = "C:/Users/ahmad/Downloads/xl_run1_results/xl_outputs"


EPSILON_PATTERN = re.compile(r"epsilon1\s*=\s*([0-9.]+)")
MAX_HIT_PATTERN = "MAX HIT |"


def analyze_log_file(path):
    epsilon_values = set()
    max_hit_count = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPSILON_PATTERN.search(line)
            if match:
                epsilon_values.add(float(match.group(1)))

            if MAX_HIT_PATTERN in line:
                max_hit_count += 1

    return {
        "file": os.path.basename(path),
        "epsilons": sorted(epsilon_values),
        "has_max_hit": max_hit_count > 0,
        "max_hit_count": max_hit_count,
    }


def main():
    files = []
    files.extend(glob.glob(os.path.join(LOG_DIR, "*.txt")))
    files.extend(glob.glob(os.path.join(LOG_DIR, "*.out")))
    files.extend(glob.glob(os.path.join(LOG_DIR, "*.log")))

    results = []

    for path in sorted(files):
        results.append(analyze_log_file(path))

    print(f"Checked {len(results)} files\n")

    for r in results:
        print(f"File: {r['file']}")
        print(f"  epsilon(s): {r['epsilons']}")
        print(f"  includes MAX HIT: {r['has_max_hit']}")
        print(f"  MAX HIT count: {r['max_hit_count']}")
        print()


if __name__ == "__main__":
    main()