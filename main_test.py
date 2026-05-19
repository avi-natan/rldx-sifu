import argparse
import time
from p_pipeline import exper_write_records_to_excel_ind


def fake_experiment(epsilon):
    records = []

    # print 2000 lines
    for i in range(2000):
        print(f"[epsilon={epsilon}] line {i + 1}/2000")
        time.sleep(0.01)

    # create some fake records
    for i in range(10):
        records.append({
            "experiment_num": i + 1,
            "epsilon": epsilon,
            "real_fault_rank": i % 3 + 1,
            "diagnosis_runtime_ms": 1000 + i * 10
        })

    file_suffix = str(epsilon).replace(".", "_")

    out_path = exper_write_records_to_excel_ind(
        records,
        f"frozen_lake_non_deterministic_PO_epsilon_{file_suffix}"
    )

    print(f"\nSaved results to:\n{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="epsilon value"
    )

    args = parser.parse_args()

    fake_experiment(args.epsilon)