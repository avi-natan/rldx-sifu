"""Run the hard, curated Taxi-v4 diagnosis benchmark.

Each instance is a frozen tuple (seed, execution_fault E, [10 graded candidates; E first])
from hard_taxi_data.BENCHMARK. Unlike the random-candidate drivers, this feeds the SAME
difficulty-graded candidate set into the diagnoser for every (visibility, fault_rate) cell,
so a rank difference reflects evidence/epsilon, not candidate luck.

Usage:
    python run_hard_taxi_benchmark.py [num_seeds] [epsilon]

Defaults: num_seeds=3, epsilon=0.03. Writes one .xlsx under
'experimental results/Taxi_v4/hard_benchmark/'.
"""
import sys
from datetime import datetime, timedelta

from h_consts import SEED_BLOCK
from hard_taxi_data import BENCHMARK
from p_pipeline import (
    run_NON_DETERMINSTIC_single_experiment_PO,
    domain_results_dir,
    exper_write_records_to_excel_ind,
)


def run_hard_taxi_benchmark(num_seeds=3, epsilon=0.03, unknown_fault_rate=False,
                            run_folder="hard_benchmark", debug_print=False, seeds=None):
    domain_name = "Taxi_v4"
    ml_model_name = "PPO"
    render_mode = "rgb_array"
    max_exec_len = 200
    num_candidate_fault_modes = 10  # informational; fixed list overrides candidate generation

    fault_rate_candidates = [0.1, 0.3, 0.5, 0.7, 0.8] if unknown_fault_rate else None

    fault_rate_list = [0.5, 0.8]
    percent_visible_states_list = [20, 40, 60, 80, 100]

    # seeds=None -> first num_seeds benchmark instances; else an explicit seed list.
    if seeds is None:
        instances = BENCHMARK[:num_seeds]
    else:
        from hard_taxi_data import get_instance
        instances = [(s, *get_instance(s)) for s in seeds]
    print(f"Running HARD Taxi-v4 benchmark | {len(instances)} seeds | epsilon={epsilon} | "
          f"unknown_fault_rate={unknown_fault_rate}\n")

    records = []
    skipped = 0

    for i, (seed, execution_fault_mode_name, candidate_fault_modes) in enumerate(instances):
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'================= {dt}: SEED {seed} ({i+1}/{len(instances)}), '
              f'E={execution_fault_mode_name} START =================')

        for percent_visible_states in percent_visible_states_list:
            for fault_rate in fault_rate_list:
                print(f'===== SEED {seed} | FR={fault_rate} | VR={percent_visible_states} =====')

                output = run_NON_DETERMINSTIC_single_experiment_PO(
                    domain_name=domain_name,
                    ml_model_name=ml_model_name,
                    render_mode=render_mode,
                    max_exec_len=max_exec_len,
                    debug_print=debug_print,
                    execution_fault_mode_name=execution_fault_mode_name,
                    instance_seed=seed * SEED_BLOCK,  # pass the block base; run_PO derives all offsets
                    fault_probability=fault_rate,
                    percent_visible_states=percent_visible_states,
                    possible_fault_mode_names=candidate_fault_modes,
                    num_candidate_fault_modes=num_candidate_fault_modes,
                    epsilon=epsilon,
                    unknown_fault_rate=unknown_fault_rate,
                    fault_rate_candidates=fault_rate_candidates,
                    fixed_candidate_fault_modes=candidate_fault_modes)

                if not output:
                    skipped += 1
                    print(f"  -> FAILED (no valid trajectory) seed={seed} FR={fault_rate} VR={percent_visible_states}")
                    continue

                output["epsilon"] = epsilon
                output["experiment_num"] = i + 1
                output["real_fault_prob"] = fault_rate
                output["map_desc"] = f"seed_{seed}"
                output["hardcoded_policy"] = f"{domain_name}_{ml_model_name}"
                output["domain_name"] = domain_name
                output["benchmark"] = "hard_taxi"
                output["num_candidates"] = len(candidate_fault_modes)

                print(f"  -> real_fault_rank = {output.get('real_fault_rank')}")
                records.append(output)

        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'================= {dt}: SEED {seed} END =================')

    if not records:
        print("No successful hard-benchmark experiments produced (all trajectories failed).")
        return

    total = sum(r["diagnosis_time_sec"] for r in records)
    print(f"\nRuns: {len(records)} (skipped {skipped})")
    print(f"Total diagnosis time: {timedelta(seconds=int(total))} ({total:.2f} sec)")

    file_suffix = str(epsilon).replace(".", "_")
    method_suffix = "UN_known_fr" if unknown_fault_rate else "known_fr"
    file_path = f"hard_taxi_PO_{method_suffix}_epsilon_{file_suffix}_SEEDS_{len(instances)}"

    output_dir = domain_results_dir(domain_name, run_folder)
    exper_write_records_to_excel_ind(records, file_path, output_dir=output_dir)
    print(f"file was written at: {output_dir}/{file_path}.xlsx")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    dbg = (sys.argv[3].lower() in ("1", "true", "debug")) if len(sys.argv) > 3 else False
    run_hard_taxi_benchmark(num_seeds=n, epsilon=eps, debug_print=dbg)
