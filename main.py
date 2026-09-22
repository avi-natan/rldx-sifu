import sys
import time

from pygame import mixer  # Load the popular external library

from p_pipeline import run_experimental_setup, run_experimental_setup_new
from p_single_experiments import (single_experiment_manual, \
                                  single_experiment_LunarLander_W, single_experiment_LunarLander_SN,
                                  single_experiment_LunarLander_SIF, \
                                  single_experiment_Acrobot_W, single_experiment_Acrobot_SN,
                                  single_experiment_Acrobot_SIF, single_experiment_Acrobot_SIFU,
                                  single_experiment_Acrobot_SIFU2, single_experiment_Acrobot_SIFU3,
                                  single_experiment_Acrobot_SIFU4, single_experiment_Acrobot_SIFU5,
                                  single_experiment_Acrobot_SIFU6, single_experiment_Acrobot_SIFU7,
                                  single_experiment_Acrobot_SIFU8, \
                                  single_experiment_CartPole_W, single_experiment_CartPole_SN,
                                  single_experiment_CartPole_SIF, single_experiment_CartPole_SIFU,
                                  single_experiment_CartPole_SIFU2, single_experiment_CartPole_SIFU3,
                                  single_experiment_CartPole_SIFU4, single_experiment_CartPole_SIFU5,
                                  single_experiment_CartPole_SIFU6, single_experiment_CartPole_SIFU7,
                                  single_experiment_CartPole_SIFU8, \
                                  single_experiment_MountainCar_W, single_experiment_MountainCar_SN,
                                  single_experiment_MountainCar_SIF, single_experiment_MountainCar_SIFU,
                                  single_experiment_MountainCar_SIFU2, single_experiment_MountainCar_SIFU3,
                                  single_experiment_MountainCar_SIFU4, single_experiment_MountainCar_SIFU5,
                                  single_experiment_MountainCar_SIFU6, single_experiment_MountainCar_SIFU7,
                                  single_experiment_MountainCar_SIFU8, \
                                  single_experiment_Taxi_W, single_experiment_Taxi_SN, single_experiment_Taxi_SIF,
                                  single_experiment_Taxi_SIFU, single_experiment_Taxi_SIFU2,
                                  single_experiment_Taxi_SIFU3, single_experiment_Taxi_SIFU4,
                                  single_experiment_Taxi_SIFU5, single_experiment_Taxi_SIFU6,
                                  single_experiment_Taxi_SIFU7, single_experiment_Taxi_SIFU8,
                                  single_experiment_FrozenLake_NON_DETERMINSTIC,
                                  multiple_experiment_FrozenLake_NON_DETERMINSTIC_FO,
                                  multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO,
                                  multiple_experiment_Taxi_v4_NON_DETERMINSTIC_PO,
                                  multiple_experiment_Taxi_v4_hard_class2_PO,
                                  multiple_experiment_FrozenLake_fault_benchmark,
                                  multiple_experiment_MiniGrid_fault_benchmark,
                                  single_experiment_stochastic_Taxi_v4, single_experiment_stochastic_FrozenLake)


def play_done_alarm():
    """Play the end-of-run chime, but never let audio problems affect the exit code.
    Headless compute nodes have no audio device, so mixer.init() raises there; we must
    not let that turn a successful experiment into a FAILED job."""
    try:
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
    except Exception as e:
        print(f"(alarm skipped: {e})")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Adaptive MC confidence interval threshold"
    )

    parser.add_argument(
        "-ufr",
        "--unknown_fault_rate",
        action="store_true",
        help="Run PO diagnosis with unknown fault rate estimation"
    )

    parser.add_argument(
        "-n",
        "--maps_num",
        type=int,
        default=49,
        help="Number of map/policy pairs to run"
    )

    parser.add_argument(
        "-o",
        "--run_folder",
        type=str,
        default=None,
        help="Sub-folder under 'experimental results/<domain>/' to collect this run's output "
             "(default: 'general')"
    )

    parser.add_argument(
        "--frozenlake",
        action="store_true",
        help="Run the FrozenLake way-2 fault benchmark (the only way we use; see "
             "experimental results/FrozenLake_v1/BENCHMARK_WAYS.md). "
             "Omit to run the Taxi-v4 hard class-2 experiment instead."
    )

    parser.add_argument(
        "--fl_fault_rates",
        type=float,
        nargs="+",
        default=[0.5, 0.8],
        help="FrozenLake injected fault rate(s) to sweep (default: 0.5 0.8). "
             "Pass '0.3' for the fr=0.3 runs. Encoded into the output filename."
    )

    parser.add_argument(
        "--fl_group",
        type=int,
        default=None,
        help="FrozenLake map-group index (0-based) for job-array splitting. Splits the 100 "
             "maps into --fl_num_groups equal groups; this task runs only its group. "
             "Set to $SLURM_ARRAY_TASK_ID. Omit to run all maps in one process."
    )

    parser.add_argument(
        "--fl_num_groups",
        type=int,
        default=10,
        help="Number of FrozenLake map groups to split into (default: 10 -> 10 maps each)."
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Override the results root folder (e.g. 'bruteforce_unknown_eps_experiments' for the "
             "sims-vs-rank study). When set, xlsx land in <results_root>/<domain>/<run_folder>/xlsx/. "
             "Default: the driver's usual root (experimental results / ufr_experiments)."
    )

    parser.add_argument(
        "--minigrid",
        action="store_true",
        help="Run the MiniGrid partial-observability fault benchmark "
             "instead of Taxi/FrozenLake."
    )

    parser.add_argument(
        "--mg_group",
        type=int,
        default=None,
        help="MiniGrid instance-group index (0-based) for job-array splitting. Splits the "
             "instances into --mg_num_groups equal groups; this task runs only its group. "
             "Set to $SLURM_ARRAY_TASK_ID. Omit to run all instances in one process."
    )

    parser.add_argument(
        "--mg_num_groups",
        type=int,
        default=10,
        help="Number of MiniGrid instance groups to split into (default: 10)."
    )

    parser.add_argument(
        "--mg_noise",
        type=float,
        default=0.7,
        choices=[0.3, 0.5, 0.7],
        help="MiniGrid Empty ONLY: env action-success probability. Selects BOTH the env noise "
             "(SeededStochasticActionWrapper prob) AND the matching trained policy "
             "(models/PPO/..._noise{mg_noise}.zip). 0.7=diagnosable, 0.5=middle, 0.3=very noisy. "
             "Ignored for non-MiniGrid domains."
    )

    parser.add_argument(
        "--mg_fault_rate",
        type=float,
        default=0.5,
        help="MiniGrid ONLY: the single injected fault-firing probability for this run "
             "(the benchmark uses one fault_rate per run; sweep it across runs)."
    )

    parser.add_argument(
        "--mg_domain",
        default="empty",
        choices=["empty", "crossing"],
        help="Which MiniGrid domain to benchmark: 'empty' (MiniGrid-Empty-16x16, fixed layout) or "
             "'crossing' (MiniGrid-SimpleCrossing-S11N2, per-seed wall layouts). Both reuse the same "
             "26-fault benchmark + candidate sets; results land under the domain's own results dir."
    )

    parser.add_argument(
        "--mg_method",
        default="full",
        choices=["full", "racing", "v1", "v1_freeze", "v2"],
        help="unknown-fault-rate diagnoser: 'full' (score every rate to confidence), 'racing' "
             "(paired-difference CRN racing), 'v1' (marginal-CI racing, no freezing), 'v1_freeze' "
             "(marginal-CI racing that also freezes rates that can't be a fault's best), or 'v2' "
             "(conservative: every computed estimate is epsilon-precise like full; saves time only by "
             "freezing rates/faults that provably can't change the rank -> matches full's rank). "
             "Anything but 'full' implies unknown fault rate and routes results to ufr_experiments/."
    )

    parser.add_argument(
        "--racing",
        action="store_true",
        help="Use the confidence-bounded RACING ufr diagnoser for the Taxi hard-class2 experiment "
             "(implies unknown fault rate). Results go to ufr_experiments/ (kept out of the main tree)."
    )

    args = parser.parse_args()

    try:
        # == single experiments (for coding and debug purposes) ==
        # single_experiment_manual()                #

        # single_experiment_LunarLander_W()         #
        # single_experiment_LunarLander_SN()        #
        # single_experiment_LunarLander_SIF()       #

        # single_experiment_Acrobot_W()            # OK ALL
        # single_experiment_Acrobot_SN()           # OK ALL
        # single_experiment_Acrobot_SIF()          # OK ALL
        # single_experiment_Acrobot_SIFU()         # OK ALL


        print("At main")
        print(f"Running with epsilon={args.epsilon}")
        print(f"unknown_fault_rate={args.unknown_fault_rate}")
        print(f"maps_num={args.maps_num}")

        # epsilon comes from --epsilon (one value per run -> one xlsx); no longer hard-set here.
        # unknown_fault_rate comes from -ufr/--unknown_fault_rate (default False); no longer hard-set here.
        args.maps_num = 49


        """        
        multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO(
            epsilon=args.epsilon,
            unknown_fault_rate=args.unknown_fault_rate,
            maps_num=args.maps_num,
            run_folder=args.run_folder
        )
        """

        # multiple_experiment_Taxi_v4_NON_DETERMINSTIC_PO(
        #     epsilon=args.epsilon,
        #     unknown_fault_rate=args.unknown_fault_rate,
        #     num_seeds=5,
        #     run_folder=args.run_folder
        # )

        #single_experiment_stochastic_FrozenLake(run_folder=args.run_folder)
        # single_experiment_stochastic_Taxi_v4(run_folder=args.run_folder)

        # === experiment selection ===
        # --frozenlake   -> FrozenLake way-2 fault benchmark (one epsilon per run -> one xlsx)
        # (default)      -> Taxi-v4 hard class-2 experiment
        if args.frozenlake:
            FL_MAPS_NUM = 100
            # Work-units = maps x 5 visibilities x injected fault rates. --fl_group splits these units
            # (not just maps) so a job array can run ONE diagnosis per task (full parallelism).
            FL_TOTAL_UNITS = FL_MAPS_NUM * 5 * len(args.fl_fault_rates)
            fl_unit_start, fl_unit_end = 0, None
            if args.fl_group is not None:
                gsize = FL_TOTAL_UNITS // args.fl_num_groups
                fl_unit_start = args.fl_group * gsize
                # last group absorbs any remainder from an uneven split
                fl_unit_end = (fl_unit_start + gsize
                               if args.fl_group < args.fl_num_groups - 1 else FL_TOTAL_UNITS)
                print(f"FrozenLake group {args.fl_group}/{args.fl_num_groups} "
                      f"-> units [{fl_unit_start}, {fl_unit_end}) of {FL_TOTAL_UNITS}")
            _fl_variant = None if args.mg_method == "full" else args.mg_method
            multiple_experiment_FrozenLake_fault_benchmark(
                epsilon=args.epsilon,
                unknown_fault_rate=args.unknown_fault_rate,
                fault_rate_list=args.fl_fault_rates,
                maps_num=FL_MAPS_NUM,
                run_folder=args.run_folder,
                unit_start=fl_unit_start,
                unit_end=fl_unit_end,
                ufr_variant=_fl_variant,
                results_root=args.results_root,
            )
        elif args.minigrid:
            # MiniGrid: select env noise + the matching trained policy together, and the domain.
            import h_wrappers
            from p_single_experiments import MINIGRID_FAULTS, MINIGRID_VISIBILITIES
            mg_domain_name = {"empty": "MiniGrid_Empty_16x16_v0",
                              "crossing": "MiniGrid_SimpleCrossing_S11N2_v0"}[args.mg_domain]
            h_wrappers.set_minigrid_action_prob(args.mg_noise)
            MG_NUM_SEEDS = 3
            # ONE run = one (noise, fault_rate); the FIXED benchmark = 26 faults x seeds, with
            # visibility swept inside. Work-units = fault x seed x visibility, split for the array.
            MG_TOTAL_UNITS = len(MINIGRID_FAULTS) * MG_NUM_SEEDS * len(MINIGRID_VISIBILITIES)
            unit_start, unit_end = 0, None
            if args.mg_group is not None:
                group_size = MG_TOTAL_UNITS // args.mg_num_groups
                unit_start = args.mg_group * group_size
                unit_end = (unit_start + group_size
                            if args.mg_group < args.mg_num_groups - 1 else MG_TOTAL_UNITS)
                print(f"MiniGrid group {args.mg_group}/{args.mg_num_groups} "
                      f"-> work-units [{unit_start}, {unit_end}) of {MG_TOTAL_UNITS}")
            _ufr_variant = None if args.mg_method == "full" else args.mg_method
            multiple_experiment_MiniGrid_fault_benchmark(
                epsilon=args.epsilon,
                unknown_fault_rate=args.unknown_fault_rate,
                fault_rate=args.mg_fault_rate,
                num_seeds=MG_NUM_SEEDS,
                run_folder=args.run_folder,
                unit_start=unit_start,
                unit_end=unit_end,
                domain_name=mg_domain_name,
                ufr_variant=_ufr_variant,
            )
        else:
            # Known-rate: default. Unknown-rate: pass -ufr on the CLI (10x more MC sims, ~10x slower).
            TAXI_SEEDS = 100
            TAXI_TOTAL_UNITS = TAXI_SEEDS * 5   # seeds x 5 visibilities (single fault rate)
            taxi_unit_start, taxi_unit_end = 0, None
            if args.mg_group is not None:   # reuse the array-split flags for parallelism
                gsize = TAXI_TOTAL_UNITS // args.mg_num_groups
                taxi_unit_start = args.mg_group * gsize
                taxi_unit_end = (taxi_unit_start + gsize
                                 if args.mg_group < args.mg_num_groups - 1 else TAXI_TOTAL_UNITS)
                print(f"Taxi group {args.mg_group}/{args.mg_num_groups} -> units "
                      f"[{taxi_unit_start}, {taxi_unit_end}) of {TAXI_TOTAL_UNITS}")
            _taxi_variant = None if args.mg_method == "full" else args.mg_method
            multiple_experiment_Taxi_v4_hard_class2_PO(
                epsilon=args.epsilon,
                num_seeds=TAXI_SEEDS,
                run_folder=args.run_folder,
                unknown_fault_rate=args.unknown_fault_rate,
                use_racing=args.racing,
                unit_start=taxi_unit_start,
                unit_end=taxi_unit_end,
                ufr_variant=_taxi_variant,
                results_root=args.results_root,
            )


        # single_experiment_FrozenLake_NON_DETERMINSTIC()
        print(f'finished gracefully1')
        play_done_alarm()

        exit(0)

        # single_experiment_Acrobot_SIFU2()        # OK ALL
        # single_experiment_Acrobot_SIFU3()        # OK ALL
        # single_experiment_Acrobot_SIFU4()        # OK ALL
        # single_experiment_Acrobot_SIFU5()        # OK ALL
        # single_experiment_Acrobot_SIFU6()        # OK ALL
        # single_experiment_Acrobot_SIFU7()        # OK ALL
        # single_experiment_Acrobot_SIFU8()        # OK ALL

        # single_experiment_CartPole_W()           # OK ALL
        # single_experiment_CartPole_SN()          # OK ALL
        # single_experiment_CartPole_SIF()         # OK ALL
        # single_experiment_CartPole_SIFU()        # OK ALL
        # single_experiment_CartPole_SIFU2()       # OK ALL
        # single_experiment_CartPole_SIFU3()       # OK ALL
        # single_experiment_CartPole_SIFU4()       # OK ALL
        # single_experiment_CartPole_SIFU5()       # OK ALL
        # single_experiment_CartPole_SIFU6()       # OK ALL
        # single_experiment_CartPole_SIFU7()       # OK ALL
        # single_experiment_CartPole_SIFU8()       # OK ALL

        # single_experiment_MountainCar_W()        # OK ALL
        # single_experiment_MountainCar_SN()       # OK ALL
        # single_experiment_MountainCar_SIF()      # OK ALL
        # single_experiment_MountainCar_SIFU()     # OK ALL
        # single_experiment_MountainCar_SIFU2()    # OK ALL
        # single_experiment_MountainCar_SIFU3()    # OK ALL
        # single_experiment_MountainCar_SIFU4()    # OK ALL
        # single_experiment_MountainCar_SIFU5()    # OK ALL
        # single_experiment_MountainCar_SIFU6()    # OK ALL
        # single_experiment_MountainCar_SIFU7()    # OK ALL
        # single_experiment_MountainCar_SIFU8()    # OK ALL

        # single_experiment_Taxi_W()               # OK ALL
        # single_experiment_Taxi_SN()              # OK ALL
        # single_experiment_Taxi_SIF()             # OK ALL
        # single_experiment_Taxi_SIFU()            # OK ALL
        # single_experiment_Taxi_SIFU2()           # OK ALL
        # single_experiment_Taxi_SIFU3()           # OK ALL
        # single_experiment_Taxi_SIFU4()           # OK ALL
        # single_experiment_Taxi_SIFU5()           # OK ALL
        # single_experiment_Taxi_SIFU6()           # OK ALL
        # single_experiment_Taxi_SIFU7()           # OK ALL
        # single_experiment_Taxi_SIFU8()           # OK ALL

        # ================== experimental setup ==================
        render_mode = "rgb_array"       # "human", "rgb_array"
        debug_print = False             # False, True
        # run_experimental_setup(arguments=sys.argv, render_mode=render_mode, debug_print=debug_print)
        run_experimental_setup_new(arguments=sys.argv, render_mode=render_mode, debug_print=debug_print)

        print(f'finisehd gracefully')
        play_done_alarm()
    except ValueError as e:
        print(f'Value error: {e}')
        play_done_alarm()

    print(9)
