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
            # --fl_group present -> run only this task's slice of the 100 maps (job-array split).
            map_start, map_end = 0, None
            if args.fl_group is not None:
                group_size = FL_MAPS_NUM // args.fl_num_groups
                map_start = args.fl_group * group_size
                # last group absorbs any remainder from an uneven split
                map_end = (map_start + group_size
                           if args.fl_group < args.fl_num_groups - 1 else FL_MAPS_NUM)
                print(f"FrozenLake group {args.fl_group}/{args.fl_num_groups} "
                      f"-> maps [{map_start}, {map_end})")
            multiple_experiment_FrozenLake_fault_benchmark(
                epsilon=args.epsilon,
                unknown_fault_rate=args.unknown_fault_rate,
                fault_rate_list=args.fl_fault_rates,
                maps_num=FL_MAPS_NUM,
                run_folder=args.run_folder,
                map_start=map_start,
                map_end=map_end,
            )
        elif args.minigrid:
            # MiniGrid Empty ONLY: select env noise + the matching trained policy together.
            import h_wrappers
            h_wrappers.set_minigrid_action_prob(args.mg_noise)
            MG_INSTANCES = 100
            # Work units = instances x 5 visibilities x 3 fault rates (one diagnosis each,
            # ~1 min). --mg_group present -> run only this task's contiguous slice of them.
            MG_TOTAL_UNITS = MG_INSTANCES * 5 * 3
            unit_start, unit_end = 0, None
            if args.mg_group is not None:
                group_size = MG_TOTAL_UNITS // args.mg_num_groups
                unit_start = args.mg_group * group_size
                unit_end = (unit_start + group_size
                            if args.mg_group < args.mg_num_groups - 1 else MG_TOTAL_UNITS)
                print(f"MiniGrid group {args.mg_group}/{args.mg_num_groups} "
                      f"-> work-units [{unit_start}, {unit_end}) of {MG_TOTAL_UNITS}")
            multiple_experiment_MiniGrid_fault_benchmark(
                epsilon=args.epsilon,
                unknown_fault_rate=args.unknown_fault_rate,
                num_instances=MG_INSTANCES,
                run_folder=args.run_folder,
                unit_start=unit_start,
                unit_end=unit_end,
            )
        else:
            # Known-rate: default. Unknown-rate: pass -ufr on the CLI (10x more MC sims, ~10x slower).
            multiple_experiment_Taxi_v4_hard_class2_PO(
                epsilon=args.epsilon,
                num_seeds=100,
                run_folder=args.run_folder,
                unknown_fault_rate=args.unknown_fault_rate,
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
