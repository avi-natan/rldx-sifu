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
        "--fl_way",
        type=int,
        default=0,
        help="FrozenLake fault-benchmark scheme: 2=distinguishable (most-used a*) -- THE ONE WE USE. "
             "1=Taxi-style (rare a*+near-twins) -- RETIRED BASELINE, do not run for real results "
             "(see experimental results/FrozenLake_v1/BENCHMARK_WAYS.md). "
             "0 (default)=run the Taxi hard class-2 experiment."
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
        # --fl_way 1 or 2  -> FrozenLake fault benchmark (known fr, one epsilon per run -> one xlsx)
        # --fl_way 0 (default) -> Taxi-v4 hard class-2 experiment
        if args.fl_way == 1:
            # Way 1 is the RETIRED Taxi-style baseline (see BENCHMARK_WAYS.md). We keep the code
            # for reproducibility but do not use its results. Require an explicit override so a
            # stray "--fl_way 1" can never silently produce results we would report.
            import os
            if os.environ.get("ALLOW_RETIRED_WAY1") != "1":
                raise ValueError(
                    "fl_way=1 is the RETIRED FrozenLake baseline and is not run by default. "
                    "We use way 2 only. To reproduce the baseline anyway, set ALLOW_RETIRED_WAY1=1."
                )

        if args.fl_way in (1, 2):
            multiple_experiment_FrozenLake_fault_benchmark(
                epsilon=args.epsilon,
                way=args.fl_way,
                unknown_fault_rate=args.unknown_fault_rate,
                maps_num=100,
                run_folder=args.run_folder,
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
