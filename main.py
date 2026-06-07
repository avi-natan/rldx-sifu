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
                                  single_experiment_FrozenLake_SIF, multiple_experiments_FrozenLake_SIF,
                                  single_experiment_FrozenLake_NON_DETERMINSTIC,
                                  multiple_experiment_FrozenLake_NON_DETERMINSTIC_FO,
                                  multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO)


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

        # single_experiment_FrozenLake_SIF()

        print("At main")
        print(f"Running with epsilon={args.epsilon}")
        print(f"unknown_fault_rate={args.unknown_fault_rate}")
        print(f"maps_num={args.maps_num}")

        multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO(
            epsilon=args.epsilon,
            unknown_fault_rate=args.unknown_fault_rate,
            maps_num=args.maps_num
        )


        # single_experiment_FrozenLake_NON_DETERMINSTIC()
        # single_experiment_FrozenLake_SIF()
        # multiple_experiments_FrozenLake_SIF()
        print(f'finished gracefully1')
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()

        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)

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
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
    except ValueError as e:
        print(f'Value error: {e}')
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)

    print(9)
