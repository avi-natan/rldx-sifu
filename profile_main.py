import cProfile
import pstats




from p_single_experiments import multiple_experiment_FrozenLake_NON_DETERMINSTIC_PO, \
    single_experiment_stochastic_FrozenLake, single_experiment_stochastic_Taxi_v4

if __name__ == "__main__":

    profiler = cProfile.Profile()

    profiler.enable()

    single_experiment_stochastic_FrozenLake()
    #single_experiment_stochastic_Taxi_v4()

    profiler.disable()

    profiler.dump_stats("profile_output.prof")

    stats = pstats.Stats("profile_output.prof")

    stats.sort_stats("cumtime")

    stats.print_stats(40)