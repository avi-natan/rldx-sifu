import gym
import gymnasium
import numpy

class AcrobotSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)

        raw_state = self.unwrapped.state

        # raw_state_as_state = numpy.array(
        #     [cos(raw_state[0]), sin(raw_state[0]), cos(raw_state[1]), sin(raw_state[1]), raw_state[2], raw_state[3]], dtype=numpy.float32
        # )
        # a = numpy.array_equal(state, raw_state_as_state)
        # TODO implement state conversion for all wrappers. for exeisting ones it will be identity
        # todo raw and refined states. the datastructures such as obs will hold raw states. consider changing some set state methods

        return raw_state, info

    def set_state(self, raw_state):
        self.unwrapped.state = raw_state

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        raw_state = self.unwrapped.state
        return raw_state, reward, done, trunc, info


class CartPoleSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)

        raw_state = self.unwrapped.state

        # refined_state = numpy.array(raw_state, dtype=numpy.float32)
        # a = numpy.array_equal(state, refined_state)

        return raw_state, info

    def set_state(self, raw_state):
        self.unwrapped.state = raw_state

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        raw_state = self.unwrapped.state
        return raw_state, reward, done, trunc, info


class MountainCarSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)

        raw_state = self.unwrapped.state

        # refined_state = numpy.array(raw_state, dtype=numpy.float32)
        # a = numpy.array_equal(state, refined_state)

        return raw_state, info

    def set_state(self, raw_state):
        self.unwrapped.state = raw_state

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        raw_state = self.unwrapped.state
        return raw_state, reward, done, trunc, info


class TaxiSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)

        raw_state = self.unwrapped.s

        # refined_state = int(raw_state)
        # a = state == refined_state

        return raw_state, info

    def set_state(self, raw_state):
        self.unwrapped.s = raw_state

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        raw_state = self.unwrapped.s
        return raw_state, reward, done, trunc, info


class FrozenLakeSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)

        raw_state = self.unwrapped.s
        return raw_state, info

    def set_state(self, raw_state):
        self.unwrapped.s = raw_state

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        raw_state = self.unwrapped.s
        return raw_state, reward, done, trunc, info


class TaxiV4SetStepWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        raw_state = int(self.unwrapped.s)
        return raw_state, info

    def set_state(self, raw_state):
        self.unwrapped.s = int(raw_state)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        raw_state = int(self.unwrapped.s)
        return raw_state, reward, terminated, truncated, info



class SeededStochasticActionWrapper(gymnasium.ActionWrapper):
    """Transition stochasticity for MiniGrid, drawn from the SEEDED env RNG.

    MiniGrid's own StochasticActionWrapper draws its coin from the GLOBAL numpy RNG
    (`np.random.uniform()`), which `reset(seed=...)` does NOT control — that makes
    rollouts non-reproducible and bypasses the diagnoser's per-trace MC seeding. This
    version draws the coin (and the replacement action) from `self.np_random`, which
    IS seeded by `reset(seed)`, so env stochasticity is a deterministic function of the
    seed — exactly like FrozenLake "slippery" / Taxi "rainy".

    With probability `prob` the intended action executes; otherwise a random action
    from the action space is taken.
    """
    def __init__(self, env, prob=0.9):
        super().__init__(env)
        self.prob = prob

    def action(self, action):
        if self.np_random.random() < self.prob:
            return action
        return int(self.np_random.integers(0, self.action_space.n))


class MiniGridSetStepWrapper(gymnasium.Wrapper):
    """Approach-A (full-state) wrapper for a MiniGrid Empty-room navigation env.

    The raw state is the FULL MDP state for an Empty room: (agent_col, agent_row,
    agent_dir). The grid (walls + goal) is fixed for a given env id, so restoring the
    agent's position + direction fully restores the state — which is exactly what the
    Monte-Carlo diagnoser needs from set_state. (Envs with keys/doors/carrying would
    need those added to the snapshot.)

    Note: MiniGrid is partially observed at the AGENT level (7x7 egocentric view), but
    the DIAGNOSER runs on the full state here; the egocentric partiality is only the
    policy's concern (approach B, which diagnoses from observations, is a separate path).
    """
    def __init__(self, env):
        super().__init__(env)

    def _raw(self):
        u = self.unwrapped
        pos = u.agent_pos
        return (int(pos[0]), int(pos[1]), int(u.agent_dir))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._raw(), info

    def set_state(self, raw_state):
        u = self.unwrapped
        u.agent_pos = (int(raw_state[0]), int(raw_state[1]))
        u.agent_dir = int(raw_state[2])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._raw(), reward, terminated, truncated, info


wrappers = {
    "Acrobot_v1": AcrobotSetStepWrapper,
    "CartPole_v1": CartPoleSetStepWrapper,
    "MountainCar_v0": MountainCarSetStepWrapper,
    "Taxi_v3": TaxiSetStepWrapper,
    "Taxi_v4": TaxiV4SetStepWrapper,
    "FrozenLake_v1": FrozenLakeSetStepWrapper,
    "MiniGrid_Empty_Random_6x6_v0": MiniGridSetStepWrapper,
    "MiniGrid_Empty_16x16_v0": MiniGridSetStepWrapper,
}

FROZENLAKE_DESC = [
    "SFFFFFFF",
    "FFHFFFFF",
    "FFFHFFFF",
    "HFFFFFFF",
    "FFFHFFFH",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFHFFG",
]
FROZENLAKE_SLIPPERY = True

DOMAIN_KWARGS = {
    "FrozenLake_v1": {"is_slippery": FROZENLAKE_SLIPPERY,
                      "desc": FROZENLAKE_DESC},
    "Taxi_v4": {
            "is_rainy": True,
            "rainy_probability": 0.7,
            "fickle_passenger": False,
    }
}

# MiniGrid is deterministic by default; we inject transition stochasticity (the analog
# of FrozenLake "slippery") with StochasticActionWrapper: with probability MINIGRID_ACTION_PROB
# the intended action executes, otherwise a random action is taken.
MINIGRID_ACTION_PROB = 0.9


def make_wrapped_env(domain_name, render_mode):
    kwargs = DOMAIN_KWARGS.get(domain_name, {})
    is_minigrid = domain_name.startswith("MiniGrid")
    # MiniGrid, like Taxi_v4, is a gymnasium (Farama) env, not legacy gym.
    used_gym = gymnasium if (domain_name == "Taxi_v4" or is_minigrid) else gym

    if is_minigrid:
        import minigrid  # noqa: F401  (registers the MiniGrid-* envs with gymnasium)

    base_env = used_gym.make(
        domain_name.replace('_', '-'),
        render_mode=render_mode,
        **kwargs
    )

    if is_minigrid:
        base_env = SeededStochasticActionWrapper(base_env, prob=MINIGRID_ACTION_PROB)

    return wrappers[domain_name](base_env)
