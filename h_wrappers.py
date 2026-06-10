import gym
import numpy
from numpy import cos, sin

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

class PongRamSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        raw_state = self.get_state()
        return raw_state, info

    def get_state(self):
        return {
            "ale_state": self.unwrapped.ale.cloneSystemState(),
            "ram": self.unwrapped.ale.getRAM().copy(),
        }

    def set_state(self, raw_state):
        self.unwrapped.ale.restoreSystemState(raw_state["ale_state"])

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        raw_state = self.get_state()
        return raw_state, reward, done, trunc, info



wrappers = {
    "Acrobot_v1": AcrobotSetStepWrapper,
    "CartPole_v1": CartPoleSetStepWrapper,
    "MountainCar_v0": MountainCarSetStepWrapper,
    "Taxi_v3": TaxiSetStepWrapper,
    "FrozenLake_v1": FrozenLakeSetStepWrapper,
    "ALE/Pong_v5": PongRamSetStepWrapper
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

    "ALE/Pong_v5": {
        "obs_type": "ram",
        "frameskip": 1,
        "repeat_action_probability": 0.25,
    },
}


def make_wrapped_env(domain_name, render_mode):
    kwargs = DOMAIN_KWARGS.get(domain_name, {})
    base_env = gym.make(domain_name.replace('_', '-'), render_mode=render_mode, **kwargs)
    return wrappers[domain_name](base_env)
