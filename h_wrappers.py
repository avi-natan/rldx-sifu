import random
from collections import defaultdict

import gym
import gymnasium
from gymnasium.utils import seeding
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

    MiniGrid's own StochasticActionWrapper draws its slip-coin from the GLOBAL numpy RNG
    (`np.random.uniform()`), which `reset(seed=...)` does NOT control — that makes rollouts
    non-reproducible and bypasses the diagnoser's per-trace MC seeding (the SEED_BLOCK scheme
    + common random numbers across candidate faults). This version draws BOTH the coin and the
    replacement action from `self.np_random`, which IS seeded by `reset(seed)`, so env
    stochasticity is a deterministic function of the seed — like FrozenLake "slippery" / Taxi "rainy".

    With probability `prob` the intended action executes; otherwise a random action is drawn
    from `noise_actions`. This defaults to the 3 MEANINGFUL Empty-room actions
    (0=left, 1=right, 2=forward): actions 3-6 (pickup/drop/toggle/done) are no-ops in an empty
    room, so sampling the full Discrete(7) would turn most slips into a "stall in place" rather
    than a genuine random move — not comparable to the movement-noise of FrozenLake/Taxi.
    (MiniGrid's own wrapper has the same flaw: it samples 0-5, i.e. 3 of 6 are no-ops.)
    """
    def __init__(self, env, prob=0.9, noise_actions=(0, 1, 2)):
        super().__init__(env)
        self.prob = prob
        self.noise_actions = tuple(noise_actions)

    def action(self, action):
        if self.np_random.random() < self.prob:
            return action
        return int(self.np_random.choice(self.noise_actions))


_MINIGRID_VIEW_MAPS = {}

def build_minigrid_view_maps(domain_name, render_seed=0):
    """Precompute, once per domain, the maps
        state2view : (col,row,dir) -> observation key = (egocentric-image bytes, direction)
        view2states: observation key -> [ (col,row,dir), ... ]
    by enumerating every interior state of the (fixed) grid and reading its observation. The
    localization and the view comparator then run as O(1) dict lookups instead of calling
    gen_obs() hundreds of thousands of times in the Monte-Carlo hot loop (the real bottleneck).
    Cached by domain, so the diagnoser's per-call wrappers all share one build.

    The observation key mirrors the DEFAULT MiniGrid observation the agent sees:
    Dict(image, direction, mission). We key on the egocentric IMAGE *and* the DIRECTION compass
    (mission is constant, so ignored). Because direction is observed, two states are aliased only
    when they share BOTH the same image AND the same heading -> the remaining ambiguity is purely
    POSITIONAL (in the open interior, many cells give the all-empty image, but only for the same
    facing)."""
    if domain_name in _MINIGRID_VIEW_MAPS:
        return _MINIGRID_VIEW_MAPS[domain_name]
    import minigrid  # noqa: F401
    env = gymnasium.make(domain_name.replace('_', '-'))
    env.reset(seed=render_seed)
    u = env.unwrapped
    state2view, view2states = {}, defaultdict(list)
    for x in range(1, u.width - 1):
        for y in range(1, u.height - 1):
            for d in range(4):
                u.agent_pos = (x, y); u.agent_dir = d
                obs = u.gen_obs()
                vb = (obs["image"].tobytes(), int(obs["direction"]))  # default obs = image + direction
                state2view[(x, y, d)] = vb
                view2states[vb].append((x, y, d))
    env.close()
    maps = (state2view, dict(view2states))
    _MINIGRID_VIEW_MAPS[domain_name] = maps
    return maps


class MiniGridSetStepWrapper(gymnasium.Wrapper):
    """Makes a MiniGrid Empty-room env diagnosable under PARTIAL OBSERVABILITY.

    We only ever observe the agent's DEFAULT observation (egocentric 7x7 image + direction
    compass), and many (col,row,dir) states produce the SAME observation (aliasing). Because
    direction is observed, the ambiguity is purely POSITIONAL: states sharing the same image AND
    the same heading. So `set_state` does NOT restore a known true state: it localizes the observed
    view to the set of states consistent with it and SAMPLES one consistent POSITION to roll
    forward from (the sampled state keeps the observed direction). The Monte-Carlo diagnoser thus
    averages over "where am I?" given a known facing. Pair with the view comparator (the default
    for MiniGrid domains) so the MC hit test also compares observations. Sampling uses a per-reset seeded RNG, so it is reproducible and folds into
    the diagnoser's per-trace seeding (see [[seeding-namespace-redesign]]).

    Perf: MiniGrid's step()/reset() build a 7x7 observation via gen_obs() and return it, but we
    never use it (we read agent_pos/agent_dir directly and use precomputed view maps). gen_obs
    was ~75% of diagnosis time (once per MC step), so it is stubbed to a no-op; the env is built
    with disable_env_checker=True so nothing validates the stubbed None observation.
    """
    def __init__(self, env, domain_name):
        super().__init__(env)
        self.unwrapped.gen_obs = lambda *a, **k: None   # perf: the observation is never used
        self._state2view, self._view_to_states = build_minigrid_view_maps(domain_name)
        self._belief_rng = random.Random(0)

    def _raw(self):
        u = self.unwrapped
        pos = u.agent_pos
        return (int(pos[0]), int(pos[1]), int(u.agent_dir))

    def reset(self, seed=None, options=None):
        # Full reset (rebuilds the grid). A "fast reset" skipping the rebuild was tried and
        # reverted: Empty-16x16 grid generation consumes np_random draws, so skipping it shifts
        # the RNG stream and changes results.
        obs, info = self.env.reset(seed=seed, options=options)
        self._belief_rng = random.Random(seed if seed is not None else 0)
        return self._raw(), info

    def set_state(self, raw_state):
        # localize the observed view -> sample a consistent (col,row,dir) to roll forward from
        key = (int(raw_state[0]), int(raw_state[1]), int(raw_state[2]))
        vb = self._state2view.get(key)
        candidates = self._view_to_states.get(vb, [key])
        sampled = self._belief_rng.choice(candidates)
        u = self.unwrapped
        u.agent_pos = (int(sampled[0]), int(sampled[1])); u.agent_dir = int(sampled[2])

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
    # MiniGrid domains are constructed directly in make_wrapped_env (they need domain_name).
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
# of FrozenLake "slippery") with SeededStochasticActionWrapper: with probability
# MINIGRID_ACTION_PROB the intended action executes, otherwise a random action is taken.
# The trained obs-policy loaded in h_rl_models is selected to MATCH this value
# (models/PPO/..._noise{MINIGRID_ACTION_PROB}.zip), so the policy is always trained under the
# same noise it is diagnosed under. This is the single knob that switches BOTH the env noise AND
# the loaded policy together -- ONLY meaningful for the MiniGrid Empty domain (that is the only
# domain with per-noise trained policies). `main.py --mg_noise {0.3,0.5,0.7}` sets it via
# set_minigrid_action_prob() for a benchmark run; the default is the diagnosable 0.7.
MINIGRID_SUPPORTED_NOISE = (0.3, 0.5, 0.7)   # the noise levels we have trained Empty policies for
MINIGRID_ACTION_PROB = 0.7

def set_minigrid_action_prob(prob):
    """Set the MiniGrid Empty env-noise level (and thereby the matching policy that h_rl_models
    loads). Only the values in MINIGRID_SUPPORTED_NOISE are allowed, since each needs a trained
    policy at models/PPO/..._noise{prob}.zip. Call this from main BEFORE running a MiniGrid
    experiment. No effect on non-MiniGrid domains (they never read MINIGRID_ACTION_PROB)."""
    global MINIGRID_ACTION_PROB
    if prob not in MINIGRID_SUPPORTED_NOISE:
        raise ValueError(f"MiniGrid noise {prob} not supported; trained policies exist for "
                         f"{MINIGRID_SUPPORTED_NOISE}.")
    MINIGRID_ACTION_PROB = prob
    print(f"[MiniGrid] env noise + policy set to noise_prob={prob}")


_MINIGRID_CUSTOM_REGISTERED = False

def _register_minigrid_custom_envs():
    """Register our custom (non-preset) MiniGrid env IDs once. SimpleCrossing at 11x11 with 2
    crossings is NOT a MiniGrid preset (only S9N1/2/3 and S11N5 exist), so we self-register it as
    `MiniGrid-SimpleCrossing-S11N2-v0` (walls) with dashes so it round-trips from the domain code
    key `MiniGrid_SimpleCrossing_S11N2_v0` via replace('_','-')."""
    global _MINIGRID_CUSTOM_REGISTERED
    if _MINIGRID_CUSTOM_REGISTERED:
        return
    import gymnasium as _gym
    from gymnasium.envs.registration import registry as _registry
    from minigrid.core.world_object import Wall
    if "MiniGrid-SimpleCrossing-S11N2-v0" not in _registry:
        _gym.register(id="MiniGrid-SimpleCrossing-S11N2-v0",
                      entry_point="minigrid.envs:CrossingEnv",
                      kwargs=dict(size=11, num_crossings=2, obstacle_type=Wall))
    _MINIGRID_CUSTOM_REGISTERED = True


def make_wrapped_env(domain_name, render_mode):
    kwargs = DOMAIN_KWARGS.get(domain_name, {})
    is_minigrid = domain_name.startswith("MiniGrid")
    # MiniGrid, like Taxi_v4, is a gymnasium (Farama) env, not legacy gym.
    used_gym = gymnasium if (domain_name == "Taxi_v4" or is_minigrid) else gym

    if is_minigrid:
        import minigrid  # noqa: F401  (registers the MiniGrid-* envs with gymnasium)
        _register_minigrid_custom_envs()  # + our custom IDs (SimpleCrossing-S11N2)
        # disable_env_checker: drop the PassiveEnvChecker wrapper (per-step overhead, and it
        # would reject our stubbed None observation — see MiniGridSetStepWrapper).
        base_env = used_gym.make(
            domain_name.replace('_', '-'),
            render_mode=render_mode,
            disable_env_checker=True,
            **kwargs
        )
        base_env = SeededStochasticActionWrapper(base_env, prob=MINIGRID_ACTION_PROB)
        return MiniGridSetStepWrapper(base_env, domain_name)  # localize view + sample state

    base_env = used_gym.make(
        domain_name.replace('_', '-'),
        render_mode=render_mode,
        **kwargs
    )
    return wrappers[domain_name](base_env)
