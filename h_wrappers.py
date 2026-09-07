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
        self._grid_built = False
        # BIG speedup: MiniGrid's step()/reset() build a 7x7 egocentric observation via
        # gen_obs() and return it — but we never use it (we read agent_pos/agent_dir directly,
        # and belief/comparator use precomputed view maps). gen_obs was ~75% of diagnosis time
        # (called once per Monte-Carlo step). Stub it to a no-op; MiniGrid only returns its
        # value, never uses it internally. (Env built with disable_env_checker=True so no
        # wrapper validates the None observation.)
        self.unwrapped.gen_obs = lambda *a, **k: None

    def _raw(self):
        u = self.unwrapped
        pos = u.agent_pos
        return (int(pos[0]), int(pos[1]), int(u.agent_dir))

    def reset(self, seed=None, options=None):
        # Full reset (rebuilds the grid). A "fast reset" that skips the rebuild was tried but
        # CHANGED RESULTS: Empty-16x16 grid generation consumes np_random draws, so skipping it
        # shifts the RNG stream and alters the Monte-Carlo outcomes. The big win is the gen_obs
        # stub (see __init__) + disable_env_checker, which are RNG-neutral.
        obs, info = self.env.reset(seed=seed, options=options)
        return self._raw(), info

    def set_state(self, raw_state):
        u = self.unwrapped
        u.agent_pos = (int(raw_state[0]), int(raw_state[1]))
        u.agent_dir = int(raw_state[2])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._raw(), reward, terminated, truncated, info


_MINIGRID_VIEW_MAPS = {}

def build_minigrid_view_maps(domain_name, render_seed=0):
    """Precompute, once per domain, the maps
        state2view : (col,row,dir) -> egocentric-view bytes
        view2states: view bytes    -> [ (col,row,dir), ... ]
    by enumerating every interior state of the (fixed) grid and rendering it. The belief
    localization and the view comparator then run as O(1) dict lookups instead of calling
    gen_obs() hundreds of thousands of times in the Monte-Carlo hot loop (the real bottleneck).
    Cached by domain, so the diagnoser's per-call wrappers all share one build."""
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
                vb = u.gen_obs()["image"].tobytes()
                state2view[(x, y, d)] = vb
                view2states[vb].append((x, y, d))
    env.close()
    maps = (state2view, dict(view2states))
    _MINIGRID_VIEW_MAPS[domain_name] = maps
    return maps


class MiniGridBeliefSetStepWrapper(MiniGridSetStepWrapper):
    """Approach B (partial observability): `set_state` does NOT restore the exact state.

    We only observe the agent's egocentric VIEW, and many states share a view (aliasing).
    So given a state whose view we observed, this wrapper localizes — looks up ALL states
    that produce the same view — and SAMPLES one of them (position AND direction) to start
    the rollout from. The Monte-Carlo thus averages over "where am I / which way am I facing?"
    Pair with the view comparator so the hit test also compares views, not exact states.

    Sampling uses a per-reset seeded RNG, so it is reproducible and folds into the
    diagnoser's per-trace MC seeding (see [[seeding-namespace-redesign]]).
    """
    def __init__(self, env, domain_name):
        super().__init__(env)
        # precomputed O(1) localization maps (shared, cached per domain)
        self._state2view, self._view_to_states = build_minigrid_view_maps(domain_name)
        self._belief_rng = random.Random(0)

    def reset(self, seed=None, options=None):
        raw, info = super().reset(seed=seed, options=options)
        self._belief_rng = random.Random(seed if seed is not None else 0)
        return raw, info

    def set_state(self, raw_state):
        key = (int(raw_state[0]), int(raw_state[1]), int(raw_state[2]))
        vb = self._state2view.get(key)
        candidates = self._view_to_states.get(vb, [key])
        sampled = self._belief_rng.choice(candidates)   # sample a consistent (col,row,dir)
        u = self.unwrapped
        u.agent_pos = (int(sampled[0]), int(sampled[1])); u.agent_dir = int(sampled[2])


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
# of FrozenLake "slippery") with SeededStochasticActionWrapper: with probability
# MINIGRID_ACTION_PROB the intended action executes, otherwise a random action is taken.
# Keep <= 0.7 (>= 30% noise) so the domain is genuinely stochastic, comparable to
# FrozenLake slippery / Taxi rainy.
MINIGRID_ACTION_PROB = 0.7

# Approach selector for MiniGrid. False (default) = approach A: set_state restores the exact
# state. True = approach B: set_state samples a state consistent with the observed view
# (MiniGridBeliefSetStepWrapper). B mode should be paired with the view comparator. Opt in by
# setting h_wrappers.MINIGRID_BELIEF_MODE = True before building the env / running the diagnoser.
MINIGRID_BELIEF_MODE = False


def make_wrapped_env(domain_name, render_mode):
    kwargs = DOMAIN_KWARGS.get(domain_name, {})
    is_minigrid = domain_name.startswith("MiniGrid")
    # MiniGrid, like Taxi_v4, is a gymnasium (Farama) env, not legacy gym.
    used_gym = gymnasium if (domain_name == "Taxi_v4" or is_minigrid) else gym

    if is_minigrid:
        import minigrid  # noqa: F401  (registers the MiniGrid-* envs with gymnasium)
        # disable_env_checker: drop the PassiveEnvChecker wrapper (per-step overhead, and it
        # would reject our stubbed None observation — see MiniGridSetStepWrapper).
        base_env = used_gym.make(
            domain_name.replace('_', '-'),
            render_mode=render_mode,
            disable_env_checker=True,
            **kwargs
        )
    else:
        base_env = used_gym.make(
            domain_name.replace('_', '-'),
            render_mode=render_mode,
            **kwargs
        )

    if is_minigrid:
        base_env = SeededStochasticActionWrapper(base_env, prob=MINIGRID_ACTION_PROB)
        if MINIGRID_BELIEF_MODE:
            return MiniGridBeliefSetStepWrapper(base_env, domain_name)  # approach B (localize+sample)

    return wrappers[domain_name](base_env)
