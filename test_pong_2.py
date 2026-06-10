import numpy as np
from h_wrappers import make_wrapped_env


DOMAIN_NAME = "ALE/Pong_v5"


def rollout(env, actions):
    states = []
    rewards = []

    for action in actions:
        obs, reward, done, trunc, info = env.step(action)
        states.append(obs["ram"].copy())
        rewards.append(reward)

        if done or trunc:
            break

    return states, rewards


def main():
    env = make_wrapped_env(DOMAIN_NAME, render_mode="rgb_array")

    obs, info = env.reset(seed=42)

    # move a little from initial state
    warmup_actions = [1, 2, 2, 3, 4, 5, 0, 2, 3, 3]
    for a in warmup_actions:
        obs, reward, done, trunc, info = env.step(a)

    saved_state = obs

    test_actions = [2, 2, 3, 3, 4, 5, 0, 0, 2, 3, 4, 5]

    env.set_state(saved_state)
    states1, rewards1 = rollout(env, test_actions)

    env.set_state(saved_state)
    states2, rewards2 = rollout(env, test_actions)

    same_rewards = rewards1 == rewards2
    same_states = all(np.array_equal(s1, s2) for s1, s2 in zip(states1, states2))

    print("Same rewards:", same_rewards)
    print("Same RAM states:", same_states)
    print("Number of compared states:", len(states1), len(states2))

    env.close()


if __name__ == "__main__":
    main()