import gymnasium as gym
import numpy as np


def rollout(seed, actions, sticky_prob):
    env = gym.make(
        "ALE/Pong-v5",
        obs_type="ram",
        repeat_action_probability=sticky_prob,
        frameskip=1,
        render_mode="human",
    )

    obs, info = env.reset(seed=seed)
    states = [obs.copy()]
    rewards = []

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)

        states.append(obs.copy())
        rewards.append(reward)

        if terminated or truncated:
            break

    env.close()
    return states, rewards


def same_rollout(states_a, states_b):
    if len(states_a) != len(states_b):
        return False

    return all(np.array_equal(a, b) for a, b in zip(states_a, states_b))


def first_difference_step(states_a, states_b):
    for i, (a, b) in enumerate(zip(states_a, states_b)):
        if not np.array_equal(a, b):
            return i
    return None


if __name__ == "__main__":

    actions = [2, 3] * 150

    print("=== Pong stochasticity test ===")

    # Test 1: no sticky actions, same seed
    states_1, rewards_1 = rollout(
        seed=42,
        actions=actions,
        sticky_prob=0.0,
    )

    states_2, rewards_2 = rollout(
        seed=42,
        actions=actions,
        sticky_prob=0.0,
    )

    print("\nTest 1: sticky_prob=0.0, same seed")
    print("Same rollout:", same_rollout(states_1, states_2))
    print("First difference step:", first_difference_step(states_1, states_2))

    # Test 2: sticky actions, same seed
    states_3, rewards_3 = rollout(
        seed=42,
        actions=actions,
        sticky_prob=0.25,
    )

    states_4, rewards_4 = rollout(
        seed=42,
        actions=actions,
        sticky_prob=0.25,
    )

    print("\nTest 2: sticky_prob=0.25, same seed")
    print("Same rollout:", same_rollout(states_3, states_4))
    print("First difference step:", first_difference_step(states_3, states_4))

    # Test 3: sticky actions, different seeds
    states_5, rewards_5 = rollout(
        seed=42,
        actions=actions,
        sticky_prob=0.25,
    )

    states_6, rewards_6 = rollout(
        seed=43,
        actions=actions,
        sticky_prob=0.25,
    )

    print("\nTest 3: sticky_prob=0.25, different seeds")
    print("Same rollout:", same_rollout(states_5, states_6))
    print("First difference step:", first_difference_step(states_5, states_6))

    # Test 4: no sticky actions, different seeds
    states_7, rewards_7 = rollout(
        seed=42,
        actions=actions,
        sticky_prob=0.0,
    )

    states_8, rewards_8 = rollout(
        seed=43,
        actions=actions,
        sticky_prob=0.0,
    )

    print("\nTest 4: sticky_prob=0.0, different seeds")
    print("Same rollout:", same_rollout(states_7, states_8))
    print("First difference step:", first_difference_step(states_7, states_8))

    print("\nDone.")