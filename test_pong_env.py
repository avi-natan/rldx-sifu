import gymnasium as gym
import numpy as np


def test_pong_ram():
    print("Testing Pong RAM environment...")

    env = gym.make(
        "ALE/Pong-v5",
        obs_type="ram",
        repeat_action_probability=0.25,  # sticky actions = stochastic ALE
        frameskip=1,
        render_mode=None,
    )

    obs, info = env.reset(seed=42)

    print("Observation type:", type(obs))
    print("Observation shape:", obs.shape)
    print("Observation dtype:", obs.dtype)
    print("Action space:", env.action_space)
    print("Number of actions:", env.action_space.n)
    print("Initial RAM sample:", obs[:10])

    total_reward = 0

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if step % 20 == 0:
            print(
                f"step={step}, action={action}, reward={reward}, "
                f"terminated={terminated}, truncated={truncated}"
            )

        if terminated or truncated:
            print("Episode ended at step:", step)
            obs, info = env.reset()

    env.close()
    print("Finished successfully.")
    print("Total reward:", total_reward)


if __name__ == "__main__":
    test_pong_ram()