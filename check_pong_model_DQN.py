import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN


MODEL_PATH = "environments/ALE/Pong_v5/models/DQN/pong_ram_dqn_sticky_0_25_frameskip_1_steps_20000000_seed_1/best_model.zip"
# or:
# MODEL_PATH = "environments/ALE/Pong_v5/models/DQN/pong_ram_dqn_sticky_0_25_frameskip_1_steps_20000000_seed_1/final_model.zip"

ENV_ID = "ALE/Pong-v5"


def make_env(render_mode="rgb_array"):
    return gym.make(
        ENV_ID,
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=0.25,
        render_mode=render_mode,
    )


def evaluate_model(model, n_episodes=10):
    env = make_env()

    episode_rewards = []
    action_counts = np.zeros(env.action_space.n, dtype=int)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        trunc = False
        total_reward = 0
        steps = 0

        while not done and not trunc:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            action_counts[action] += 1

            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward={total_reward}, steps={steps}")

    env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {np.mean(episode_rewards)}")
    print(f"Min reward: {np.min(episode_rewards)}")
    print(f"Max reward: {np.max(episode_rewards)}")

    print("\n=== Action Counts ===")
    env2 = make_env()
    meanings = env2.unwrapped.get_action_meanings()
    env2.close()

    for i, count in enumerate(action_counts):
        print(f"{i} = {meanings[i]}: {count}")


def watch_model(model):
    env = make_env(render_mode="human")

    obs, info = env.reset(seed=42)
    done = False
    trunc = False
    total_reward = 0

    while not done and not trunc:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

    env.close()
    print(f"Watched episode reward: {total_reward}")


if __name__ == "__main__":
    print(f"Loading DQN model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH, device="cpu")

    evaluate_model(model, n_episodes=50)

    # Uncomment locally if you want to watch:
    watch_model(model)