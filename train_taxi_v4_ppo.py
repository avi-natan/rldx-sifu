import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def make_env(rainy_probability: float, seed: int):
    env = gym.make(
        "Taxi-v4",
        is_rainy=True,
        rainy_probability=rainy_probability,
        fickle_passenger=False,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--rainy_probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="environments/Taxi_v4/models/PPO")
    args = parser.parse_args()

    args.timesteps = 1000000

    os.makedirs(args.save_dir, exist_ok=True)

    env = make_env(args.rainy_probability, args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed
    )

    model.learn(total_timesteps=args.timesteps)

    save_path = os.path.join(
        args.save_dir,
        f"Taxi_v4_PPO_rainy_{args.rainy_probability}_steps_{args.timesteps}_seed_{args.seed}"
    )

    model.save(save_path)
    env.close()

    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()