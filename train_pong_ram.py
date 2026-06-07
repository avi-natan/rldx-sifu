import os
import argparse
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


ENV_ID = "ALE/Pong-v5"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--sticky", type=float, default=0.25)
    parser.add_argument("--frameskip", type=int, default=1)

    return parser.parse_args()


def make_env(seed, sticky, frameskip):
    env = gym.make(
        ENV_ID,
        obs_type="ram",
        frameskip=frameskip,
        repeat_action_probability=sticky,
        render_mode="rgb_array",
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def build_model(algo, env, seed):
    if algo == "PPO":
        return PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            seed=seed,
            device="auto",
        )

    if algo == "DQN":
        return DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=50_000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10_000,
            exploration_fraction=0.15,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            seed=seed,
            device="auto",
        )

    raise ValueError(f"Unknown algo: {algo}")


def main():
    args = parse_args()

    run_name = (
        f"pong_ram_{args.algo.lower()}"
        f"_sticky_{str(args.sticky).replace('.', '_')}"
        f"_frameskip_{args.frameskip}"
        f"_steps_{args.timesteps}"
        f"_seed_{args.seed}"
    )

    models_dir = f"environments/ALE/Pong_v5/models/{args.algo}/{run_name}"
    log_dir = f"environments/ALE/Pong_v5/logs/{args.algo}/{run_name}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("====================================")
    print(f"ENV_ID: {ENV_ID}")
    print(f"Algorithm: {args.algo}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Sticky actions: {args.sticky}")
    print(f"Frameskip: {args.frameskip}")
    print(f"Models dir: {models_dir}")
    print("====================================")

    env = make_env(args.seed, args.sticky, args.frameskip)
    eval_env = make_env(args.seed + 10_000, args.sticky, args.frameskip)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=models_dir,
        name_prefix=f"pong_ram_{args.algo.lower()}",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    model = build_model(args.algo, env, args.seed)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    final_path = f"{models_dir}/final_model"
    model.save(final_path)

    env.close()
    eval_env.close()

    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()