import os
import argparse
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


ENV_ID = "ALE/Pong-v5"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--sticky", type=float, default=0.25)
    parser.add_argument("--frameskip", type=int, default=1)

    # Optional: continue training from existing model
    parser.add_argument("--load-path", type=str, default=None)

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


def build_new_model(env, seed):
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


def main():
    args = parse_args()
    args.load_path = "environments/ALE/Pong_v5/models/DQN/pong_ram_dqn_sticky_0_25_frameskip_1_steps_20000000_seed_1/best_model.zip"


    continue_tag = "continued" if args.load_path else "new"

    run_name = (
        f"pong_ram_dqn"
        f"_sticky_{str(args.sticky).replace('.', '_')}"
        f"_frameskip_{args.frameskip}"
        f"_steps_{args.timesteps}"
        f"_seed_{args.seed}"
        f"_{continue_tag}"
    )

    models_dir = f"environments/ALE/Pong_v5/models/DQN/{run_name}"
    log_dir = f"environments/ALE/Pong_v5/logs/DQN/{run_name}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("====================================")
    print(f"ENV_ID: {ENV_ID}")
    print("Algorithm: DQN")
    print(f"Timesteps to train now: {args.timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Sticky actions: {args.sticky}")
    print(f"Frameskip: {args.frameskip}")
    print(f"Load path: {args.load_path}")
    print(f"Models dir: {models_dir}")
    print("====================================")

    env = make_env(args.seed, args.sticky, args.frameskip)
    eval_env = make_env(args.seed + 10_000, args.sticky, args.frameskip)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    if args.load_path:
        print(f"Loading existing DQN model from: {args.load_path}")
        model = DQN.load(
            args.load_path,
            env=env,
            seed=args.seed,
            device="auto",
        )
        reset_num_timesteps = False
    else:
        model = build_new_model(env, args.seed)
        reset_num_timesteps = True

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_path = f"{models_dir}/final_model"
    model.save(final_path)

    env.close()
    eval_env.close()

    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()