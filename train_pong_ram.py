import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


ENV_ID = "ALE/Pong-v5"
MODEL_NAME = "PPO"

TOTAL_TIMESTEPS = 3_000_000
SAVE_FREQ = 100_000

RUN_NAME = "pong_ram_ppo_sticky_025_frameskip_1"

MODELS_DIR = f"environments/ALE/Pong_v5/models/{MODEL_NAME}/{RUN_NAME}"
LOG_DIR = f"environments/ALE/Pong_v5/logs/{MODEL_NAME}/{RUN_NAME}"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    env = gym.make(
        ENV_ID,
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=0.25,
        render_mode="rgb_array",
    )
    env = Monitor(env)
    return env


def main():
    env = make_env()
    eval_env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODELS_DIR,
        name_prefix="pong_ram_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=SAVE_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    model = PPO(
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
        seed=42,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    final_path = f"{MODELS_DIR}/final_model"
    model.save(final_path)

    env.close()
    eval_env.close()

    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()