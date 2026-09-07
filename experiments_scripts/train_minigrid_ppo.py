"""Train a PPO policy for MiniGrid Empty that consumes the OBSERVATION (not the true state).

The diagnoser is partially observed; item 6 of the MiniGrid plan is to train a policy that maps
the agent's OBSERVATION -> action (instead of the hardcoded navigator that reads (col,row,dir)).

Observation fed to the net = the DEFAULT MiniGrid observation, made SB3-friendly:
    flatten(image 7x7x3, /255)  ++  one_hot(direction, 4)   -> a 151-dim float vector.
Mission text is constant in Empty, so it is dropped. This transform is a DETERMINISTIC function
of the raw obs (hence of the true state), so it can be reproduced from (col,row,dir) at diagnosis
time by a refiner.

Optionally trains UNDER the same action stochasticity the diagnoser uses
(h_wrappers.SeededStochasticActionWrapper, prob = --noise_prob), so the greedy policy is robust to
the 30% action noise it will face during diagnosis. --noise_prob 1.0 = deterministic dynamics.

Run (local quick check):
  ./.venv_domains/Scripts/python.exe experiments_scripts/train_minigrid_ppo.py \
      --domain MiniGrid-Empty-16x16-v0 --timesteps 20000 --n_envs 4 --noise_prob 1.0 --out_dir runs/minigrid_smoke
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath("."))

import gymnasium
import minigrid  # noqa: F401  (registers MiniGrid-* with gymnasium)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from h_wrappers import SeededStochasticActionWrapper


class ImgDirFlatWrapper(gymnasium.ObservationWrapper):
    """DEFAULT MiniGrid obs -> flat float vector: flatten(image)/255 ++ one_hot(direction)."""
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        self._img_size = int(np.prod(img_space.shape))
        n = self._img_size + 4  # + one-hot direction
        self.observation_space = gymnasium.spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        img = obs["image"].astype(np.float32).ravel() / 255.0
        d = np.zeros(4, dtype=np.float32)
        d[int(obs["direction"])] = 1.0
        return np.concatenate([img, d])


def make_env(domain, noise_prob, seed, rank):
    def _thunk():
        env = gymnasium.make(domain, disable_env_checker=True)
        if noise_prob < 1.0:
            env = SeededStochasticActionWrapper(env, prob=noise_prob)  # match diagnosis-time noise
        env = ImgDirFlatWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _thunk


def evaluate(model, domain, noise_prob, seed, episodes=200):
    """Greedy (deterministic) rollouts; report success rate + avg reward/steps under noise_prob."""
    env = make_env(domain, noise_prob, seed + 10_000, 0)()
    succ, rets, lens = 0, [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = trunc = False
        ret, steps = 0.0, 0
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(int(a))
            ret += r; steps += 1
        rets.append(ret); lens.append(steps)
        if r > 0:  # MiniGrid gives positive reward only on reaching the goal
            succ += 1
    env.close()
    return {"success_rate": succ / episodes, "avg_return": float(np.mean(rets)),
            "avg_steps": float(np.mean(lens)), "episodes": episodes}


def render_episode_gif(model, domain, noise_prob, seed, out_path, max_steps=200):
    """Roll out ONE greedy episode with rgb_array frames and save an animated GIF (needs PIL)."""
    env = gymnasium.make(domain, disable_env_checker=True, render_mode="rgb_array")
    if noise_prob < 1.0:
        env = SeededStochasticActionWrapper(env, prob=noise_prob)
    env = ImgDirFlatWrapper(env)
    obs, _ = env.reset(seed=seed + 55_555)
    frames, done, trunc, steps = [], False, False, 0
    while not (done or trunc) and steps < max_steps:
        frames.append(env.render())
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, _ = env.step(int(a))
        steps += 1
    frames.append(env.render())
    env.close()
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=200, loop=0)
        print(f"[render] saved {len(frames)}-frame GIF -> {out_path} (reached_goal={r>0})", flush=True)
    except Exception as e:
        print(f"[render] GIF save failed ({e}); frames={len(frames)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="MiniGrid-Empty-16x16-v0")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--noise_prob", type=float, default=0.7,
                   help="prob intended action executes during TRAINING (1.0 = deterministic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/minigrid")
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--learning_rate", type=float, default=2.5e-4)
    p.add_argument("--net_width", type=int, default=64, help="MLP hidden width (both layers)")
    p.add_argument("--use_subproc", action="store_true", help="SubprocVecEnv (parallel cores)")
    p.add_argument("--render_gif", action="store_true", help="save a greedy-episode GIF after training")
    args = p.parse_args()

    tag = f"{args.domain.replace('-','_')}_noise{args.noise_prob}_seed{args.seed}"
    out = os.path.join(args.out_dir, tag)
    os.makedirs(out, exist_ok=True)
    print(f"[train] {tag}  timesteps={args.timesteps} n_envs={args.n_envs} -> {out}", flush=True)

    VecCls = SubprocVecEnv if args.use_subproc else DummyVecEnv
    venv = VecCls([make_env(args.domain, args.noise_prob, args.seed, r) for r in range(args.n_envs)])
    venv = VecMonitor(venv)

    policy_kwargs = dict(net_arch=[args.net_width, args.net_width])
    model = PPO("MlpPolicy", venv, seed=args.seed, n_steps=args.n_steps, batch_size=256,
                gae_lambda=0.95, gamma=0.99, ent_coef=args.ent_coef,
                learning_rate=args.learning_rate, policy_kwargs=policy_kwargs, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(os.path.join(out, "model"))
    dt = time.time() - t0

    res = evaluate(model, args.domain, args.noise_prob, args.seed, episodes=200)
    line = (f"[done] {tag}  train_sec={dt:.0f}  success_rate={res['success_rate']:.3f}  "
            f"avg_return={res['avg_return']:.3f}  avg_steps={res['avg_steps']:.1f}")
    print(line, flush=True)
    with open(os.path.join(out, "eval.txt"), "w") as f:
        f.write(line + "\n" + repr(res) + "\n")

    if args.render_gif:
        render_episode_gif(model, args.domain, args.noise_prob, args.seed,
                           os.path.join(out, "greedy_episode.gif"))


if __name__ == "__main__":
    main()
