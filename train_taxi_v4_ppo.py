"""Train a Taxi-v4 PPO policy on the (stochastic) rainy env — fully parametrized.

Designed to be run many times with different timesteps / hyperparameters so we can
sweep, evaluate (with eval_taxi_policy.py), and pick a recipe that actually solves
the rainy task. The old single-env / default-hyperparameter / hard-coded-1M-steps
recipe produced a degenerate 0/20 policy (exploration starvation), so the levers
the brief calls out are all exposed here:

  * vectorized envs        (--n_envs)        — throughput + better gradients
  * exploration            (--ent_coef)      — default PPO 0.0 starves Taxi
  * the usual PPO knobs     --learning_rate --n_steps --batch_size --n_epochs --gamma
  * best-by-eval checkpoint (--eval_freq --n_eval_episodes via EvalCallback)
  * curriculum / warm-start (--init_from)     — e.g. learn deterministic first,
                                                then fine-tune on rain
  * live --timesteps (the old script hard-overrode it to 1_000_000)

Each run writes to its own sub-dir under --save_dir so repeated runs never collide:
  <save_dir>/<run_name>/final_model.zip      (model at end of training)
  <save_dir>/<run_name>/best_model.zip       (best by eval mean reward)
Outputs default to an untracked runs/ dir; copy a proven winner into
environments/Taxi_v4/models/PPO/ under a descriptive name only after evaluation
(do NOT overwrite Taxi_v4__PPO.zip — it is gated, see references/TRAIN_TAXI_HANDOFF.md).

Examples:
  # short local smoke run (just confirm the pipeline works)
  python train_taxi_v4_ppo.py --timesteps 50000 --n_envs 4 --eval_freq 10000 --tag smoke
  # a small probe with exploration on
  python train_taxi_v4_ppo.py --timesteps 400000 --n_envs 8 --ent_coef 0.01
  # curriculum step 2: fine-tune a deterministic-competent model on rain
  python train_taxi_v4_ppo.py --init_from runs/taxi_v4/det/best_model.zip \
      --rainy_probability 0.7 --timesteps 400000 --tag curriculum
"""
import argparse
import os

import gymnasium as gym  # importing registers the patched Taxi-v4 (is_rainy/rainy_probability)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


TAXI_ENV_KWARGS = dict(fickle_passenger=False)

# Taxi pickup/dropoff locations R,G,Y,B (taxi grid is 5x5), confirmed from env.unwrapped.locs.
TAXI_LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]


class TaxiRewardShaping(gym.Wrapper):
    """Potential-based reward shaping for Taxi (TRAINING ONLY).

    Taxi's reward is sparse (+20 only on a full deliver), so from-scratch PPO
    collapses to a "never attempt pickup/dropoff" local optimum (flat -200/episode)
    and never discovers a delivery. This adds a dense shaping term that guides the
    taxi toward the passenger, then toward the destination.

    Potential-based shaping (Ng et al. 1999) is policy-invariant: F = gamma*phi(s')
    - phi(s) does not change the optimal policy, so the learned policy is still
    correct for the true (unshaped) env the diagnoser uses. Only the training env is
    wrapped; the eval env stays the standard rainy Taxi.

    phi(s):
      * passenger not yet in taxi: -(taxi->passenger dist) - PICKUP_BONUS
      * passenger in taxi:         -(taxi->destination dist)
    The -PICKUP_BONUS offset (> grid diameter) makes the pickup transition raise the
    potential, rewarding pickup; reaching the destination drives phi -> 0.
    """
    PICKUP_BONUS = 10.0

    def __init__(self, env, gamma=0.99, scale=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self._prev_phi = 0.0

    def _phi(self, state):
        tr, tc, pass_idx, dest_idx = self.env.unwrapped.decode(int(state))
        if pass_idx < 4:  # passenger still waiting at a location
            gr, gc = TAXI_LOCS[pass_idx]
            return -(abs(tr - gr) + abs(tc - gc)) - self.PICKUP_BONUS
        gr, gc = TAXI_LOCS[dest_idx]  # passenger in taxi -> head to destination
        return -(abs(tr - gr) + abs(tc - gc))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = 0.0 if terminated else self._phi(obs)
        shaped = self.gamma * phi_next - self._prev_phi
        self._prev_phi = phi_next
        return obs, reward + self.scale * shaped, terminated, truncated, info


def make_vec(rainy_probability, n_envs, seed, shaped=False, gamma=0.99, shaping_scale=1.0):
    wrapper_class = TaxiRewardShaping if shaped else None
    wrapper_kwargs = dict(gamma=gamma, scale=shaping_scale) if shaped else None
    return make_vec_env(
        "Taxi-v4",
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(is_rainy=True, rainy_probability=rainy_probability, **TAXI_ENV_KWARGS),
        wrapper_class=wrapper_class,
        wrapper_kwargs=wrapper_kwargs,
    )


def build_run_name(args):
    if args.tag:
        base = args.tag
    else:
        warm = "warm_" if args.init_from else ""
        base = (f"{warm}rainy{args.rainy_probability}_t{args.timesteps}"
                f"_n{args.n_envs}_ent{args.ent_coef}_lr{args.learning_rate}_seed{args.seed}")
    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # what / how long
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="total env steps to train (LIVE — no longer hard-overridden)")
    parser.add_argument("--rainy_probability", type=float, default=0.7,
                        help="P(intended action succeeds); 1.0 = deterministic Taxi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=8,
                        help="parallel envs for the vectorized rollout")
    # PPO hyperparameters
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="entropy bonus; PPO default 0.0 starves exploration on Taxi")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024,
                        help="rollout length per env before each update")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    # eval / checkpointing
    parser.add_argument("--eval_freq", type=int, default=25_000,
                        help="eval every this many steps PER ENV (EvalCallback semantics)")
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    # reward shaping (training env only; eval env stays the true rainy Taxi)
    parser.add_argument("--shaped", action="store_true",
                        help="add potential-based shaping that guides taxi->passenger->dest")
    parser.add_argument("--shaping_scale", type=float, default=1.0,
                        help="multiplier on the shaping term")
    # curriculum / warm start
    parser.add_argument("--init_from", type=str, default=None,
                        help="path to a .zip to warm-start from (continue training)")
    # output
    parser.add_argument("--save_dir", type=str, default="runs/taxi_v4",
                        help="untracked dir for run outputs; winner copied to models dir later")
    parser.add_argument("--tag", type=str, default=None, help="run-name label")
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = build_run_name(args)
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[train] run_dir = {run_dir}")
    print(f"[train] config = {vars(args)}")

    train_env = make_vec(args.rainy_probability, args.n_envs, args.seed,
                         shaped=args.shaped, gamma=args.gamma, shaping_scale=args.shaping_scale)
    # eval on the TARGET env (same rainy_probability), UNSHAPED, separate seed offset
    eval_env = make_vec(args.rainy_probability, 1, args.seed + 10_000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    if args.init_from:
        print(f"[train] warm-starting from {args.init_from}")
        model = PPO.load(args.init_from, env=train_env)
        # let CLI override the exploration knob on continuation (others carry over)
        model.ent_coef = args.ent_coef
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=args.seed,
            ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
        )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback,
                reset_num_timesteps=not bool(args.init_from))

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    train_env.close()
    eval_env.close()

    print(f"[train] saved final model to: {final_path}.zip")
    print(f"[train] best-by-eval model (if any) at: {os.path.join(run_dir, 'best_model.zip')}")
    print(f"[train] evaluate with:  python eval_taxi_policy.py --model {final_path}.zip")


if __name__ == "__main__":
    main()
