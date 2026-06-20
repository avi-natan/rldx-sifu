"""Honest success@N evaluation for a Taxi-v4 PPO policy on the rainy env.

Mirrors what the diagnoser actually uses: a *deterministic* policy
(`model.predict(obs, deterministic=True)`, same as `TaxiHardcodedPolicy` in
`h_rl_models.py`) rolled out on the stochastic rainy Taxi env. Counts how many
seeds end in a successful dropoff (the +20 terminal reward) within the step cap.

This is the 11/20-vs-0/20 metric from the handoff brief. Run it on both the old
`Taxi_v4__PPO.zip` and any freshly-trained candidate before promoting anything.

Usage:
  python eval_taxi_policy.py                      # eval both default models, seeds 0..19
  python eval_taxi_policy.py --model path/to.zip  # eval one model
  python eval_taxi_policy.py --n_seeds 50 --rainy_probability 0.7
"""
import argparse
import os

import gymnasium as gym  # importing registers the patched Taxi-v4 (is_rainy/rainy_probability)
from stable_baselines3 import PPO

MODELS_DIR = "environments/Taxi_v4/models/PPO"
OLD_GOOD = os.path.join(MODELS_DIR, "Taxi_v4__PPO.zip")

# Taxi: +20 on successful dropoff (the only terminating reward), -10 illegal, -1/step.
DELIVERY_REWARD = 20
TRUNCATION = 200


def evaluate(model_path, n_seeds=20, rainy_probability=0.7, max_steps=TRUNCATION,
             base_seed=0, verbose=False):
    """Roll out the deterministic policy on `n_seeds` rainy episodes.

    Returns (successes, results) where results is a list of per-seed dicts.
    A seed counts as success iff the episode terminates with the dropoff reward.
    """
    model = PPO.load(model_path)
    env = gym.make(
        "Taxi-v4",
        is_rainy=True,
        rainy_probability=rainy_probability,
        fickle_passenger=False,
    )

    successes = 0
    results = []
    for i in range(n_seeds):
        seed = base_seed + i
        obs, _info = env.reset(seed=seed)
        delivered = False
        total_reward = 0.0
        steps = 0
        for steps in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            total_reward += reward
            if terminated:
                delivered = reward == DELIVERY_REWARD
                break
            if truncated:
                break
        successes += int(delivered)
        results.append({
            "seed": seed,
            "delivered": delivered,
            "steps": steps,
            "return": total_reward,
        })
        if verbose:
            flag = "OK " if delivered else "...."
            print(f"  seed {seed:>4}  {flag}  steps={steps:>3}  return={total_reward:>7.1f}")

    env.close()
    return successes, results


def report(model_path, n_seeds, rainy_probability, base_seed, verbose):
    if not os.path.exists(model_path):
        print(f"[skip] not found: {model_path}")
        return
    print(f"\n=== {model_path} ===")
    successes, results = evaluate(
        model_path, n_seeds=n_seeds, rainy_probability=rainy_probability,
        base_seed=base_seed, verbose=verbose,
    )
    avg_return = sum(r["return"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    print(f"success@{n_seeds} = {successes}/{n_seeds}"
          f"   (rainy_probability={rainy_probability}, seeds {base_seed}..{base_seed + n_seeds - 1})")
    print(f"avg_return={avg_return:.2f}  avg_steps={avg_steps:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None,
                        help="path to a single .zip to evaluate; default = both stock models")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--rainy_probability", type=float, default=0.7)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print per-seed outcome")
    args = parser.parse_args()

    targets = [args.model] if args.model else [OLD_GOOD]
    for model_path in targets:
        report(model_path, args.n_seeds, args.rainy_probability, args.base_seed, args.verbose)


if __name__ == "__main__":
    main()
