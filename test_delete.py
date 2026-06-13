import gymnasium as gym


if __name__ == '__main__':
    env = gym.make(
        "Taxi-v4",
        is_rainy=True,
        rainy_probability=0.8,
        fickle_passenger=False,
    )

    obs, info = env.reset(seed=42)

    print("obs =", obs)
    print("internal =", env.unwrapped.s)

    saved_state = env.unwrapped.s

    set1 = set()
    for i in range(50):
        env.unwrapped.s = saved_state
        obs2, reward, done, trunc, info = env.step(0)
        set1.add(obs2)
        print(obs2)

    print(set1)

    obs, info = env.reset(seed=42)

    print(obs)
    print(env.unwrapped.s)

    for i in range(10):
        obs2, reward, done, trunc, info = env.step(0)

        print(
            obs2,
            env.unwrapped.s,
            obs2 == env.unwrapped.s
        )