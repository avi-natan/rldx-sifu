import gymnasium as gym
import time


ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT / paddle up",
    3: "LEFT / paddle down",
    4: "RIGHTFIRE / up + fire",
    5: "LEFTFIRE / down + fire",
}


def wait(msg):
    input(f"\n{msg}\nPress Enter to continue...")


def run_action(env, action, steps=60, delay=0.03):
    print(f"\nRunning action {action}: {ACTION_NAMES[action]} for {steps} frames")

    total_reward = 0

    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"  reward at frame {i}: {reward}")

        if terminated or truncated:
            print("  episode ended, resetting...")
            obs, info = env.reset()

        time.sleep(delay)

    print(f"Total reward during this action block: {total_reward}")


def main():
    env = gym.make(
        "ALE/Pong-v5",
        render_mode="human",
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=0.0,  # no sticky actions for clean observation
    )

    obs, info = env.reset(seed=42)

    print("=== Pong Step-by-Step Action Test ===")
    print("Action meanings:")
    print(env.unwrapped.get_action_meanings())

    wait("A window should be open now. First, look at the paddles and the ball.")

    # FIRE to start the game
    wait("Now we will press FIRE to start/serve the ball.")
    run_action(env, action=1, steps=30)

    wait("Now test NOOP. The paddle should not intentionally move.")
    run_action(env, action=3, steps=90)

    wait("Now test action 2: RIGHT. In Pong this usually moves your paddle UP.")
    run_action(env, action=2, steps=90)

    wait("Now test action 3: LEFT. In Pong this usually moves your paddle DOWN.")
    run_action(env, action=3, steps=90)

    wait("Now test action 4: RIGHTFIRE. Usually UP + FIRE.")
    run_action(env, action=4, steps=90)

    wait("Now test action 5: LEFTFIRE. Usually DOWN + FIRE.")
    run_action(env, action=5, steps=90)

    wait("Final free random play for 10 seconds.")
    for step in range(600):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Random play reward at step {step}: {reward}")

        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(0.015)

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()