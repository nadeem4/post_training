import os
import time

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1", render_mode="human")

    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Environment Metadata: {env.metadata}")
    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {steps}: action={action}, observation={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    env.close()
    print(f"Episode finished: steps={steps} total_reward={total_reward}")


if __name__ == "__main__":
    main()
