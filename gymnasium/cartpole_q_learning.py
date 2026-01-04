"""Q-learning for CartPole with discretized observations."""

import random
from bisect import bisect
from collections import defaultdict

import gymnasium as gym
import numpy as np

BOUNDS = [
    (-4.8, 4.8),
    (-3.0, 3.0),
    (-0.418, 0.418),
    (-3.5, 3.5),
]
BIN_COUNTS = [6, 6, 12, 12]


def build_bins() -> list[list[float]]:
    """Build discretization bins for each observation dimension.

    Returns:
        A list of per-dimension threshold lists used for discretization.
    """
    bins = []
    for (low, high), count in zip(BOUNDS, BIN_COUNTS):
        step = (high - low) / count
        bins.append([low + step * i for i in range(1, count)])
    return bins


def discretize(obs: list[float], bins: list[list[float]]) -> tuple[int, ...]:
    """Map a continuous observation to a discrete state index tuple.

    Args:
        obs: Continuous observation from the environment.
        bins: Thresholds for each observation dimension.

    Returns:
        A tuple of discrete indices representing the state.
    """
    state = []
    for value, edges, (low, high) in zip(obs, bins, BOUNDS):
        clipped = min(max(value, low), high)
        state.append(bisect(edges, clipped))
    return tuple(state)


def choose_action(q_values: list[float], epsilon: float) -> int:
    """Select an action using epsilon-greedy exploration.

    Args:
        q_values: Q-values for each action at the current state.
        epsilon: Probability of taking a random action.

    Returns:
        The selected action index.
    """
    if random.random() < epsilon:
        # Exploration
        return random.randrange(len(q_values))
    # Exploitation
    return int(np.argmax(q_values))


def train_q_learning(
    env: gym.Env,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    seed: int,
) -> tuple[dict[tuple[int, ...], list[float]], list[float]]:
    """Train a Q-learning agent on the provided environment.

    Args:
        env: Gymnasium environment instance.
        episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial epsilon for exploration.
        epsilon_end: Minimum epsilon for exploration.
        epsilon_decay: Multiplicative decay for epsilon after each episode.
        seed: Seed used for reproducibility.

    Returns:
        A tuple of (q_table, episode_rewards).
    """
    random.seed(seed)
    bins = build_bins()
    q_table: dict[tuple[int, ...], list[float]] = defaultdict(
        lambda: [0.0] * env.action_space.n
    )
    print("Starting Q-learning training...")
    episode_rewards = []
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = discretize(obs, bins)
        total_reward = 0.0

        while True:
            action = choose_action(q_table[state], epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins)
            old_q = q_table[state][action]
            max_future_q = 0.0 if done else max(q_table[next_state])
            q_table[state][action] = old_q + alpha * (
                reward + gamma * max_future_q - old_q
            )

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            avg_reward = sum(recent) / len(recent)
            print(
                f"Episode {episode + 1}: avg_reward={avg_reward:.1f} "
                f"epsilon={epsilon:.3f}"
            )

    return q_table, episode_rewards


def evaluate(
    env: gym.Env, q_table: dict[tuple[int, ...], list[float]], episodes: int, seed: int
) -> None:
    """Evaluate a trained Q-table with greedy actions.

    Args:
        env: Gymnasium environment instance.
        q_table: Trained Q-values by discrete state.
        episodes: Number of evaluation episodes to run.
        seed: Seed used for reproducibility.
    """
    bins = build_bins()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + episode)
        state = discretize(obs, bins)
        total_reward = 0.0

        while True:
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize(obs, bins)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Eval episode {episode + 1}: total_reward={total_reward:.1f}")


def main() -> None:
    """Run training and evaluation with fixed hyperparameters."""
    train_episodes = 2000
    eval_episodes = 5
    alpha = 0.1
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    seed = 0

    env = gym.make("CartPole-v1")
    q_table, _ = train_q_learning(
        env,
        episodes=train_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    env.close()

    eval_env = gym.make("CartPole-v1", render_mode="human")
    evaluate(eval_env, q_table, episodes=eval_episodes, seed=seed)
    eval_env.close()


if __name__ == "__main__":
    main()
