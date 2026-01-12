"""Deep Q-Network (DQN) for CartPole with replay and a target network."""

from __future__ import annotations

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    """Fixed-size replay buffer for experience tuples."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.rng = random.Random(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP used for the DQN value function."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(
    q_net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    action_space: gym.Space,
    device: torch.device,
    rng: random.Random,
) -> int:
    """Pick an action using epsilon-greedy exploration."""
    if rng.random() < epsilon:
        return int(action_space.sample())

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_t)
    return int(torch.argmax(q_values, dim=1).item())


def train_dqn(
    env: gym.Env,
    episodes: int,
    max_steps: int,
    batch_size: int,
    capacity: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    target_update_steps: int,
    seed: int,
) -> tuple[QNetwork, list[float], torch.device]:
    """Train a DQN agent on the provided environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    q_net = QNetwork(obs_dim, action_dim).to(device)
    target_net = QNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=alpha)
    loss_fn = nn.SmoothL1Loss()

    replay = ReplayBuffer(capacity, seed + 1)
    rng = random.Random(seed + 2)
    epsilon = epsilon_start
    total_steps = 0
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        total_reward = 0.0

        for _ in range(max_steps):
            action = select_action(q_net, obs, epsilon, env.action_space, device, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            total_steps += 1

            if len(replay) >= batch_size:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = q_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1, keepdim=True).values
                    targets = rewards_t + gamma * next_q * (1 - dones_t)

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            if total_steps % target_update_steps == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 50 == 0:
            recent = episode_rewards[-50:]
            avg_reward = sum(recent) / len(recent)
            print(
                f"Episode {episode + 1}: avg_reward={avg_reward:.1f} "
                f"epsilon={epsilon:.3f}"
            )

    return q_net, episode_rewards, device


def evaluate(
    env: gym.Env,
    q_net: QNetwork,
    episodes: int,
    seed: int,
    device: torch.device,
) -> None:
    """Evaluate a trained DQN agent using greedy actions."""
    rng = random.Random(seed + 123)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + episode)
        total_reward = 0.0

        while True:
            action = select_action(q_net, obs, 0.0, env.action_space, device, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Eval episode {episode + 1}: total_reward={total_reward:.1f}")


def main() -> None:
    """Run DQN training and evaluation with fixed hyperparameters."""
    train_episodes = 600
    eval_episodes = 5
    max_steps = 500
    batch_size = 64
    capacity = 50_000
    alpha = 1e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update_steps = 500
    seed = 0

    env = gym.make("CartPole-v1")
    q_net, _, device = train_dqn(
        env,
        episodes=train_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        capacity=capacity,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_steps=target_update_steps,
        seed=seed,
    )
    env.close()

    eval_env = gym.make("CartPole-v1", render_mode="human")
    evaluate(eval_env, q_net, episodes=eval_episodes, seed=seed, device=device)
    eval_env.close()


if __name__ == "__main__":
    main()
