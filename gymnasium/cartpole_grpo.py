"""Group Relative Policy Optimization (GRPO) for CartPole.

GRPO removes PPO's learned value function (the critic). Instead of asking a
value network "how good is this state?", it samples a *group* of episodes
with the current policy and scores each episode against the group: episodes
with above-average return get positive advantage, below-average get negative.
Every step in an episode shares that episode's advantage. This is the same
mechanism DeepSeek-R1 uses for LLMs, where the group is several sampled
answers to the same prompt. The PPO clipped surrogate objective is reused
unchanged; only the advantage source differs.
"""

from __future__ import annotations

import random
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from cartpole_ppo import ppo_clipped_loss


class PolicyNetwork(nn.Module):
    """Policy-only MLP producing action logits (no value head)."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_group_advantages(returns: np.ndarray) -> np.ndarray:
    """Normalize episode returns within a group.

    Args:
        returns: Total return of each episode in the group.

    Returns:
        Per-episode advantages: (return - group mean) / (group std + 1e-8),
        using the population standard deviation.
    """
    mean = returns.mean()
    std = returns.std()
    return (returns - mean) / (std + 1e-8)


def collect_episode(
    env: gym.Env, policy: PolicyNetwork, device: torch.device, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Roll out one full episode with the current policy.

    Args:
        env: Gymnasium environment instance.
        policy: Current policy network.
        device: Torch device for inference.
        seed: Seed for the episode reset.

    Returns:
        A tuple of (observations, actions, log_probs, total_reward).
    """
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    logp_list: list[float] = []
    total_reward = 0.0

    obs, _ = env.reset(seed=seed)
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        obs_list.append(obs)
        action_list.append(int(action.item()))
        logp_list.append(float(log_prob.item()))

        obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        total_reward += reward
        if terminated or truncated:
            break

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.int64),
        np.asarray(logp_list, dtype=np.float32),
        total_reward,
    )


# training code
def train_grpo(
    env: gym.Env,
    iterations: int,
    group_size: int,
    update_epochs: int,
    clip_coef: float,
    lr: float,
    seed: int,
) -> Tuple[PolicyNetwork, list[float], torch.device]:
    """Train a GRPO agent on the provided environment.

    Each iteration samples `group_size` complete episodes, converts their
    total returns into group-normalized advantages, and optimizes the PPO
    clipped surrogate over all collected steps for `update_epochs` epochs.
    No value network, no GAE, no per-step reward shaping.

    Args:
        env: Gymnasium environment instance.
        iterations: Number of group-sampling iterations.
        group_size: Episodes sampled per iteration.
        update_epochs: Optimization epochs per iteration.
        clip_coef: PPO clip coefficient.
        lr: Learning rate.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (trained_policy, per-iteration mean returns, device).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    mean_returns: list[float] = []

    for iteration in range(iterations):
        episodes = [
            collect_episode(env, policy, device, seed + iteration * group_size + i)
            for i in range(group_size)
        ]
        returns = np.asarray([ep[3] for ep in episodes], dtype=np.float32)
        advantages = compute_group_advantages(returns)
        mean_returns.append(float(returns.mean()))

        obs_t = torch.tensor(
            np.concatenate([ep[0] for ep in episodes]),
            dtype=torch.float32,
            device=device,
        )
        actions_t = torch.tensor(
            np.concatenate([ep[1] for ep in episodes]),
            dtype=torch.int64,
            device=device,
        )
        old_logp_t = torch.tensor(
            np.concatenate([ep[2] for ep in episodes]),
            dtype=torch.float32,
            device=device,
        )
        adv_t = torch.tensor(
            np.concatenate(
                [
                    np.full(len(ep[1]), advantages[i], dtype=np.float32)
                    for i, ep in enumerate(episodes)
                ]
            ),
            dtype=torch.float32,
            device=device,
        )

        for _ in range(update_epochs):
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(actions_t)
            loss = ppo_clipped_loss(new_logp, old_logp_t, adv_t, clip_coef)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if (iteration + 1) % 10 == 0:
            recent = mean_returns[-10:]
            avg_reward = sum(recent) / len(recent)
            print(
                f"Iteration {iteration + 1}: avg_group_return={avg_reward:.1f}"
            )

    return policy, mean_returns, device


# inference code
def evaluate(
    env: gym.Env,
    policy: PolicyNetwork,
    episodes: int,
    seed: int,
    device: torch.device,
) -> None:
    """Evaluate a trained GRPO policy using greedy actions.

    Args:
        env: Gymnasium environment instance.
        policy: Trained policy network.
        episodes: Number of evaluation episodes.
        seed: Random seed for evaluation rollouts.
        device: Torch device for inference.
    """
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + episode)
        total_reward = 0.0

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Eval episode {episode + 1}: total_reward={total_reward:.1f}")


def main() -> None:
    """Run GRPO training and evaluation with fixed hyperparameters."""
    iterations = 300
    group_size = 8
    update_epochs = 4
    clip_coef = 0.2
    lr = 1e-3
    seed = 0

    env = gym.make("CartPole-v1")
    policy, _, device = train_grpo(
        env,
        iterations=iterations,
        group_size=group_size,
        update_epochs=update_epochs,
        clip_coef=clip_coef,
        lr=lr,
        seed=seed,
    )
    env.close()

    eval_env = gym.make("CartPole-v1", render_mode="human")
    evaluate(eval_env, policy, episodes=5, seed=seed, device=device)
    eval_env.close()


if __name__ == "__main__":
    main()
