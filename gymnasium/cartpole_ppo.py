"""Proximal Policy Optimization (PPO) for CartPole."""

from __future__ import annotations

import random
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    """Actor-critic network with shared trunk and separate policy/value heads.

    The network builds a shared MLP for feature extraction and then uses
    independent linear heads to produce policy logits and state-value estimates.
    This layout reduces parameters while keeping the policy and value functions
    coupled through common features.

    Args:
        obs_dim: Size of the observation vector.
        action_dim: Number of discrete actions.

    Attributes:
        shared: Feature extractor applied to observations.
        policy_head: Linear layer producing action logits.
        value_head: Linear layer producing a scalar value estimate.
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """Initialize the actor-critic network.

        Args:
            obs_dim: Size of the observation vector.
            action_dim: Number of discrete actions.
        """
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and value estimates.

        Args:
            x: Batch of observations with shape (batch, obs_dim).

        Returns:
            A tuple of (policy_logits, values) where policy_logits has shape
            (batch, action_dim) and values has shape (batch, 1).
        """
        features = self.shared(x)
        return self.policy_head(features), self.value_head(features)


def select_action(
    model: ActorCritic, obs: np.ndarray, device: torch.device
) -> Tuple[int, float, float]:
    """Sample an action from the current policy.

    Args:
        model: Actor-critic network used to compute policy logits and values.
        obs: Current environment observation.
        device: Torch device for inference.

    Returns:
        A tuple of (action, log_prob, value) where action is an int, log_prob is
        the log-probability of the sampled action, and value is the state value.
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


def evaluate_actions(
    model: ActorCritic, obs_t: torch.Tensor, actions_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute log-probs, entropy, and values for a batch.

    Args:
        model: Actor-critic network used for evaluation.
        obs_t: Batch of observations.
        actions_t: Batch of actions taken.

    Returns:
        A tuple of (log_probs, entropy, values) for the given batch.
    """
    logits, values = model(obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions_t)
    entropy = dist.entropy()
    return log_probs, entropy, values.squeeze(-1)


def ppo_clipped_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    """Compute the PPO clipped surrogate policy loss.

    Args:
        new_logp: Log-probs of actions under the current policy.
        old_logp: Log-probs of the same actions under the rollout policy.
        advantages: Advantage estimates for each action.
        clip_coef: Clipping range epsilon.

    Returns:
        Scalar policy loss (negative clipped surrogate objective).
    """
    ratios = (new_logp - old_logp).exp()
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -torch.min(unclipped, clipped).mean()


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE-Lambda advantages and returns.

    Args:
        rewards: Reward sequence for the rollout.
        dones: Done flags indicating episode termination at each step.
        values: Value predictions for each step.
        next_value: Value prediction for the next state after the rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        A tuple of (advantages, returns) for each step in the rollout.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[t]
        if t == len(rewards) - 1:
            next_values = next_value
        else:
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# training code
def train_ppo(
    env: gym.Env,
    total_timesteps: int,
    rollout_steps: int,
    update_epochs: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    lr: float,
    seed: int,
) -> Tuple[ActorCritic, list[float], torch.device]:
    """Train PPO on the provided environment.

    This routine alternates between collecting on-policy rollouts and updating
    the actor-critic with the PPO clipped objective. Each rollout stores
    observations, actions, rewards, dones, and value predictions. Generalized
    Advantage Estimation (GAE) computes advantages/returns, which are then
    normalized and used to optimize the policy and value heads over multiple
    epochs of minibatches. Periodic logging reports recent episodic returns.

    Args:
        env: Gymnasium environment instance.
        total_timesteps: Total number of environment steps to collect.
        rollout_steps: Number of steps to collect per rollout.
        update_epochs: Number of optimization epochs per rollout.
        minibatch_size: Minibatch size for PPO updates.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_coef: PPO clip coefficient.
        vf_coef: Value loss coefficient.
        ent_coef: Entropy bonus coefficient.
        max_grad_norm: Gradient clipping threshold.
        lr: Learning rate.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (trained_model, episode_returns, device).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    episode_returns: list[float] = []
    episode_return = 0.0
    timesteps = 0
    update_count = 0

    while timesteps < total_timesteps:
        obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        actions_buf = np.zeros(rollout_steps, dtype=np.int64)
        logp_buf = np.zeros(rollout_steps, dtype=np.float32)
        rewards_buf = np.zeros(rollout_steps, dtype=np.float32)
        dones_buf = np.zeros(rollout_steps, dtype=np.float32)
        values_buf = np.zeros(rollout_steps, dtype=np.float32)

        for step in range(rollout_steps):
            obs_buf[step] = obs
            action, log_prob, value = select_action(model, obs, device)
            actions_buf[step] = action
            logp_buf[step] = log_prob
            values_buf[step] = value

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards_buf[step] = reward
            dones_buf[step] = float(done)
            episode_return += reward
            timesteps += 1
            obs = next_obs

            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset(seed=seed + timesteps)
                env.action_space.seed(seed + timesteps)

            if timesteps >= total_timesteps:
                break

        steps_collected = step + 1
        obs_buf = obs_buf[:steps_collected]
        actions_buf = actions_buf[:steps_collected]
        logp_buf = logp_buf[:steps_collected]
        rewards_buf = rewards_buf[:steps_collected]
        dones_buf = dones_buf[:steps_collected]
        values_buf = values_buf[:steps_collected]

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value_t = model(obs_t)
            next_value = float(next_value_t.item())

        advantages, returns = compute_gae(
            rewards_buf,
            dones_buf,
            values_buf,
            next_value,
            gamma,
            gae_lambda,
        )

        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        batch_size = len(obs_buf)
        indices = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = advantages_t[mb_idx]

                new_logp, entropy, values = evaluate_actions(
                    model, mb_obs, mb_actions
                )
                policy_loss = ppo_clipped_loss(new_logp, mb_old_logp, mb_adv, clip_coef)
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        update_count += 1
        if episode_returns and update_count % 10 == 0:
            recent = episode_returns[-10:]
            avg_reward = sum(recent) / len(recent)
            print(
                f"Update {update_count}: avg_reward={avg_reward:.1f} "
                f"timesteps={timesteps}"
            )

    return model, episode_returns, device


# inference code
def evaluate(
    env: gym.Env, model: ActorCritic, episodes: int, seed: int, device: torch.device
) -> None:
    """Evaluate a trained PPO policy using greedy actions.

    Args:
        env: Gymnasium environment instance.
        model: Trained actor-critic network.
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
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Eval episode {episode + 1}: total_reward={total_reward:.1f}")


def main() -> None:
    """Run PPO training and evaluation with fixed hyperparameters."""
    total_timesteps = 200_000
    rollout_steps = 1024
    update_epochs = 10
    minibatch_size = 256
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    lr = 3e-4
    seed = 0

    env = gym.make("CartPole-v1")
    model, _, device = train_ppo(
        env,
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        lr=lr,
        seed=seed,
    )
    env.close()

    eval_env = gym.make("CartPole-v1", render_mode="human")
    evaluate(eval_env, model, episodes=5, seed=seed, device=device)
    eval_env.close()


if __name__ == "__main__":
    main()
