"""Unit tests for pure functions in cartpole_ppo and a training smoke test."""
import gymnasium as gym
import numpy as np
import torch

from cartpole_ppo import compute_gae, ppo_clipped_loss, train_ppo


def test_gae_no_discount_sums_future_rewards() -> None:
    # gamma=1, lambda=1, zero values: advantage_t = sum of future rewards.
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    dones = np.array([0.0, 0.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards, dones, values, next_value=0.0, gamma=1.0, gae_lambda=1.0
    )
    np.testing.assert_allclose(advantages, [2.0, 1.0])
    np.testing.assert_allclose(returns, [2.0, 1.0])


def test_gae_done_flag_cuts_bootstrap() -> None:
    # Episode ends at t=0, so t=0 must not see t=1's reward or value.
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    dones = np.array([1.0, 0.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    advantages, _ = compute_gae(
        rewards, dones, values, next_value=0.0, gamma=1.0, gae_lambda=1.0
    )
    np.testing.assert_allclose(advantages, [1.0, 1.0])


def test_gae_returns_equal_advantages_plus_values() -> None:
    rewards = np.array([0.5, -0.5, 2.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards, dones, values, next_value=0.4, gamma=0.99, gae_lambda=0.95
    )
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-6)


def test_clipped_loss_caps_large_ratio_on_positive_advantage() -> None:
    # ratio = exp(new - old) = 2.0, clip 0.2 -> objective uses 1.2 * adv.
    new_logp = torch.tensor([np.log(2.0)], dtype=torch.float32)
    old_logp = torch.tensor([0.0], dtype=torch.float32)
    adv = torch.tensor([1.0], dtype=torch.float32)
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip_coef=0.2)
    assert torch.isclose(loss, torch.tensor(-1.2), atol=1e-6)


def test_clipped_loss_unclipped_when_ratio_inside_band() -> None:
    # ratio = 1.0 -> objective is exactly adv.
    new_logp = torch.tensor([0.0], dtype=torch.float32)
    old_logp = torch.tensor([0.0], dtype=torch.float32)
    adv = torch.tensor([3.0], dtype=torch.float32)
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip_coef=0.2)
    assert torch.isclose(loss, torch.tensor(-3.0), atol=1e-6)


def test_clipped_loss_pessimistic_on_negative_advantage() -> None:
    # ratio = 0.5 with adv = -1: min(0.5*-1, clamp(0.5,0.8,1.2)*-1) = -0.8
    # -> loss = 0.8 (the *worse* of the two objectives is kept).
    new_logp = torch.tensor([np.log(0.5)], dtype=torch.float32)
    old_logp = torch.tensor([0.0], dtype=torch.float32)
    adv = torch.tensor([-1.0], dtype=torch.float32)
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip_coef=0.2)
    assert torch.isclose(loss, torch.tensor(0.8), atol=1e-6)


def test_train_ppo_smoke_runs_and_returns_episodes() -> None:
    env = gym.make("CartPole-v1")
    model, episode_returns, _ = train_ppo(
        env,
        total_timesteps=2048,
        rollout_steps=256,
        update_epochs=2,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=3e-4,
        seed=0,
    )
    env.close()
    assert len(episode_returns) > 0
    assert model is not None
