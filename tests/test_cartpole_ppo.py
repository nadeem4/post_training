"""Unit tests for pure functions in cartpole_ppo."""
import numpy as np

from cartpole_ppo import compute_gae


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
