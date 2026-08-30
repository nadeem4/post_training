"""Unit tests for pure functions in cartpole_grpo."""
import gymnasium as gym
import numpy as np

from cartpole_grpo import compute_group_advantages, train_grpo


def test_group_advantages_are_zero_mean_unit_std() -> None:
    returns = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    adv = compute_group_advantages(returns)
    # Population std of [1,2,3] is sqrt(2/3) ~= 0.8165.
    np.testing.assert_allclose(adv, [-1.2247, 0.0, 1.2247], atol=1e-3)


def test_group_advantages_identical_returns_give_zero() -> None:
    # All episodes equal -> no learning signal, not a NaN explosion.
    returns = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    adv = compute_group_advantages(returns)
    np.testing.assert_allclose(adv, [0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_group_advantages_best_episode_is_positive() -> None:
    returns = np.array([10.0, 200.0], dtype=np.float32)
    adv = compute_group_advantages(returns)
    assert adv[0] < 0 < adv[1]


def test_train_grpo_smoke_improves_or_runs() -> None:
    env = gym.make("CartPole-v1")
    policy, mean_returns, _ = train_grpo(
        env,
        iterations=10,
        group_size=4,
        update_epochs=2,
        clip_coef=0.2,
        lr=1e-3,
        seed=0,
    )
    env.close()
    assert len(mean_returns) == 10
    assert policy is not None
