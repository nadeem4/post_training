"""Toy Q-learning on a simplified chess endgame (KQ vs K).

This example uses python-chess to handle legal moves. The agent plays White
(King + Queen) and trains against a random Black king. The state is the FEN
string, and actions are UCI move strings.
"""

from __future__ import annotations

import random
from collections import defaultdict

import chess


def random_kqk_position(rng: random.Random) -> chess.Board:
    """Sample a legal KQ vs K position with White to move."""
    squares = list(chess.SQUARES)
    while True:
        wk, wq, bk = rng.sample(squares, 3)
        if chess.square_distance(wk, bk) <= 1:
            continue

        board = chess.Board.empty()
        board.clear_stack()
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        board.halfmove_clock = 0
        board.fullmove_number = 1

        if not board.is_valid():
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue
        return board


def select_action(
    q_table: dict[str, dict[str, float]],
    state: str,
    legal_moves: list[chess.Move],
    epsilon: float,
    rng: random.Random,
) -> chess.Move:
    """Pick an action with epsilon-greedy exploration."""
    if rng.random() < epsilon:
        return rng.choice(legal_moves)

    q_state = q_table.get(state, {})
    best_score = None
    best_moves: list[chess.Move] = []
    for move in legal_moves:
        score = q_state.get(move.uci(), 0.0)
        if best_score is None or score > best_score + 1e-9:
            best_score = score
            best_moves = [move]
        elif abs(score - best_score) <= 1e-9:
            best_moves.append(move)
    return rng.choice(best_moves)


def update_q(
    q_table: dict[str, dict[str, float]],
    state: str,
    action: chess.Move,
    reward: float,
    next_state: str | None,
    alpha: float,
    gamma: float,
) -> None:
    """Apply the Q-learning update rule for a single transition."""
    action_key = action.uci()
    state_actions = q_table.setdefault(state, {})
    old_q = state_actions.get(action_key, 0.0)

    max_future = 0.0
    if next_state is not None:
        next_actions = q_table.get(next_state, {})
        if next_actions:
            max_future = max(next_actions.values())

    state_actions[action_key] = old_q + alpha * (reward + gamma * max_future - old_q)


def train_q_learning(
    episodes: int,
    max_steps: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    step_penalty: float,
    stalemate_penalty: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Train Q-values for White in a KQ vs K endgame."""
    rng = random.Random(seed)
    q_table: dict[str, dict[str, float]] = defaultdict(dict)
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(episodes):
        board = random_kqk_position(rng)
        total_reward = 0.0

        for _ in range(max_steps):
            state = board.fen()
            legal_moves = list(board.legal_moves)
            action = select_action(q_table, state, legal_moves, epsilon, rng)

            board.push(action)
            if board.is_checkmate():
                reward = 1.0
                update_q(q_table, state, action, reward, None, alpha, gamma)
                total_reward += reward
                break
            if board.is_stalemate():
                reward = stalemate_penalty
                update_q(q_table, state, action, reward, None, alpha, gamma)
                total_reward += reward
                break

            black_move = rng.choice(list(board.legal_moves))
            board.push(black_move)

            if board.is_stalemate():
                reward = stalemate_penalty
                next_state = None
                done = True
            else:
                reward = -step_penalty
                next_state = board.fen()
                done = False

            update_q(q_table, state, action, reward, next_state, alpha, gamma)
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 200 == 0:
            recent = episode_rewards[-200:]
            avg_reward = sum(recent) / len(recent)
            print(
                f"Episode {episode + 1}: avg_reward={avg_reward:.3f} "
                f"epsilon={epsilon:.3f}"
            )

    return q_table


def run_greedy_episode(
    q_table: dict[str, dict[str, float]],
    max_steps: int,
    rng: random.Random,
) -> tuple[str, int]:
    """Evaluate a greedy policy from a random KQ vs K position."""
    board = random_kqk_position(rng)
    ply = 0

    while ply < max_steps:
        state = board.fen()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        action = select_action(q_table, state, legal_moves, 0.0, rng)
        board.push(action)
        ply += 1

        if board.is_checkmate():
            return "checkmate", ply
        if board.is_stalemate():
            return "stalemate", ply

        black_move = rng.choice(list(board.legal_moves))
        board.push(black_move)
        ply += 1

        if board.is_stalemate():
            return "stalemate", ply

    return "timeout", ply


def main() -> None:
    """Run training and a short greedy evaluation."""
    episodes = 2000
    max_steps = 80
    alpha = 0.2
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    step_penalty = 0.02
    stalemate_penalty = -0.3
    seed = 7

    q_table = train_q_learning(
        episodes=episodes,
        max_steps=max_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        step_penalty=step_penalty,
        stalemate_penalty=stalemate_penalty,
        seed=seed,
    )

    rng = random.Random(seed + 100)
    eval_episodes = 5
    for episode in range(eval_episodes):
        result, ply = run_greedy_episode(q_table, max_steps, rng)
        print(f"Eval {episode + 1}: result={result} ply={ply}")


if __name__ == "__main__":
    main()
