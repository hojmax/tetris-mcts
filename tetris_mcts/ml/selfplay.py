"""
Self-Play Data Generation for Tetris AlphaZero

Generates training data by playing games with either:
- Random policy (for initial data / Phase 2)
- Neural network policy (via MCTS / Phase 3+)
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
import time

from tetris_core import TetrisEnv, Placement, MCTSConfig, MCTSAgent
from tetris_mcts.ml.action_space import (
    NUM_ACTIONS,
    PLACEMENT_TO_ACTION,
    ACTION_TO_PLACEMENT,
    get_action_mask,
)
from tetris_mcts.ml.data import TrainingExample


# Game configuration
MAX_MOVES = 100
DROP_LAST_N = 10  # Drop last N moves from training (incomplete value targets)

# Board dimensions (should match Rust)
BOARD_WIDTH = 10
BOARD_HEIGHT = 20


@dataclass
class GameHistory:
    """Stores history of a single game for training data extraction."""

    states: list[dict] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    attacks: list[int] = field(default_factory=list)  # Attack gained at each step


def get_state_dict(env: TetrisEnv, move_number: int) -> dict:
    """Extract state as dictionary from environment."""
    # Get board state
    board = np.array(env.get_board(), dtype=np.uint8)

    # Get current piece
    current_piece = env.get_current_piece()
    current_piece_type = current_piece.piece_type if current_piece else 0

    # Get hold piece
    hold_piece = env.get_hold_piece()
    hold_piece_type = hold_piece.piece_type if hold_piece else None

    # Hold availability
    hold_available = not env.is_hold_used()

    # Get next queue (up to 5 pieces)
    next_pieces = env.get_next_pieces(5)
    next_queue = [p.piece_type for p in next_pieces]

    # Pad queue if needed
    while len(next_queue) < 5:
        next_queue.append(0)

    # Get action mask
    action_mask = get_action_mask(board, current_piece_type)

    return {
        "board": board,
        "current_piece": current_piece_type,
        "hold_piece": hold_piece_type,
        "hold_available": hold_available,
        "next_queue": next_queue,
        "move_number": move_number,
        "action_mask": action_mask,
    }


def placement_to_action_index(placement: Placement) -> Optional[int]:
    """Convert a Placement object to action index."""
    x = placement.piece.x
    y = placement.piece.y
    rot = placement.piece.rotation

    return PLACEMENT_TO_ACTION.get((x, y, rot))


def get_valid_action_indices(env: TetrisEnv) -> list[int]:
    """Get list of valid action indices from environment."""
    placements = env.get_all_placements()
    valid_indices = []
    for p in placements:
        idx = placement_to_action_index(p)
        if idx is not None:
            valid_indices.append(idx)
    return valid_indices


def random_policy(valid_actions: list[int]) -> tuple[int, np.ndarray]:
    """
    Generate a random policy over valid actions.

    Returns:
        action: Selected action index
        policy: Full policy vector (734,) with uniform prob over valid actions
    """
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)

    if not valid_actions:
        return 0, policy

    prob = 1.0 / len(valid_actions)
    for idx in valid_actions:
        policy[idx] = prob

    action = np.random.choice(valid_actions)
    return action, policy


def execute_action(env: TetrisEnv, action_idx: int) -> tuple[int, bool]:
    """
    Execute an action in the environment.

    Args:
        env: Tetris environment
        action_idx: Action index (0-733)

    Returns:
        attack: Attack gained from this action
        game_over: Whether the game ended
    """
    # Get the placement coordinates for this action
    x, y, rot = ACTION_TO_PLACEMENT[action_idx]

    # Find matching placement for proper T-spin detection
    placements = env.get_all_placements()
    for p in placements:
        if p.piece.x == x and p.piece.y == y and p.piece.rotation == rot:
            # Use execute_placement for proper T-spin detection
            attack = env.execute_placement(p)
            return int(attack), env.game_over

    # Fallback to direct placement
    attack = env.place_piece(x, y, rot)
    return int(attack), env.game_over


def play_game_random(
    seed: Optional[int] = None,
    max_moves: int = MAX_MOVES,
    verbose: bool = False,
) -> GameHistory:
    """
    Play a game with random policy.

    Args:
        seed: Random seed for reproducibility
        max_moves: Maximum moves per game
        verbose: Print progress

    Returns:
        GameHistory with states, policies, and attacks
    """
    if seed is not None:
        np.random.seed(seed)

    env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
    history = GameHistory()

    for move_idx in range(max_moves):
        if env.game_over:
            break

        # Get state
        state = get_state_dict(env, move_idx)
        history.states.append(state)

        # Get valid actions and select one
        valid_actions = get_valid_action_indices(env)

        if not valid_actions:
            if verbose:
                print(f"Move {move_idx}: No valid actions!")
            break

        action, policy = random_policy(valid_actions)
        history.policies.append(policy)

        # Execute action
        attack, _ = execute_action(env, action)
        history.attacks.append(attack)

        if verbose and attack > 0:
            print(f"Move {move_idx}: Action {action}, Attack +{attack}")

    return history


def play_game_with_policy(
    policy_fn: Callable[[dict, list[int], float], tuple[int, np.ndarray]],
    seed: Optional[int] = None,
    max_moves: int = MAX_MOVES,
    temperature: float = 1.0,
    temperature_drop_move: int = 15,
    temperature_final: float = 0.1,
) -> GameHistory:
    """
    Play a game with a policy function (e.g., MCTS-guided neural network).

    Args:
        policy_fn: Function(state_dict, valid_actions) -> (action, policy)
        seed: Random seed
        max_moves: Maximum moves
        temperature: Initial temperature for policy sampling
        temperature_drop_move: Move number to drop temperature
        temperature_final: Temperature after drop

    Returns:
        GameHistory
    """
    if seed is not None:
        np.random.seed(seed)

    env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
    history = GameHistory()

    for move_idx in range(max_moves):
        if env.game_over:
            break

        state = get_state_dict(env, move_idx)
        history.states.append(state)

        valid_actions = get_valid_action_indices(env)
        if not valid_actions:
            break

        # Get temperature for this move
        temp = temperature if move_idx < temperature_drop_move else temperature_final

        # Get policy from policy function
        action, policy = policy_fn(state, valid_actions, temp)
        history.policies.append(policy)

        # Execute
        attack, _ = execute_action(env, action)
        history.attacks.append(attack)

    return history


def compute_value_targets(attacks: list[int]) -> list[float]:
    """
    Compute cumulative attack values for each position.

    For a 100-move horizon, we predict total future attack from each state.
    No discounting - just sum of remaining attacks in the game.

    Args:
        attacks: List of attacks at each step

    Returns:
        List of value targets (same length as attacks)
    """
    n = len(attacks)
    values = [0.0] * n

    # Compute backwards: cumulative sum of future attacks
    cumulative = 0.0
    for i in range(n - 1, -1, -1):
        cumulative += attacks[i]
        values[i] = cumulative

    return values


def history_to_examples(
    history: GameHistory,
    drop_last_n: int = DROP_LAST_N,
) -> list[TrainingExample]:
    """
    Convert game history to training examples.

    Args:
        history: GameHistory from a played game
        drop_last_n: Number of final moves to drop (incomplete value targets)

    Returns:
        List of TrainingExample objects
    """
    if not history.states:
        return []

    # Compute value targets
    values = compute_value_targets(history.attacks)

    # Determine how many examples to use
    usable_moves = max(0, len(history.states) - drop_last_n)

    examples = []
    for i in range(usable_moves):
        state = history.states[i]
        examples.append(
            TrainingExample(
                board=state["board"].astype(bool),
                current_piece=state["current_piece"],
                hold_piece=state["hold_piece"],
                hold_available=state["hold_available"],
                next_queue=state["next_queue"],
                move_number=state["move_number"],
                policy_target=history.policies[i],
                value_target=values[i],
                action_mask=state["action_mask"].astype(bool),
            )
        )

    return examples


def generate_random_games(
    num_games: int,
    max_moves: int = MAX_MOVES,
    verbose: bool = True,
) -> list[TrainingExample]:
    """
    Generate training data from random self-play games.

    Args:
        num_games: Number of games to play
        max_moves: Max moves per game
        verbose: Print progress

    Returns:
        List of all training examples
    """
    all_examples = []
    total_attack = 0
    total_moves = 0

    start_time = time.time()

    for game_idx in range(num_games):
        history = play_game_random(max_moves=max_moves)
        examples = history_to_examples(history)
        all_examples.extend(examples)

        game_attack = sum(history.attacks)
        total_attack += game_attack
        total_moves += len(history.states)

        if verbose and (game_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_idx + 1) / elapsed
            print(
                f"Game {game_idx + 1}/{num_games}: "
                f"{len(examples)} examples, attack={game_attack}, "
                f"total={len(all_examples)} examples, "
                f"{games_per_sec:.1f} games/sec"
            )

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nGenerated {len(all_examples)} examples from {num_games} games")
        print(f"Avg attack per game: {total_attack / num_games:.1f}")
        print(f"Avg moves per game: {total_moves / num_games:.1f}")
        print(f"Time: {elapsed:.1f}s ({num_games / elapsed:.1f} games/sec)")

    return all_examples


@dataclass
class EvalMetrics:
    """Evaluation metrics from a set of games."""

    num_games: int = 0
    total_attack: int = 0
    max_attack: int = 0
    total_lines: int = 0
    total_moves: int = 0

    # Line clear breakdown
    clears_0: int = 0  # No clear
    clears_1: int = 0  # Single
    clears_2: int = 0  # Double
    clears_3: int = 0  # Triple
    clears_4: int = 0  # Tetris

    @property
    def avg_attack(self) -> float:
        return self.total_attack / max(1, self.num_games)

    @property
    def avg_lines(self) -> float:
        return self.total_lines / max(1, self.num_games)

    @property
    def avg_moves(self) -> float:
        return self.total_moves / max(1, self.num_games)

    @property
    def attack_per_piece(self) -> float:
        return self.total_attack / max(1, self.total_moves)

    def to_dict(self) -> dict:
        return {
            "num_games": self.num_games,
            "avg_attack": self.avg_attack,
            "max_attack": self.max_attack,
            "avg_lines": self.avg_lines,
            "avg_moves": self.avg_moves,
            "attack_per_piece": self.attack_per_piece,
        }


def rust_example_to_training_example(rust_ex) -> TrainingExample:
    """Convert Rust TrainingExample to Python TrainingExample."""
    # Reshape board from flat list to 2D array
    board = np.array(rust_ex.board, dtype=np.uint8).reshape(BOARD_HEIGHT, BOARD_WIDTH)

    return TrainingExample(
        board=board.astype(bool),
        current_piece=rust_ex.current_piece,
        hold_piece=rust_ex.hold_piece if rust_ex.hold_piece < 7 else None,
        hold_available=rust_ex.hold_available,
        next_queue=list(rust_ex.next_queue),
        move_number=rust_ex.move_number,
        policy_target=np.array(rust_ex.policy, dtype=np.float32),
        value_target=rust_ex.value,
        action_mask=np.array(rust_ex.action_mask, dtype=bool),
    )


def generate_mcts_games(
    model_path: str,
    num_games: int,
    mcts_config: Optional[MCTSConfig] = None,
    max_moves: int = MAX_MOVES,
    add_noise: bool = True,
    drop_last_n: int = DROP_LAST_N,
    verbose: bool = True,
) -> list[TrainingExample]:
    """
    Generate training data using MCTS with neural network (pure Rust).

    All neural network inference happens in Rust via tract-onnx.

    Args:
        model_path: Path to ONNX model file
        num_games: Number of games to play
        mcts_config: MCTS configuration
        max_moves: Max moves per game
        add_noise: Whether to add Dirichlet noise
        drop_last_n: Drop last N moves from each game
        verbose: Print progress

    Returns:
        List of training examples
    """
    if mcts_config is None:
        mcts_config = MCTSConfig()

    agent = MCTSAgent(mcts_config)

    if not agent.load_model(model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")

    start_time = time.time()

    # Generate all games in Rust
    rust_examples = agent.generate_games(num_games, max_moves, add_noise, drop_last_n)

    # Convert to Python TrainingExample objects
    examples = [rust_example_to_training_example(ex) for ex in rust_examples]

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nGenerated {len(examples)} examples from {num_games} games")
        print(f"Avg examples per game: {len(examples) / max(1, num_games):.1f}")
        print(f"Time: {elapsed:.1f}s ({num_games / max(0.1, elapsed):.2f} games/sec)")

    return examples


def evaluate_policy(
    policy_fn: Optional[Callable] = None,
    seeds: list[int] = list(range(20)),
    max_moves: int = MAX_MOVES,
    verbose: bool = True,
) -> EvalMetrics:
    """
    Evaluate a policy on fixed seeds.

    Args:
        policy_fn: Policy function or None for random
        seeds: List of seeds to use
        max_moves: Max moves per game
        verbose: Print progress

    Returns:
        EvalMetrics summary
    """
    metrics = EvalMetrics()

    for seed in seeds:
        if policy_fn is None:
            history = play_game_random(seed=seed, max_moves=max_moves)
        else:
            history = play_game_with_policy(policy_fn, seed=seed, max_moves=max_moves)

        game_attack = sum(history.attacks)
        metrics.num_games += 1
        metrics.total_attack += game_attack
        metrics.max_attack = max(metrics.max_attack, game_attack)
        metrics.total_moves += len(history.states)

    if verbose:
        print(f"Evaluation ({metrics.num_games} games):")
        print(f"  Avg attack: {metrics.avg_attack:.1f}")
        print(f"  Max attack: {metrics.max_attack}")
        print(f"  Avg moves: {metrics.avg_moves:.1f}")
        print(f"  Attack/piece: {metrics.attack_per_piece:.3f}")

    return metrics


if __name__ == "__main__":
    print("Testing self-play data generation...")
    print()

    # Test single game
    print("Playing single random game...")
    history = play_game_random(seed=42, max_moves=100, verbose=True)
    print(
        f"Game finished: {len(history.states)} moves, {sum(history.attacks)} total attack"
    )
    print()

    # Convert to examples
    examples = history_to_examples(history)
    print(f"Generated {len(examples)} training examples")
    if examples:
        ex = examples[0]
        print(
            f"First example: board={ex.board.shape}, policy={ex.policy_target.shape}, value={ex.value_target:.2f}"
        )
    print()

    # Generate batch of games
    print("Generating 20 random games...")
    all_examples = generate_random_games(num_games=20, verbose=True)
    print()

    # Evaluate random policy
    print("Evaluating random policy on 10 seeds...")
    metrics = evaluate_policy(seeds=list(range(10)))
