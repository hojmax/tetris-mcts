"""
Self-Play Data Generation

Generates training data by playing games with MCTS.
"""

import logging
import time
from pathlib import Path

import numpy as np

from tetris_mcts.ml.network import (
    TetrisNet,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_PIECE_TYPES,
    MAX_MOVES,
)
from tetris_mcts.ml.data import TrainingExample
from tetris_mcts.ml.weights import export_onnx

from tetris_core import MCTSConfig, MCTSAgent

logger = logging.getLogger(__name__)


class SelfPlayGenerator:
    """Generates self-play training data using MCTS."""

    def __init__(
        self,
        model: TetrisNet,
        checkpoint_dir: str,
        num_simulations: int = 100,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.15,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def generate(self, num_games: int) -> tuple[list[TrainingExample], dict]:
        """Generate self-play data using MCTS.

        Args:
            num_games: Number of games to play

        Returns:
            Tuple of (examples, stats) where stats contains game statistics.
        """
        print(f"Generating {num_games} games...")

        # Export current model to ONNX for Rust inference
        onnx_path = self.checkpoint_dir / "selfplay.onnx"
        export_onnx(self.model, onnx_path)

        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export failed - file not created: {onnx_path}")

        # Configure MCTS
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.num_simulations
        mcts_config.temperature = self.temperature
        mcts_config.dirichlet_alpha = self.dirichlet_alpha
        mcts_config.dirichlet_epsilon = self.dirichlet_epsilon

        # Create agent and load model
        agent = MCTSAgent(mcts_config)
        if not agent.load_model(str(onnx_path)):
            raise RuntimeError(f"Failed to load model from {onnx_path}")

        start_time = time.time()

        # Generate games individually to collect stats
        examples = []
        total_attack = 0
        total_moves = 0
        failed_games = 0
        successful_games = 0

        # Aggregate game stats
        total_singles = 0
        total_doubles = 0
        total_triples = 0
        total_tetrises = 0
        total_tspin_minis = 0
        total_tspin_singles = 0
        total_tspin_doubles = 0
        total_tspin_triples = 0
        total_perfect_clears = 0
        total_back_to_backs = 0
        total_lines = 0
        max_combo = 0

        for game_idx in range(num_games):
            result = agent.play_game(max_moves=MAX_MOVES, add_noise=True)
            if result is not None:
                for ex in result.examples:
                    examples.append(rust_example_to_training_example(ex))
                total_attack += result.total_attack
                total_moves += result.num_moves
                successful_games += 1

                # Aggregate detailed stats
                game_stats = result.stats
                total_singles += game_stats.singles
                total_doubles += game_stats.doubles
                total_triples += game_stats.triples
                total_tetrises += game_stats.tetrises
                total_tspin_minis += game_stats.tspin_minis
                total_tspin_singles += game_stats.tspin_singles
                total_tspin_doubles += game_stats.tspin_doubles
                total_tspin_triples += game_stats.tspin_triples
                total_perfect_clears += game_stats.perfect_clears
                total_back_to_backs += game_stats.back_to_backs
                total_lines += game_stats.total_lines
                if game_stats.max_combo > max_combo:
                    max_combo = game_stats.max_combo
            else:
                failed_games += 1
                logger.warning(f"Game {game_idx} failed (returned None)")

        elapsed = time.time() - start_time

        if failed_games > 0:
            logger.warning(
                f"{failed_games}/{num_games} games failed during self-play "
                f"({100 * failed_games / num_games:.1f}%)"
            )

        # Compute statistics (use successful_games for averages)
        avg_attack = total_attack / successful_games if successful_games > 0 else 0
        avg_moves = total_moves / successful_games if successful_games > 0 else 0
        attack_per_move = total_attack / total_moves if total_moves > 0 else 0

        stats = {
            "selfplay/total_attack": total_attack,
            "selfplay/total_moves": total_moves,
            "selfplay/avg_attack": avg_attack,
            "selfplay/avg_moves": avg_moves,
            "selfplay/attack_per_move": attack_per_move,
            "selfplay/games_per_sec": num_games / max(0.1, elapsed),
            "selfplay/failed_games": failed_games,
            "selfplay/successful_games": successful_games,
            # Detailed move statistics
            "selfplay/total_lines": total_lines,
            "selfplay/singles": total_singles,
            "selfplay/doubles": total_doubles,
            "selfplay/triples": total_triples,
            "selfplay/tetrises": total_tetrises,
            "selfplay/tspin_minis": total_tspin_minis,
            "selfplay/tspin_singles": total_tspin_singles,
            "selfplay/tspin_doubles": total_tspin_doubles,
            "selfplay/tspin_triples": total_tspin_triples,
            "selfplay/perfect_clears": total_perfect_clears,
            "selfplay/back_to_backs": total_back_to_backs,
            "selfplay/max_combo": max_combo,
        }

        print(f"Generated {len(examples)} examples from {successful_games} games")
        if failed_games > 0:
            print(f"  Warning: {failed_games} games failed")
        print(f"Time: {elapsed:.1f}s ({stats['selfplay/games_per_sec']:.2f} games/sec)")
        print(f"Avg attack: {avg_attack:.2f}, Attack/move: {attack_per_move:.4f}")
        print(
            f"Lines: {total_lines} (T={total_tetrises}, TSD={total_tspin_doubles}, PC={total_perfect_clears})"
        )

        return examples, stats


def rust_example_to_training_example(rust_ex) -> TrainingExample:
    """Convert Rust TrainingExample to Python TrainingExample."""
    # Reshape board from flat list to 2D array
    board = np.array(rust_ex.board, dtype=np.uint8).reshape(BOARD_HEIGHT, BOARD_WIDTH)

    return TrainingExample(
        board=board.astype(bool),
        current_piece=rust_ex.current_piece,
        hold_piece=rust_ex.hold_piece if rust_ex.hold_piece < NUM_PIECE_TYPES else None,
        hold_available=rust_ex.hold_available,
        next_queue=list(rust_ex.next_queue),
        move_number=rust_ex.move_number,
        policy_target=np.array(rust_ex.policy, dtype=np.float32),
        value_target=rust_ex.value,
        action_mask=np.array(rust_ex.action_mask, dtype=bool),
    )
