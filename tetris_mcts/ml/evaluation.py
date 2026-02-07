"""
Model Evaluation

Implements evaluation of trained models using MCTS on fixed seeds,
with optional trajectory visualization.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_EVAL_TRAJECTORY_MAX_FRAMES,
    EVAL_ONNX_FILENAME,
    EVAL_REPLAYS_FILENAME,
)
from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.weights import export_onnx

from tetris_core import (
    MCTSConfig,
    TetrisEnv,
    evaluate_model,
    EvalResult,
)

logger = structlog.get_logger()


class Evaluator:
    """Evaluates models using MCTS on fixed seeds."""

    def __init__(
        self,
        model: TetrisNet,
        checkpoint_dir: str | Path,
        num_simulations: int,
        max_moves: int,
        eval_seeds: list[int],
        eval_mcts_seed: int,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_simulations = num_simulations
        self.max_moves = max_moves
        self.eval_seeds = [int(s) for s in eval_seeds]
        self.eval_mcts_seed = eval_mcts_seed

    def evaluate(
        self, render_trajectory: bool = False
    ) -> tuple[EvalResult, Optional[list]]:
        """Evaluate current model using MCTS on fixed seeds.

        Args:
            render_trajectory: If True, render one game as images for visualization

        Returns:
            Tuple of (EvalResult, trajectory_frames) where trajectory_frames is
            a list of PIL Images if render_trajectory=True, else None
        """
        self.model.eval()

        # Export model to ONNX for Rust evaluation
        onnx_path = self.checkpoint_dir / EVAL_ONNX_FILENAME
        export_onnx(self.model, onnx_path)

        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export failed - file not created: {onnx_path}")

        # Create MCTS config for evaluation (temperature=0 enforced by evaluate_model)
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.num_simulations
        mcts_config.max_moves = self.max_moves
        mcts_config.seed = self.eval_mcts_seed

        replay_path = self.checkpoint_dir / EVAL_REPLAYS_FILENAME

        # Run evaluation in Rust with seeded environments.
        # If trajectory rendering is requested, save replays and render from that file
        # instead of running a second Python-side MCTS rollout.
        result = evaluate_model(
            model_path=str(onnx_path),
            seeds=[int(s) for s in self.eval_seeds],
            config=mcts_config,
            max_moves=self.max_moves,
            output_path=str(replay_path) if render_trajectory else None,
        )

        logger.info(
            "Evaluation complete",
            num_games=result.num_games,
            avg_attack=result.avg_attack,
            avg_lines=result.avg_lines,
            max_attack=result.max_attack,
            max_lines=result.max_lines,
            avg_moves=result.avg_moves,
            attack_per_piece=result.attack_per_piece,
            lines_per_piece=result.lines_per_piece,
        )

        # Optionally render one trajectory from saved replay (always first seed).
        trajectory_frames = None
        if render_trajectory and self.eval_seeds:
            try:
                trajectory_frames = self.render_first_replay(
                    replay_path=replay_path,
                    max_frames=DEFAULT_EVAL_TRAJECTORY_MAX_FRAMES,
                )
            except Exception as e:
                logger.warning("Failed to render evaluation trajectory", error=str(e))

        return result, trajectory_frames

    def render_first_replay(
        self,
        replay_path: Path,
        max_frames: int,
    ) -> list:
        """Render first game trajectory from a replay JSONL file."""
        with replay_path.open() as f:
            first_line = f.readline().strip()

        if first_line == "":
            return []

        replay = json.loads(first_line)
        return self.render_replay(
            replay=replay,
            max_frames=max_frames,
        )

    def render_replay(
        self,
        replay: dict,
        max_frames: int,
    ) -> list:
        """Render a replay dict (seed + moves) into trajectory frames."""
        from tetris_mcts.ml.visualization import render_board

        seed = int(replay["seed"])
        moves = replay["moves"]
        env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)
        frames = []
        total_attack = 0

        for move_idx, move in enumerate(moves):
            if env.game_over or len(frames) >= max_frames:
                break

            # Get current state for rendering
            board = np.array(env.get_board())
            board_colors = env.get_board_colors()

            piece = env.get_current_piece()
            piece_cells = None
            piece_type = None
            ghost_cells = None

            if piece:
                piece_cells = piece.get_cells()
                piece_type = piece.piece_type
                ghost = env.get_ghost_piece()
                if ghost:
                    ghost_cells = ghost.get_cells()

            frame = render_board(
                board=board,
                board_colors=board_colors,
                current_piece_cells=piece_cells,
                current_piece_type=piece_type,
                ghost_cells=ghost_cells,
                move_number=move_idx,
                attack=total_attack,
            )
            frames.append(frame)

            action = int(move["action"])
            attack = env.execute_action_index(action)
            if attack is None:
                raise ValueError(f"Invalid replay action index: {action}")
            total_attack += int(move["attack"])

        if len(frames) < max_frames:
            board = np.array(env.get_board())
            board_colors = env.get_board_colors()
            frame = render_board(
                board=board,
                board_colors=board_colors,
                move_number=len(frames),
                attack=total_attack,
                info_text="Final" if env.game_over else "",
            )
            frames.append(frame)

        return frames
