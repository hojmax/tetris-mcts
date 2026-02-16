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
    EVAL_ONNX_FILENAME,
    EVAL_REPLAYS_FILENAME,
    QUEUE_SIZE,
)
from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.weights import export_onnx, export_split_models, split_model_paths

from tetris_core import (
    MCTSConfig,
    TetrisEnv,
    evaluate_model,
    EvalResult,
)

logger = structlog.get_logger()


def assert_rust_inference_artifacts(onnx_path: Path) -> None:
    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    required_paths = [onnx_path, conv_path, heads_path, fc_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError(
            "Model export incomplete for Rust inference; missing artifacts: "
            + ", ".join(missing_paths)
        )


class Evaluator:
    """Evaluates models using MCTS on fixed seeds."""

    def __init__(
        self,
        model: TetrisNet,
        checkpoint_dir: str | Path,
        num_simulations: int,
        max_placements: int,
        overhang_penalty_weight: float,
        eval_seeds: list[int],
        eval_mcts_seed: int,
        nn_value_weight: float,
        q_scale: float | None,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_simulations = num_simulations
        self.max_placements = max_placements
        self.overhang_penalty_weight = overhang_penalty_weight
        self.eval_seeds = [int(s) for s in eval_seeds]
        self.eval_mcts_seed = eval_mcts_seed
        self.nn_value_weight = nn_value_weight
        self.q_scale = q_scale

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

        # Export model to ONNX for Rust evaluation (full + split for cached inference)
        onnx_path = self.checkpoint_dir / EVAL_ONNX_FILENAME
        full_export_ok = export_onnx(self.model, onnx_path)
        split_export_ok = export_split_models(self.model, onnx_path)
        if not full_export_ok:
            raise RuntimeError("ONNX export failed due to missing dependencies")
        if not split_export_ok:
            raise RuntimeError("Split-model export failed due to missing dependencies")
        assert_rust_inference_artifacts(onnx_path)

        # Create MCTS config for evaluation (temperature=0 enforced by evaluate_model)
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.num_simulations
        mcts_config.max_placements = self.max_placements
        mcts_config.overhang_penalty_weight = self.overhang_penalty_weight
        mcts_config.visit_sampling_epsilon = 0.0
        mcts_config.seed = self.eval_mcts_seed
        mcts_config.nn_value_weight = self.nn_value_weight
        mcts_config.q_scale = self.q_scale

        replay_path = self.checkpoint_dir / EVAL_REPLAYS_FILENAME

        # Run evaluation in Rust with seeded environments.
        # If trajectory rendering is requested, save replays and render from that file
        # instead of running a second Python-side MCTS rollout.
        result = evaluate_model(
            model_path=str(onnx_path),
            seeds=[int(s) for s in self.eval_seeds],
            config=mcts_config,
            max_placements=self.max_placements,
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
            trajectory_frames = self.render_first_replay(
                replay_path=replay_path,
            )

        return result, trajectory_frames

    def render_first_replay(
        self,
        replay_path: Path,
    ) -> list:
        """Render first game trajectory from a replay JSONL file."""
        with replay_path.open() as f:
            first_line = f.readline().strip()

        if first_line == "":
            return []

        replay = json.loads(first_line)
        return self.render_replay(
            replay=replay,
        )

    def render_replay(
        self,
        replay: dict,
    ) -> list:
        """Render a replay dict (seed + moves) into trajectory frames."""
        from tetris_mcts.ml.visualization import render_board

        seed = int(replay["seed"])
        moves = replay["moves"]
        env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)
        frames = []
        total_attack = 0

        for move_idx, move in enumerate(moves):
            if env.game_over:
                break

            # Get current state for rendering
            board = np.array(env.get_board())
            board_piece_types = env.get_board_piece_types()
            piece = env.get_current_piece()
            hold_piece = env.get_hold_piece()
            queue_piece_types = env.get_queue(QUEUE_SIZE)
            can_hold = not env.is_hold_used()
            piece_cells = piece.get_cells() if piece else None
            piece_type = piece.piece_type if piece else None
            ghost = env.get_ghost_piece()
            ghost_cells = ghost.get_cells() if ghost else None

            frame = render_board(
                board=board,
                board_piece_types=board_piece_types,
                current_piece_cells=piece_cells,
                current_piece_type=piece_type,
                ghost_cells=ghost_cells,
                move_number=move_idx,
                attack=total_attack,
                can_hold=can_hold,
                combo=env.combo,
                back_to_back=env.back_to_back,
                show_piece_info=True,
                hold_piece_type=hold_piece.piece_type if hold_piece else None,
                queue_piece_types=list(queue_piece_types),
            )
            frames.append(frame)

            action = int(move["action"])
            attack = env.execute_action_index(action)
            if attack is None:
                raise ValueError(f"Invalid replay action index: {action}")
            total_attack += int(move["attack"])

        board = np.array(env.get_board())
        board_piece_types = env.get_board_piece_types()
        piece = env.get_current_piece()
        hold_piece = env.get_hold_piece()
        queue_piece_types = env.get_queue(QUEUE_SIZE)
        can_hold = not env.is_hold_used()
        piece_cells = piece.get_cells() if piece else None
        piece_type = piece.piece_type if piece else None
        ghost = env.get_ghost_piece()
        ghost_cells = ghost.get_cells() if ghost else None
        frame = render_board(
            board=board,
            board_piece_types=board_piece_types,
            current_piece_cells=piece_cells,
            current_piece_type=piece_type,
            ghost_cells=ghost_cells,
            move_number=len(frames),
            attack=total_attack,
            can_hold=can_hold,
            combo=env.combo,
            back_to_back=env.back_to_back,
            is_terminal=env.game_over,
            show_piece_info=True,
            hold_piece_type=hold_piece.piece_type if hold_piece else None,
            queue_piece_types=list(queue_piece_types),
        )
        frames.append(frame)

        return frames
