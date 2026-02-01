"""
Model Evaluation

Implements evaluation of trained models using MCTS on fixed seeds,
with optional trajectory visualization.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from tetris_mcts.ml.network import TetrisNet, MAX_MOVES
from tetris_mcts.ml.weights import export_onnx

from tetris_core import MCTSConfig, MCTSAgent, TetrisEnv, evaluate_model, EvalResult


class Evaluator:
    """Evaluates models using MCTS on fixed seeds."""

    def __init__(
        self,
        model: TetrisNet,
        checkpoint_dir: str,
        num_simulations: int = 100,
        eval_seeds: Optional[list[int]] = None,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_simulations = num_simulations
        self.eval_seeds = eval_seeds if eval_seeds is not None else list(range(20))

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
        onnx_path = self.checkpoint_dir / "eval.onnx"
        export_onnx(self.model, onnx_path)

        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export failed - file not created: {onnx_path}")

        # Create MCTS config for evaluation (temperature=0 enforced by evaluate_model)
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.num_simulations

        # Run evaluation in Rust with seeded environments
        result = evaluate_model(
            model_path=str(onnx_path),
            seeds=[int(s) for s in self.eval_seeds],
            config=mcts_config,
            max_moves=MAX_MOVES,
        )

        print(f"Evaluation ({result.num_games} games):")
        print(f"  Avg attack: {result.avg_attack:.1f}")
        print(f"  Max attack: {result.max_attack}")
        print(f"  Avg moves: {result.avg_moves:.1f}")
        print(f"  Attack/piece: {result.attack_per_piece:.3f}")

        # Optionally render one trajectory for visualization (always first seed)
        trajectory_frames = None
        if render_trajectory and self.eval_seeds:
            first_seed = self.eval_seeds[0]
            try:
                trajectory_frames = self.render_trajectory(
                    str(onnx_path), mcts_config, seed=first_seed
                )
            except Exception as e:
                print(f"  Warning: Failed to render trajectory: {e}")

        return result, trajectory_frames

    def render_trajectory(
        self,
        model_path: str,
        mcts_config: MCTSConfig,
        seed: int = 0,
        max_frames: int = 30,
    ) -> list:
        """Render a single evaluation game as images.

        Returns:
            List of PIL Images showing the game progression
        """
        from tetris_mcts.ml.visualization import render_board

        # Create agent and load model
        agent = MCTSAgent(mcts_config)
        if not agent.load_model(model_path):
            raise RuntimeError(f"Failed to load model from {model_path}")

        # Play one game with the seed
        env = TetrisEnv.with_seed(10, 20, seed)
        frames = []
        total_attack = 0

        for move_idx in range(MAX_MOVES):
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

            # Render frame
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

            # Get action from MCTS
            placements = env.get_possible_placements()
            if not placements:
                break

            # Use MCTS to select action
            result = agent.select_action(env, add_noise=False, move_number=move_idx)
            if result is None:
                break

            # Execute action
            attack = env.execute_action_index(result.action)
            if attack is None:
                break
            total_attack += attack

        # Add final frame
        if not env.game_over and len(frames) < max_frames:
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
