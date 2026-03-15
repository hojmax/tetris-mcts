"""Render a seeded NN-driven rollout as a GIF with per-frame root NN values.

The overlay uses each search root's exported `nn_value`, so it matches the
value shown in `make viz` for the same run/checkpoint configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import MCTSAgent, TetrisEnv
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FRAME_DURATION_MS,
)
from tetris_bot.scripts.utils.run_search_config import (
    build_mcts_config,
    default_checkpoint_path,
    default_model_path,
)
from tetris_bot.visualization import _capture_frame, create_trajectory_gif

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    run_dir: Path
    seed: int
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    model_path: Path | None = None
    add_noise: bool = False
    frame_duration: int = DEFAULT_GIF_FRAME_DURATION_MS


def resolve_output_path(run_dir: Path, seed: int, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return run_dir / "analysis" / "renders" / f"seed{seed}_nn_overlay.gif"


def main(args: ScriptArgs) -> None:
    run_dir = args.run_dir.resolve()
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None
        else default_model_path(run_dir).resolve()
    )
    checkpoint_path = (
        args.checkpoint_path.resolve()
        if args.checkpoint_path is not None
        else default_checkpoint_path(run_dir)
    )
    output_path = resolve_output_path(run_dir, args.seed, args.output_path).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = build_mcts_config(run_dir, checkpoint_path, track_value_history=False)
    agent = MCTSAgent(config)
    if not agent.load_model(str(model_path)):
        raise RuntimeError(f"Failed to load model: {model_path}")

    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, args.seed)
    playback = agent.play_game_with_trees(
        env,
        max_placements=config.max_placements,
        add_noise=args.add_noise,
    )
    if playback is None:
        raise RuntimeError("play_game_with_trees returned None")

    frames = []
    frame_values: list[float] = []
    total_attack = 0
    for step in playback.steps:
        root = step.tree.get_root()
        frame_value = float(root.nn_value)
        frame_values.append(frame_value)
        frames.append(
            _capture_frame(
                root.state.clone_state(),
                move_number=int(step.frame_index),
                attack=total_attack,
                value_pred=frame_value,
            )
        )
        total_attack += int(step.attack)

    if total_attack != int(playback.total_attack):
        raise RuntimeError(
            f"Rendered attack total {total_attack} did not match playback total {playback.total_attack}"
        )
    if not frames:
        raise RuntimeError("Playback contained no frames to render")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_trajectory_gif(frames, str(output_path), duration=args.frame_duration)

    logger.info(
        "Rendered seed rollout with NN overlay",
        run_dir=str(run_dir),
        seed=args.seed,
        model_path=str(model_path),
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        output_path=str(output_path),
        num_frames=len(frames),
        total_attack=int(playback.total_attack),
        nn_value_weight=float(config.nn_value_weight),
        first_frame_nn=round(frame_values[0], 6),
        last_frame_nn=round(frame_values[-1], 6),
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
