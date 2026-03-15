"""Render plain and move-ghost seeded rollout GIFs with per-frame NN values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import (
    MCTSAgent,
    TetrisEnv,
    debug_encode_state,
    debug_get_action_mask,
    debug_predict_masked_from_tensors,
)
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FRAME_DURATION_MS,
    NUM_ACTIONS,
)
from tetris_bot.scripts.utils.run_search_config import (
    build_mcts_config,
    default_checkpoint_path,
    default_model_path,
)
from tetris_bot.visualization import (
    PredictedMoveOverlay,
    _capture_frame,
    create_trajectory_gif,
)

logger = structlog.get_logger()
HOLD_ACTION_INDEX = NUM_ACTIONS - 1


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


def resolve_move_overlay_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_top3_moves{output_path.suffix}")


def compute_raw_nn_value(
    model_path: Path,
    env: TetrisEnv,
    max_placements: int,
) -> float | None:
    action_mask = debug_get_action_mask(env)
    if not any(action_mask):
        return None

    board_flat, aux_flat = debug_encode_state(env, max_placements)
    _, value = debug_predict_masked_from_tensors(
        str(model_path),
        list(board_flat),
        list(aux_flat),
        action_mask,
    )
    return float(value)


def build_predicted_move_overlays(
    env: TetrisEnv,
    valid_actions: list[int],
    action_priors: list[float],
    limit: int = 3,
) -> list[PredictedMoveOverlay]:
    if len(valid_actions) != len(action_priors):
        raise ValueError(
            "valid_actions and action_priors must have the same length "
            f"(got {len(valid_actions)} vs {len(action_priors)})"
        )

    current_piece = env.get_current_piece()
    placements_by_action = {
        int(placement.action_index): placement
        for placement in env.get_possible_placements()
    }
    top_actions = sorted(
        zip(valid_actions, action_priors),
        key=lambda item: (-item[1], item[0]),
    )[:limit]

    overlays: list[PredictedMoveOverlay] = []
    for rank, (action_idx, probability) in enumerate(top_actions, start=1):
        action_idx = int(action_idx)
        if action_idx == HOLD_ACTION_INDEX:
            if current_piece is None:
                continue
            overlays.append(
                PredictedMoveOverlay(
                    probability=float(probability),
                    piece_type=int(current_piece.piece_type),
                    cells=tuple((int(x), int(y)) for x, y in current_piece.get_cells()),
                    rank=rank,
                    is_hold=True,
                )
            )
            continue

        placement = placements_by_action.get(action_idx)
        if placement is None:
            raise RuntimeError(
                f"Missing placement for valid action {action_idx} while rendering NN overlay"
            )
        overlays.append(
            PredictedMoveOverlay(
                probability=float(probability),
                piece_type=int(placement.piece.piece_type),
                cells=tuple((int(x), int(y)) for x, y in placement.piece.get_cells()),
                rank=rank,
            )
        )

    return overlays


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

    move_overlay_output_path = resolve_move_overlay_output_path(output_path)
    plain_frames = []
    move_overlay_frames = []
    frame_values: list[float] = []
    total_attack = 0
    for step in playback.steps:
        root = step.tree.get_root()
        frame_value = float(root.nn_value)
        frame_values.append(frame_value)
        predicted_move_overlays = build_predicted_move_overlays(
            env,
            list(root.valid_actions),
            list(root.action_priors),
        )
        plain_frames.append(
            _capture_frame(
                env,
                placement_number=int(step.placement_count),
                attack=total_attack,
                value_pred=frame_value,
            )
        )
        move_overlay_frames.append(
            _capture_frame(
                env,
                placement_number=int(step.placement_count),
                attack=total_attack,
                value_pred=frame_value,
                predicted_move_overlays=predicted_move_overlays,
                show_ghost_piece=False,
            )
        )
        if env.placement_count != int(step.placement_count):
            raise RuntimeError(
                f"Replay placement count {env.placement_count} did not match step {step.placement_count}"
            )
        attack = env.execute_action_index(int(step.selected_action))
        if attack is None:
            raise RuntimeError(f"Replay failed for action {step.selected_action}")
        if int(attack) != int(step.attack):
            raise RuntimeError(
                f"Replay attack {attack} did not match step attack {step.attack}"
            )
        total_attack += int(attack)

    if total_attack != int(playback.total_attack):
        raise RuntimeError(
            f"Rendered attack total {total_attack} did not match playback total {playback.total_attack}"
        )

    final_frame_value = compute_raw_nn_value(model_path, env, config.max_placements)
    final_frame = _capture_frame(
        env,
        placement_number=int(env.placement_count),
        attack=total_attack,
        is_terminal=env.game_over,
        value_pred=final_frame_value,
    )
    plain_frames.append(final_frame)
    move_overlay_frames.append(final_frame.copy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    move_overlay_output_path.parent.mkdir(parents=True, exist_ok=True)
    create_trajectory_gif(plain_frames, str(output_path), duration=args.frame_duration)
    create_trajectory_gif(
        move_overlay_frames,
        str(move_overlay_output_path),
        duration=args.frame_duration,
    )

    logger.info(
        "Rendered seed rollout with NN overlays",
        run_dir=str(run_dir),
        seed=args.seed,
        model_path=str(model_path),
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        output_path=str(output_path),
        move_overlay_output_path=str(move_overlay_output_path),
        num_frames=len(plain_frames),
        total_attack=int(playback.total_attack),
        nn_value_weight=float(config.nn_value_weight),
        first_frame_nn=round(frame_values[0], 6),
        last_frame_nn=round(frame_values[-1], 6),
        final_frame_nn=round(final_frame_value, 6)
        if final_frame_value is not None
        else None,
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
