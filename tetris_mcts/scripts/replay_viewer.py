"""
Replay viewer - renders game replays as GIFs.

Usage:
    python replay_viewer.py <replays.jsonl> [output_dir]

Loads JSONL replay files and renders each game as an animated GIF.
"""

import json
import sys
from pathlib import Path

import numpy as np

from tetris_core import TetrisEnv
from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FRAME_DURATION_MS,
    PIECE_NAMES,
    QUEUE_SIZE,
)
from tetris_mcts.ml.visualization import render_board


def load_and_render_replay(replay_data: dict, output_path: Path):
    """Load a replay and render it as a GIF."""
    seed = replay_data["seed"]
    moves = replay_data["moves"]

    # Create environment with the replay seed
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)
    frames = []
    total_attack = 0

    # Render initial state
    frames.append(
        _render_frame(
            env,
            0,
            total_attack,
            info_text="",
        )
    )

    # Execute each move and render
    for i, move in enumerate(moves):
        action = int(move["action"])
        attack_from_env = env.execute_action_index(action)
        if attack_from_env is None:
            raise ValueError(f"Invalid replay action index: {action}")
        attack_from_replay = int(move["attack"])
        if int(attack_from_env) != attack_from_replay:
            raise ValueError(
                "Replay attack mismatch: "
                f"move={i} env_attack={int(attack_from_env)} "
                f"replay_attack={attack_from_replay}"
            )
        total_attack += int(attack_from_env)
        frames.append(
            _render_frame(
                env,
                i + 1,
                total_attack,
                info_text="Final" if i + 1 == len(moves) else "",
            )
        )

    # Save as GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=DEFAULT_GIF_FRAME_DURATION_MS,
            loop=0,
        )

    return len(frames), total_attack


def _render_frame(env: TetrisEnv, move_num: int, attack: int, info_text: str):
    """Render a single frame."""
    board = np.array(env.get_board())
    board_colors = env.get_board_colors()
    piece = env.get_current_piece()
    ghost = env.get_ghost_piece()
    hold_piece = env.get_hold_piece()

    current_piece_name = PIECE_NAMES[piece.piece_type] if piece is not None else "?"
    hold_piece_name = (
        PIECE_NAMES[hold_piece.piece_type] if hold_piece is not None else "-"
    )
    queue_pieces = [PIECE_NAMES[piece] for piece in env.get_queue(QUEUE_SIZE)]

    return render_board(
        board=board,
        board_colors=board_colors,
        current_piece_cells=piece.get_cells() if piece else None,
        current_piece_type=piece.piece_type if piece else None,
        ghost_cells=ghost.get_cells() if ghost else None,
        move_number=move_num,
        attack=attack,
        info_text=info_text,
        show_piece_info=True,
        current_piece_name=current_piece_name,
        hold_piece_name=hold_piece_name,
        queue_pieces=queue_pieces,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_viewer.py <replays.jsonl> [output_dir]")
        sys.exit(1)

    replay_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    if not replay_path.exists():
        print(f"File not found: {replay_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and render each replay
    with open(replay_path) as f:
        for i, line in enumerate(f):
            replay = json.loads(line)
            output_path = output_dir / f"game_{i:03d}_seed{replay['seed']}.gif"

            frames, attack = load_and_render_replay(replay, output_path)
            print(f"Game {i}: {frames} frames, {attack} attack -> {output_path}")

    print(f"\nDone! GIFs saved to {output_dir}")


if __name__ == "__main__":
    main()
