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
from tetris_mcts.ml.visualization import render_board


def load_and_render_replay(replay_data: dict, output_path: Path):
    """Load a replay and render it as a GIF."""
    seed = replay_data["seed"]
    moves = replay_data["moves"]

    # Create environment with the replay seed
    env = TetrisEnv.with_seed(10, 20, seed)
    frames = []
    total_attack = 0

    # Render initial state
    frames.append(_render_frame(env, 0, total_attack))

    # Execute each move and render
    for i, move in enumerate(moves):
        # Find and execute placement
        placements = env.get_possible_placements()
        placement = None
        for p in placements:
            if (
                p.piece.x == move["x"]
                and p.piece.y == move["y"]
                and p.piece.rotation == move["rotation"]
            ):
                placement = p
                break

        if placement:
            env.execute_placement(placement)
            total_attack += move["attack"]
            frames.append(_render_frame(env, i + 1, total_attack))

    # Save as GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
        )

    return len(frames), total_attack


def _render_frame(env: TetrisEnv, move_num: int, attack: int):
    """Render a single frame."""
    board = np.array(env.get_board())
    board_colors = env.get_board_colors()
    piece = env.get_current_piece()
    ghost = env.get_ghost_piece()

    return render_board(
        board=board,
        board_colors=board_colors,
        current_piece_cells=piece.get_cells() if piece else None,
        current_piece_type=piece.piece_type if piece else None,
        ghost_cells=ghost.get_cells() if ghost else None,
        move_number=move_num,
        attack=attack,
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
