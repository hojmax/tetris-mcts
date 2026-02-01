"""Inspect training data by rendering games as GIFs."""

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from simple_parsing import parse

from tetris_mcts.ml.visualization import render_board

SCRIPT_DIR = Path(__file__).parent
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]


def get_piece_type(one_hot: np.ndarray) -> int | None:
    """Convert one-hot encoded piece to type index."""
    idx = np.argmax(one_hot)
    # For hold_pieces, index 7 means empty
    if len(one_hot) == 8 and idx == 7:
        return None
    return int(idx)


def find_game_boundaries(move_numbers: np.ndarray) -> list[tuple[int, int]]:
    """Find start/end indices for each game (where move_number resets to 0)."""
    # Game starts where move_number equals 0 (or very close to 0)
    game_starts = np.where(move_numbers < 0.001)[0]

    games = []
    for i, start in enumerate(game_starts):
        end = game_starts[i + 1] if i + 1 < len(game_starts) else len(move_numbers)
        games.append((start, end))
    return games


@dataclass
class ScriptArgs:
    """Inspect training data by rendering games as GIFs."""

    data_path: Path  # Path to training_data.npz file
    game_index: int = -1  # Which game to render (-1 for last)
    save_path: Path | None = None  # Output path (default: script_dir/game_{index}.gif)
    frame_duration: int = 300  # Milliseconds per frame


def main(args: ScriptArgs) -> None:
    # Validate file
    if not args.data_path.exists():
        print(f"Error: File not found: {args.data_path}")
        return
    if args.data_path.suffix != ".npz":
        print(f"Error: Expected .npz file, got: {args.data_path}")
        return

    # Load data
    data = np.load(args.data_path)
    n_examples = len(data["boards"])
    move_numbers = data["move_numbers"]

    # Find game boundaries
    games = find_game_boundaries(move_numbers)
    n_games = len(games)

    # Handle negative game index
    game_idx = args.game_index if args.game_index >= 0 else n_games + args.game_index
    if game_idx < 0 or game_idx >= n_games:
        print(f"Error: Game index {args.game_index} out of range [0, {n_games})")
        return

    start, end = games[game_idx]
    game_length = end - start

    print(f"Found {n_games} games in dataset ({n_examples} total examples)")
    print(
        f"Rendering game {game_idx} (examples {start}-{end - 1}, {game_length} frames)"
    )

    # Render each frame
    frames = []
    for frame_idx, i in enumerate(range(start, end)):
        board = data["boards"][i]
        current_piece = get_piece_type(data["current_pieces"][i])
        hold_piece = get_piece_type(data["hold_pieces"][i])
        next_queue = [get_piece_type(data["next_queue"][i][j]) for j in range(5)]
        move_number = frame_idx  # Use frame index as move number
        value_target = float(data["value_targets"][i])

        # Build piece info
        current_name = PIECE_NAMES[current_piece] if current_piece is not None else "?"
        hold_name = PIECE_NAMES[hold_piece] if hold_piece is not None else "-"
        queue_names = [PIECE_NAMES[p] if p is not None else "?" for p in next_queue]

        frame = render_board(
            board=board,
            move_number=move_number,
            attack=int(value_target),
            info_text=f"Value: {value_target:.1f}",
            show_piece_info=True,
            current_piece_name=current_name,
            hold_piece_name=hold_name,
            queue_pieces=queue_names,
        )
        frames.append(frame)

    # Determine save path
    if args.save_path is None:
        save_path = SCRIPT_DIR / f"game_{game_idx}.gif"
    else:
        save_path = args.save_path

    # Save as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration,
        loop=0,
    )

    print(f"Saved {len(frames)} frames to {save_path}")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
