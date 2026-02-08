"""Inspect training data by rendering games as GIFs."""

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import structlog
from rich.console import Console
from simple_parsing import parse

from tetris_mcts.config import (
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    DEFAULT_GIF_FRAME_DURATION_MS,
    LATEST_CHECKPOINT_FILENAME,
    PIECE_NAMES,
    QUEUE_SIZE,
)
from tetris_mcts.ml.value_predictor import ValuePredictor
from tetris_mcts.ml.visualization import render_board

SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
logger = structlog.get_logger()
console = Console()


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
    save_path: Path | None = (
        None  # Output path (default: script outputs/game_{index}.gif)
    )
    frame_duration: int = DEFAULT_GIF_FRAME_DURATION_MS  # Milliseconds per frame
    print_buffer_vectors: bool = (
        True  # Print full literal vectors/matrices for selected game
    )
    checkpoint_path: Path | None = None  # Checkpoint path (default: <run_dir>/checkpoints/latest.pt)
    config_path: Path | None = None  # Config path (default: <run_dir>/config.json)

    def __post_init__(self) -> None:
        run_dir = self.data_path.parent
        if self.checkpoint_path is None:
            self.checkpoint_path = (
                run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME
            )
        if self.config_path is None:
            self.config_path = run_dir / CONFIG_FILENAME


def format_array(arr: np.ndarray) -> str:
    """Render full array contents without NumPy truncation."""
    return np.array2string(
        arr,
        separator=", ",
        threshold=sys.maxsize,
        max_line_width=160,
    )


def print_game_buffer_vectors(data: np.lib.npyio.NpzFile, start: int, end: int) -> None:
    """Print full game slice vectors and matrices from the replay buffer."""
    game_slice = slice(start, end)
    dump_keys = [
        "boards",
        "current_pieces",
        "hold_pieces",
        "hold_available",
        "next_queue",
        "value_targets",
        "policy_targets",
        "move_numbers",
    ]

    console.rule("[bold]Replay Buffer Slice[/bold]")
    for key in dump_keys:
        if key not in data:
            logger.warning("Buffer key missing, skipping dump", key=key)
            continue
        console.print(f"[bold]{key}[/bold]")
        console.print(format_array(data[key][game_slice]))
        console.print()


def main(args: ScriptArgs) -> None:
    # Validate file
    if not args.data_path.exists():
        logger.error("File not found", path=str(args.data_path))
        return
    if args.data_path.suffix != ".npz":
        logger.error(
            "Expected .npz file", path=str(args.data_path), suffix=args.data_path.suffix
        )
        return

    value_predictor: ValuePredictor | None = None
    if args.checkpoint_path.exists() and args.config_path.exists():
        value_predictor = ValuePredictor(args.checkpoint_path, args.config_path)
        logger.info(
            "Loaded model predictions",
            checkpoint_path=str(args.checkpoint_path),
            config_path=str(args.config_path),
        )
    else:
        logger.warning(
            "Model predictions disabled: checkpoint/config not found",
            checkpoint_path=str(args.checkpoint_path),
            config_path=str(args.config_path),
        )

    # Load data
    with np.load(args.data_path) as data:
        n_examples = len(data["boards"])
        move_numbers = data["move_numbers"]

        # Find game boundaries
        games = find_game_boundaries(move_numbers)
        n_games = len(games)

        # Handle negative game index
        game_idx = (
            args.game_index if args.game_index >= 0 else n_games + args.game_index
        )
        if game_idx < 0 or game_idx >= n_games:
            logger.error(
                "Game index out of range",
                requested_game_index=args.game_index,
                min_index=0,
                max_exclusive=n_games,
            )
            return

        start, end = games[game_idx]
        start = int(start)
        end = int(end)
        game_length = end - start

        logger.info("Loaded dataset", num_games=n_games, total_examples=n_examples)
        logger.info(
            "Rendering game",
            game_index=game_idx,
            example_start=start,
            example_end=end - 1,
            num_frames=game_length,
        )

        if args.print_buffer_vectors:
            print_game_buffer_vectors(data, start, end)

        # Render each frame
        frames = []
        for frame_idx, i in enumerate(range(start, end)):
            board = data["boards"][i]
            current_piece = get_piece_type(data["current_pieces"][i])
            hold_piece = get_piece_type(data["hold_pieces"][i])
            next_queue = [
                get_piece_type(data["next_queue"][i][j]) for j in range(QUEUE_SIZE)
            ]
            can_hold = bool(data["hold_available"][i])
            move_number = frame_idx  # Use frame index as move number
            value_target = float(data["value_targets"][i])
            value_pred = None
            if value_predictor is not None:
                value_pred = value_predictor.predict_value(
                    index=i,
                    board=data["boards"][i],
                    current_piece=data["current_pieces"][i],
                    hold_piece=data["hold_pieces"][i],
                    hold_available=float(data["hold_available"][i]),
                    next_queue=data["next_queue"][i],
                    move_number=float(data["move_numbers"][i]),
                )

            # Build piece info
            current_name = (
                PIECE_NAMES[current_piece] if current_piece is not None else "?"
            )
            hold_name = PIECE_NAMES[hold_piece] if hold_piece is not None else "-"
            queue_names = [PIECE_NAMES[p] if p is not None else "?" for p in next_queue]

            frame = render_board(
                board=board,
                move_number=move_number,
                attack=int(value_target),
                value_pred=value_pred,
                info_text=f"Can hold: {'yes' if can_hold else 'no'}",
                show_piece_info=True,
                current_piece_name=current_name,
                hold_piece_name=hold_name,
                queue_pieces=queue_names,
            )
            frames.append(frame)

        # Determine save path
        if args.save_path is None:
            save_path = OUTPUTS_DIR / f"game_{game_idx}.gif"
        else:
            save_path = args.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as GIF
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=args.frame_duration,
            loop=0,
        )

        logger.info("Saved gif", num_frames=len(frames), path=str(save_path))


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
