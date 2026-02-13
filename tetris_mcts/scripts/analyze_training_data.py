"""Analyze training data statistics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import parse


def find_game_boundaries(move_numbers: np.ndarray) -> list[tuple[int, int]]:
    """Find start/end indices for each game (where move_number resets to 0)."""
    game_starts = np.where(move_numbers < 0.001)[0]
    games = []
    for i, start in enumerate(game_starts):
        end = game_starts[i + 1] if i + 1 < len(game_starts) else len(move_numbers)
        games.append((start, end))
    return games


def print_histogram(values: np.ndarray, bins: int = 10, title: str = "") -> None:
    """Print a simple text histogram."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))

    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts)
    bar_width = 40

    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {left:6.1f} - {right:6.1f}: {bar} ({count})")


def print_percentiles(values: np.ndarray, title: str = "") -> None:
    """Print percentile statistics."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))

    percentiles = [0, 10, 25, 50, 75, 90, 100]
    vals = np.percentile(values, percentiles)
    for p, v in zip(percentiles, vals):
        print(f"  p{p:3d}: {v:8.2f}")


@dataclass
class ScriptArgs:
    """Analyze training data statistics."""

    data_path: Path  # Path to training_data.npz file


def main(args: ScriptArgs) -> None:
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
    value_targets = data["value_targets"]
    if len(move_numbers) != n_examples:
        raise ValueError("move_numbers length does not match boards length")
    if len(value_targets) != n_examples:
        raise ValueError("value_targets length does not match boards length")

    # Find game boundaries
    games = find_game_boundaries(move_numbers)
    n_games = len(games)

    # Basic stats
    print("=" * 50)
    print("TRAINING DATA ANALYSIS")
    print("=" * 50)
    print(f"\nFile: {args.data_path}")
    print(f"Total examples: {n_examples:,}")
    print(f"Total games: {n_games:,}")
    print(f"Avg examples per game: {n_examples / n_games:.1f}")

    # Per-game statistics
    game_lengths = []
    game_final_values = []

    for start, end in games:
        length = end - start
        game_lengths.append(length)
        # First state's value target is cumulative future attack (= total game attack).
        total_attack = value_targets[start]
        game_final_values.append(total_attack)

    game_lengths = np.array(game_lengths)
    game_total_attacks = np.array(game_final_values)

    # Game length stats
    print_percentiles(game_lengths, "Game Length (moves)")
    print_histogram(game_lengths, bins=10, title="Game Length Distribution")

    # Total attack per game stats
    print_percentiles(game_total_attacks, "Total Attack Per Game")
    print_histogram(game_total_attacks, bins=10, title="Total Attack Distribution")

    # Value target distribution (all examples)
    print_percentiles(value_targets, "Value Targets (all examples)")
    print_histogram(value_targets, bins=15, title="Value Target Distribution")

    # Summary table
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Games:          {n_games:>10,}")
    print(f"  Examples:       {n_examples:>10,}")
    print(f"  Avg game len:   {np.mean(game_lengths):>10.1f}")
    print(f"  Median attack:  {np.median(game_total_attacks):>10.1f}")
    print(f"  Mean attack:    {np.mean(game_total_attacks):>10.1f}")
    print(f"  Max attack:     {np.max(game_total_attacks):>10.1f}")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
