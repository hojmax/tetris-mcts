"""Export a full-game saved playback artifact for one run/seed.

This resolves rollout settings from the saved checkpoint state first, then
falls back to `config.json`, so exported trees match the model's effective
search-time configuration at the time of save.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import MCTSAgent, TetrisEnv
from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH
from tetris_bot.scripts.inspection.tree_playback_artifact import (
    save_tree_playback_artifact,
)
from tetris_bot.scripts.utils.run_search_config import (
    build_mcts_config,
    default_checkpoint_path,
    default_model_path,
)

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    run_dir: Path
    seed: int
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    model_path: Path | None = None
    add_noise: bool = False


def resolve_model_path(args: ScriptArgs) -> Path:
    return args.model_path if args.model_path is not None else default_model_path(args.run_dir)


def resolve_output_path(args: ScriptArgs) -> Path:
    if args.output_path is not None:
        return args.output_path
    return (
        args.run_dir
        / "analysis"
        / "eval_trees"
        / f"seed{args.seed}_full_game_playback.json"
    )


def main(args: ScriptArgs) -> None:
    run_dir = args.run_dir.resolve()
    model_path = resolve_model_path(args).resolve()
    checkpoint_path = (
        args.checkpoint_path.resolve()
        if args.checkpoint_path is not None
        else default_checkpoint_path(run_dir)
    )
    output_path = resolve_output_path(args).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = build_mcts_config(run_dir, checkpoint_path, track_value_history=True)
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

    save_tree_playback_artifact(
        playback,
        output_path,
        initial_seed=args.seed,
        config=config,
        add_noise=args.add_noise,
        model_path=model_path,
        source="manual_seed_export",
    )
    logger.info(
        "Exported saved playback",
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        model_path=str(model_path),
        output_path=str(output_path),
        seed=args.seed,
        total_attack=int(playback.total_attack),
        num_moves=int(playback.num_moves),
        tree_reuse_hits=int(playback.tree_reuse_hits),
        tree_reuse_misses=int(playback.tree_reuse_misses),
        nn_value_weight=float(config.nn_value_weight),
        death_penalty=float(config.death_penalty),
        overhang_penalty_weight=float(config.overhang_penalty_weight),
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
