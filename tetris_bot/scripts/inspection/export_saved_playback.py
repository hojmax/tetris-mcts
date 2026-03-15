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

from tetris_core.tetris_core import MCTSAgent, MCTSConfig, TetrisEnv
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_DIRNAME,
    INCUMBENT_ONNX_FILENAME,
)
from tetris_bot.scripts.inspection.tree_playback_artifact import (
    save_tree_playback_artifact,
)
from tetris_bot.scripts.utils.run_search_config import (
    default_checkpoint_path,
    load_effective_self_play_config,
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
    return (
        args.model_path
        if args.model_path is not None
        else args.run_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME
    )


def resolve_output_path(args: ScriptArgs) -> Path:
    if args.output_path is not None:
        return args.output_path
    return (
        args.run_dir
        / "analysis"
        / "eval_trees"
        / f"seed{args.seed}_full_game_playback.json"
    )


def build_mcts_config(run_dir: Path, checkpoint_path: Path | None) -> MCTSConfig:
    config_data = load_effective_self_play_config(run_dir, checkpoint_path)

    config = MCTSConfig()
    config.num_simulations = int(config_data["num_simulations"])
    config.c_puct = float(config_data["c_puct"])
    config.temperature = float(config_data["temperature"])
    config.dirichlet_alpha = float(config_data["dirichlet_alpha"])
    config.dirichlet_epsilon = float(config_data["dirichlet_epsilon"])
    config.visit_sampling_epsilon = float(config_data.get("visit_sampling_epsilon", 0.0))
    config.max_placements = int(config_data["max_placements"])
    config.q_scale = (
        float(config_data["q_scale"])
        if config_data.get("use_tanh_q_normalization", True)
        and config_data.get("q_scale") is not None
        else None
    )
    config.reuse_tree = bool(config_data.get("reuse_tree", True))
    config.nn_value_weight = float(config_data.get("nn_value_weight", config.nn_value_weight))
    config.death_penalty = float(config_data.get("death_penalty", config.death_penalty))
    config.overhang_penalty_weight = float(
        config_data.get("overhang_penalty_weight", config.overhang_penalty_weight)
    )
    mcts_seed = config_data.get("mcts_seed")
    config.seed = int(mcts_seed) if mcts_seed is not None else None
    config.track_value_history = True
    return config


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

    config = build_mcts_config(run_dir, checkpoint_path)
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
