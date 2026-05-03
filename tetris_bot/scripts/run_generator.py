"""Generator-only entrypoint for multi-machine training.

Run this on a machine that should produce self-play games for a remote trainer.
The script:

1. Downloads the trainer's current incumbent ONNX bundle from R2 (waiting up to
   `bootstrap_wait_seconds` if no model has been published yet).
2. Spawns a Rust GameGenerator with `candidate_gating_enabled=False` (the
   trainer is the source of truth for promotion decisions).
3. Starts a `ChunkUploader` that drains replay deltas to R2.
4. Starts a `ModelDownloader` that polls the incumbent pointer and applies new
   bundles via `sync_model_directly`.
5. Runs until SIGINT/SIGTERM, then shuts everything down cleanly.

Usage:
    R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    python tetris_bot/scripts/run_generator.py \
        --config config.yaml \
        --sync_run_id v9 \
        --machine_id laptop \
        --num_workers 6
"""

from __future__ import annotations

import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from dotenv import load_dotenv
from simple_parsing import parse

# Load `.env` (R2_ENDPOINT_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)
# before any code that touches `os.environ` for those keys.
load_dotenv()

from tetris_core.tetris_core import GameGenerator, MCTSConfig  # noqa: E402

from tetris_bot.constants import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.config import (  # noqa: E402
    R2SyncConfig,
    SelfPlayConfig,
    load_training_config,
)
from tetris_bot.ml.r2_sync import (  # noqa: E402
    ChunkUploader,
    GameStatsUploader,
    ModelDownloader,
    R2Settings,
    download_model_bundle,
    fetch_model_pointer,
    make_s3_client,
)

logger = structlog.get_logger()


@dataclass
class GeneratorArgs:
    """Generator script arguments."""

    config: Path = DEFAULT_CONFIG_PATH
    workspace: Path = Path("generator_workspace")
    machine_id: str | None = None  # defaults to hostname
    num_workers: int | None = None  # overrides config.self_play.num_workers
    sync_run_id: str | None = None  # overrides r2_sync.sync_run_id
    bootstrap_wait_seconds: float = 600.0


def _build_mcts_config(self_play: SelfPlayConfig) -> MCTSConfig:
    cfg = MCTSConfig()
    cfg.num_simulations = self_play.num_simulations
    cfg.c_puct = self_play.c_puct
    cfg.temperature = self_play.temperature
    cfg.dirichlet_alpha = self_play.dirichlet_alpha
    cfg.dirichlet_epsilon = self_play.dirichlet_epsilon
    cfg.visit_sampling_epsilon = self_play.visit_sampling_epsilon
    cfg.seed = self_play.mcts_seed
    cfg.max_placements = self_play.max_placements
    cfg.death_penalty = self_play.death_penalty
    cfg.overhang_penalty_weight = self_play.overhang_penalty_weight
    cfg.nn_value_weight = self_play.nn_value_weight
    cfg.use_parent_value_for_unvisited_q = self_play.use_parent_value_for_unvisited_q
    cfg.reuse_tree = self_play.reuse_tree
    return cfg


def _resolve_r2_settings(
    cfg: R2SyncConfig, sync_run_id_override: str | None
) -> R2Settings:
    if sync_run_id_override is not None:
        cfg = cfg.model_copy(update={"sync_run_id": sync_run_id_override})
    if cfg.role == "off":
        raise ValueError(
            "r2_sync.role must be 'generator' (or 'both') in the config used by run_generator.py"
        )
    return R2Settings.from_config(cfg)


def _wait_for_initial_pointer(settings: R2Settings, timeout_seconds: float):
    """Block until the trainer publishes an incumbent pointer, or raise."""
    deadline = time.monotonic() + timeout_seconds
    client = make_s3_client(settings)
    poll_interval = 5.0
    while time.monotonic() < deadline:
        pointer = fetch_model_pointer(settings, client=client)
        if pointer is not None:
            return pointer
        logger.info(
            "run_generator.waiting_for_initial_model",
            sync_run_id=settings.sync_run_id,
            poll_interval_seconds=poll_interval,
        )
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting {timeout_seconds:.0f}s for initial model pointer at "
        f"{settings.bucket}/{settings.model_pointer_key()}"
    )


def main(args: GeneratorArgs) -> None:
    config = load_training_config(args.config.resolve())
    settings = _resolve_r2_settings(config.r2_sync, args.sync_run_id)
    machine_id = args.machine_id or socket.gethostname().replace(".", "_")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    models_dir = workspace / "models"
    cursor_path = workspace / "r2_upload_cursor.json"
    game_stats_cursor_path = workspace / "r2_game_stats_cursor.json"
    training_data_path = workspace / TRAINING_DATA_FILENAME

    logger.info(
        "run_generator.starting",
        sync_run_id=settings.sync_run_id,
        machine_id=machine_id,
        workspace=str(workspace),
        endpoint_url=settings.endpoint_url,
        bucket=settings.bucket,
    )

    pointer = _wait_for_initial_pointer(settings, args.bootstrap_wait_seconds)
    bundle_dir = models_dir / f"step_{pointer.step:020d}"
    if bundle_dir.exists():
        logger.info(
            "run_generator.using_cached_initial_bundle",
            step=pointer.step,
            local_path=str(bundle_dir),
        )
        initial_main = bundle_dir / "bundle.onnx"
        if not initial_main.exists():
            initial_main = download_model_bundle(
                settings=settings, pointer=pointer, dest_dir=bundle_dir
            )
    else:
        initial_main = download_model_bundle(
            settings=settings, pointer=pointer, dest_dir=bundle_dir
        )
    logger.info(
        "run_generator.initial_bundle_ready",
        step=pointer.step,
        nn_value_weight=pointer.nn_value_weight,
        local_path=str(initial_main),
    )

    num_workers = args.num_workers or config.self_play.num_workers
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")

    mcts_config = _build_mcts_config(config.self_play)
    generator = GameGenerator(
        model_path=str(initial_main),
        training_data_path=str(training_data_path),
        config=mcts_config,
        max_placements=config.self_play.max_placements,
        add_noise=config.self_play.add_noise,
        max_examples=config.replay.buffer_size,
        save_interval_seconds=0.0,
        num_workers=num_workers,
        initial_model_step=pointer.step,
        candidate_eval_seeds=[],
        start_with_network=True,
        non_network_num_simulations=config.self_play.bootstrap_num_simulations,
        initial_incumbent_eval_avg_attack=0.0,
        nn_value_weight_cap=config.self_play.nn_value_weight_cap,
        candidate_gating_enabled=False,
        save_eval_trees=False,
    )
    generator.start()
    logger.info("run_generator.generator_started", num_workers=num_workers)

    uploader = ChunkUploader(
        generator=generator,
        settings=settings,
        machine_id=machine_id,
        cursor_path=cursor_path,
        chunk_max_examples=config.r2_sync.chunk_max_examples,
        upload_interval_seconds=config.r2_sync.chunk_upload_interval_seconds,
    )
    game_stats_uploader = GameStatsUploader(
        generator=generator,
        settings=settings,
        machine_id=machine_id,
        cursor_path=game_stats_cursor_path,
        upload_interval_seconds=config.r2_sync.chunk_upload_interval_seconds,
    )
    downloader = ModelDownloader(
        generator=generator,
        settings=settings,
        local_models_dir=models_dir,
        poll_interval_seconds=config.r2_sync.model_pointer_poll_interval_seconds,
        initial_model_step=pointer.step,
    )
    uploader.start()
    game_stats_uploader.start()
    downloader.start()

    stop_event_handled = False

    def _shutdown(signum: int, _frame) -> None:
        nonlocal stop_event_handled
        if stop_event_handled:
            return
        stop_event_handled = True
        logger.info("run_generator.shutdown_signal", signum=signum)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop_event_handled:
            time.sleep(0.5)
    finally:
        logger.info("run_generator.stopping_workers")
        try:
            generator.stop()
        except Exception:
            logger.exception("run_generator.generator_stop_failed")
        downloader.stop()
        game_stats_uploader.stop()
        uploader.stop()
        logger.info(
            "run_generator.shutdown_complete",
            games_generated=generator.games_generated(),
            examples_generated=generator.examples_generated(),
        )


if __name__ == "__main__":
    main(parse(GeneratorArgs))
