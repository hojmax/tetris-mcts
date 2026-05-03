"""Tests for the trainer's W&B-facing sequential game-number renumbering.

The trainer holds two parallel game-number namespaces: buffer game_numbers
(per-machine block offsets, used as unique keys for examples in the replay
buffer) and display game_numbers (strictly sequential 1, 2, 3, …, used as
W&B per-game x-axis). Renumbering happens at the single drain chokepoint
`_drain_completed_games` so all downstream W&B logging consumes display
numbers transparently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import TrainingConfig, load_training_config
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import save_checkpoint
from tetris_bot.scripts.train import ScriptArgs, restore_trainer_from_checkpoint


def _make_config(tmp_path: Path) -> TrainingConfig:
    config = load_training_config(DEFAULT_CONFIG_PATH)
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": tmp_path / "checkpoints",
            "data_dir": tmp_path / "data",
        }
    )
    return config


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(_make_config(tmp_path), device="cpu")


def _local_game_payload(
    game_number: int, completed_time_s: float, total_attack: int = 5
) -> dict[str, Any]:
    return {
        "game_number": game_number,
        "stats": {
            "total_attack": float(total_attack),
            "episode_length": 30.0,
        },
        "completed_time_s": completed_time_s,
        "replay": None,
    }


class _FakeGenerator:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)

    def drain_completed_games(self) -> list[dict[str, Any]]:
        out = self._payloads
        self._payloads = []
        return out


def test_drain_assigns_strictly_sequential_display_numbers(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    # Local games (Rust counter) and remote games (block-offset by trainer
    # ingestion) both reach the drain, intermixed and in arbitrary order.
    fake_generator = _FakeGenerator(
        [
            _local_game_payload(1, completed_time_s=10.0),
            _local_game_payload(2, completed_time_s=12.0),
        ]
    )
    trainer.push_remote_completed_games(
        [_local_game_payload(7, completed_time_s=11.0, total_attack=20)],
        game_number_offset=1_000_000_000,
    )
    trainer.push_remote_completed_games(
        [_local_game_payload(8, completed_time_s=13.0, total_attack=22)],
        game_number_offset=1_000_000_000,
    )

    drained = trainer._drain_completed_games(fake_generator)  # type: ignore[arg-type]
    # Expect strictly 1, 2, 3, 4 — ordered by completed_time_s, regardless
    # of source. Buffer/block numbers are NOT what surfaces here.
    assert [entry.game_number for entry in drained] == [1, 2, 3, 4]
    # Stats payloads stay intact (only game_number is rewritten).
    assert [int(e.stats["total_attack"]) for e in drained] == [5, 20, 5, 22]
    # Counter advanced past the highest assigned number.
    assert trainer._next_display_game_number == 5


def test_drain_continues_counter_across_calls(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    gen = _FakeGenerator([_local_game_payload(1, completed_time_s=1.0)])
    first = trainer._drain_completed_games(gen)  # type: ignore[arg-type]
    assert [e.game_number for e in first] == [1]

    # Empty drain doesn't waste numbers.
    empty_gen = _FakeGenerator([])
    assert trainer._drain_completed_games(empty_gen) == []  # type: ignore[arg-type]
    assert trainer._next_display_game_number == 2

    gen2 = _FakeGenerator(
        [
            _local_game_payload(2, completed_time_s=2.0),
            _local_game_payload(3, completed_time_s=3.0),
        ]
    )
    second = trainer._drain_completed_games(gen2)  # type: ignore[arg-type]
    assert [e.game_number for e in second] == [2, 3]
    assert trainer._next_display_game_number == 4


def test_display_counter_round_trips_through_checkpoint(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    # Simulate having drained 99 games already.
    trainer._next_display_game_number = 100

    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=42,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_death_penalty=0.0,
        incumbent_overhang_penalty_weight=0.0,
        incumbent_eval_avg_attack=0.0,
        next_display_game_number=trainer._next_display_game_number,
    )

    fresh = _make_trainer(tmp_path)
    assert fresh._next_display_game_number == 1  # default
    restore_trainer_from_checkpoint(
        fresh,
        ScriptArgs(config=tmp_path / "config.yaml", resume_dir=None),
        _make_config(tmp_path),
        checkpoint,
        incumbent_model_path=None,
    )
    assert fresh._next_display_game_number == 100

    # Next drained game continues at 100, not at 1.
    gen = _FakeGenerator([_local_game_payload(50, completed_time_s=999.0)])
    drained = fresh._drain_completed_games(gen)  # type: ignore[arg-type]
    assert [e.game_number for e in drained] == [100]
    assert fresh._next_display_game_number == 101


def test_legacy_checkpoint_without_counter_starts_at_one(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=10,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_death_penalty=0.0,
        incumbent_overhang_penalty_weight=0.0,
        incumbent_eval_avg_attack=0.0,
    )

    fresh = _make_trainer(tmp_path)
    restore_trainer_from_checkpoint(
        fresh,
        ScriptArgs(config=tmp_path / "config.yaml", resume_dir=None),
        _make_config(tmp_path),
        checkpoint,
        incumbent_model_path=None,
    )
    assert fresh._next_display_game_number == 1
