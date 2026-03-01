from __future__ import annotations

from pathlib import Path

import pytest

from tetris_bot.constants import CHECKPOINT_DIRNAME, INCUMBENT_ONNX_FILENAME
from tetris_bot.ml.wandb_resume import (
    resolve_wandb_model_artifact_reference,
    stage_resume_directory_from_wandb_artifact,
)


def test_resolve_run_reference_to_model_artifact() -> None:
    artifact_ref = resolve_wandb_model_artifact_reference("entity/project/abc123")
    assert artifact_ref == "entity/project/tetris-model-abc123:latest"


def test_resolve_run_url_to_model_artifact() -> None:
    artifact_ref = resolve_wandb_model_artifact_reference(
        "https://wandb.ai/entity/project/runs/abc123?workspace=user-foo"
    )
    assert artifact_ref == "entity/project/tetris-model-abc123:latest"


def test_resolve_artifact_reference_without_alias_uses_default() -> None:
    artifact_ref = resolve_wandb_model_artifact_reference(
        "entity/project/tetris-model-abc123",
        default_alias="final",
    )
    assert artifact_ref == "entity/project/tetris-model-abc123:final"


def test_resolve_artifact_url_preserves_alias() -> None:
    artifact_ref = resolve_wandb_model_artifact_reference(
        "https://wandb.ai/entity/project/artifacts/model/tetris-model-abc123/step-900"
    )
    assert artifact_ref == "entity/project/tetris-model-abc123:step-900"


def test_stage_resume_directory_chooses_highest_checkpoint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "checkpoint_100.pt").write_text("step100")
    (artifact_dir / "checkpoint_300.pt").write_text("step300")
    (artifact_dir / "training_data.npz").write_bytes(b"npz")

    destination_dir = tmp_path / "staged"
    staged_dir = stage_resume_directory_from_wandb_artifact(
        artifact_dir, destination_dir
    )

    staged_checkpoint = staged_dir / CHECKPOINT_DIRNAME / "latest.pt"
    assert staged_checkpoint.read_text() == "step300"
    assert (staged_dir / "training_data.npz").read_bytes() == b"npz"


def test_stage_resume_directory_accepts_single_checkpoint_file(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.pt").write_text("weights")

    staged_dir = stage_resume_directory_from_wandb_artifact(
        artifact_dir, tmp_path / "staged"
    )
    staged_checkpoint = staged_dir / CHECKPOINT_DIRNAME / "latest.pt"
    assert staged_checkpoint.read_text() == "weights"


def test_stage_resume_directory_raises_without_any_checkpoint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoint .pt file found"):
        stage_resume_directory_from_wandb_artifact(artifact_dir, tmp_path / "staged")


def test_stage_resume_directory_copies_incumbent_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "checkpoint_1.pt").write_text("step1")
    source_incumbent = artifact_dir / INCUMBENT_ONNX_FILENAME
    source_incumbent.write_text("source-incumbent")

    copied_paths: dict[str, tuple[Path, Path]] = {}

    def fake_copy_model_artifact_bundle(source: Path, destination: Path) -> None:
        copied_paths["incumbent"] = (source, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text())

    monkeypatch.setattr(
        "tetris_bot.ml.wandb_resume.copy_model_artifact_bundle",
        fake_copy_model_artifact_bundle,
    )

    staged_dir = stage_resume_directory_from_wandb_artifact(
        artifact_dir, tmp_path / "staged"
    )
    destination_incumbent = staged_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME

    assert "incumbent" in copied_paths
    assert copied_paths["incumbent"][0] == source_incumbent
    assert copied_paths["incumbent"][1] == destination_incumbent
    assert destination_incumbent.read_text() == "source-incumbent"
