from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tetris_bot.constants import PROJECT_ROOT, TRAINING_DATA_FILENAME
from tetris_bot.scripts.inspection.download_wandb_training_data import (
    ScriptArgs,
    atomic_copy,
    resolve_output_path,
    validate_npz_zip,
)


def write_minimal_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("dummy.npy", b"123")


def test_resolve_output_path_defaults_to_project_root() -> None:
    output_path = resolve_output_path(ScriptArgs(reference="entity/project/run"))
    assert output_path == PROJECT_ROOT / TRAINING_DATA_FILENAME


def test_resolve_output_path_uses_run_dir() -> None:
    output_path = resolve_output_path(
        ScriptArgs(reference="entity/project/run", run_dir=Path("training_runs/v9"))
    )
    assert output_path == Path("training_runs/v9") / TRAINING_DATA_FILENAME


def test_resolve_output_path_rejects_conflict() -> None:
    with pytest.raises(ValueError, match="Cannot set both output_path and run_dir"):
        resolve_output_path(
            ScriptArgs(
                reference="entity/project/run",
                output_path=Path("a.npz"),
                run_dir=Path("training_runs/v9"),
            )
        )


def test_validate_npz_zip_accepts_valid_archive(tmp_path: Path) -> None:
    npz_path = tmp_path / "training_data.npz"
    write_minimal_zip(npz_path)
    validate_npz_zip(npz_path)


def test_validate_npz_zip_rejects_non_zip(tmp_path: Path) -> None:
    bad_path = tmp_path / "training_data.npz"
    bad_path.write_text("not a zip")
    with pytest.raises(zipfile.BadZipFile):
        validate_npz_zip(bad_path)


def test_atomic_copy_rejects_existing_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "dest.npz"
    source.write_text("new")
    destination.write_text("old")
    with pytest.raises(FileExistsError):
        atomic_copy(source, destination, overwrite=False)


def test_atomic_copy_overwrites_when_enabled(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "dest.npz"
    source.write_text("new")
    destination.write_text("old")
    atomic_copy(source, destination, overwrite=True)
    assert destination.read_text() == "new"
