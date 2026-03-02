from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tetris_bot.ml.npz_validation import (
    REQUIRED_TRAINING_DATA_NPZ_ENTRIES,
    validate_training_data_npz,
)


def write_training_data_stub(path: Path, *, include_all_required: bool) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        entries = (
            REQUIRED_TRAINING_DATA_NPZ_ENTRIES
            if include_all_required
            else {"boards.npy"}
        )
        for name in entries:
            archive.writestr(name, b"stub")


def test_validate_training_data_npz_accepts_required_entries(tmp_path: Path) -> None:
    npz_path = tmp_path / "training_data.npz"
    write_training_data_stub(npz_path, include_all_required=True)
    validate_training_data_npz(npz_path)


def test_validate_training_data_npz_rejects_missing_required_entries(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "training_data.npz"
    write_training_data_stub(npz_path, include_all_required=False)
    with pytest.raises(ValueError, match="missing required entries"):
        validate_training_data_npz(npz_path)


def test_validate_training_data_npz_rejects_non_zip(tmp_path: Path) -> None:
    npz_path = tmp_path / "training_data.npz"
    npz_path.write_text("not a zip archive")
    with pytest.raises(ValueError, match="not a valid zip archive"):
        validate_training_data_npz(npz_path)
