from __future__ import annotations

from pathlib import Path
import zipfile

REQUIRED_TRAINING_DATA_NPZ_ENTRIES = {
    "boards.npy",
    "policy_targets.npy",
    "value_targets.npy",
    "action_masks.npy",
    "game_numbers.npy",
    "game_total_attacks.npy",
}


def validate_training_data_npz(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ValueError(
                    f"Corrupt NPZ member {corrupt_member!r} in training snapshot {path}"
                )
            archive_entries = set(archive.namelist())
            if not archive_entries:
                raise ValueError(f"Training snapshot is empty: {path}")
            missing_entries = sorted(REQUIRED_TRAINING_DATA_NPZ_ENTRIES - archive_entries)
            if missing_entries:
                raise ValueError(
                    "Training snapshot is missing required entries: "
                    + ", ".join(missing_entries)
                )
    except zipfile.BadZipFile as error:
        raise ValueError(f"Training snapshot is not a valid zip archive: {path}") from error
