from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import structlog
import wandb
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT, TRAINING_DATA_FILENAME
from tetris_bot.ml.wandb_resume import resolve_wandb_model_artifact_reference

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    reference: str  # WandB run/artifact reference (e.g. entity/project/run_id)
    output_path: Path | None = (
        None  # Destination file path (default: <PROJECT_ROOT>/training_data.npz)
    )
    run_dir: Path | None = (
        None  # Destination run directory (writes <run_dir>/training_data.npz)
    )
    alias: str = "latest"  # Artifact alias for run references
    overwrite: bool = False  # Overwrite destination file if it exists
    verify_zip: bool = True  # Validate NPZ zip integrity before final replace
    fallback_full_download: bool = (
        True  # If direct-file download fails, download full artifact and extract file
    )
    artifact_type: str = "model"  # Artifact type for WandB API lookup


def resolve_output_path(args: ScriptArgs) -> Path:
    if args.output_path is not None and args.run_dir is not None:
        raise ValueError("Cannot set both output_path and run_dir")

    if args.run_dir is not None:
        return args.run_dir / TRAINING_DATA_FILENAME
    if args.output_path is not None:
        return args.output_path
    return PROJECT_ROOT / TRAINING_DATA_FILENAME


def validate_npz_zip(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Downloaded file does not exist: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt NPZ archive member {corrupt_member!r} in {path}")
        if not archive.namelist():
            raise ValueError(f"NPZ archive is empty: {path}")


def download_training_data_file(
    *,
    artifact: wandb.apis.public.Artifact,
    temp_dir: Path,
    fallback_full_download: bool,
) -> Path:
    try:
        artifact_file = artifact.get_path(TRAINING_DATA_FILENAME)
        downloaded_path = Path(artifact_file.download(root=str(temp_dir / "direct")))
        logger.info(
            "Downloaded training_data.npz via direct artifact path",
            path=str(downloaded_path),
        )
        return downloaded_path
    except Exception as error:
        if not fallback_full_download:
            raise RuntimeError(
                "Failed direct file download and fallback_full_download=False"
            ) from error
        logger.warning(
            "Direct training_data.npz download failed; trying full artifact download",
            error=str(error),
        )

    artifact_root = Path(artifact.download(root=str(temp_dir / "artifact")))
    downloaded_path = artifact_root / TRAINING_DATA_FILENAME
    if not downloaded_path.exists():
        raise FileNotFoundError(
            f"Artifact does not contain {TRAINING_DATA_FILENAME}: {artifact_root}"
        )
    logger.info(
        "Downloaded full artifact and located training_data.npz",
        path=str(downloaded_path),
    )
    return downloaded_path


def atomic_copy(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination exists (set --overwrite true to replace): {destination}"
        )
    temporary_destination = destination.with_name(destination.name + ".tmp")
    temporary_destination.unlink(missing_ok=True)
    shutil.copy2(source, temporary_destination)
    temporary_destination.replace(destination)


def main(args: ScriptArgs) -> None:
    output_path = resolve_output_path(args)
    artifact_ref = resolve_wandb_model_artifact_reference(
        args.reference, default_alias=args.alias
    )

    logger.info(
        "Preparing WandB training-data download",
        reference=args.reference,
        artifact_ref=artifact_ref,
        output_path=str(output_path),
        artifact_type=args.artifact_type,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="tetris-wandb-training-data-"))
    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_ref, type=args.artifact_type)
        downloaded_path = download_training_data_file(
            artifact=artifact,
            temp_dir=temp_dir,
            fallback_full_download=args.fallback_full_download,
        )
        if args.verify_zip:
            validate_npz_zip(downloaded_path)

        atomic_copy(downloaded_path, output_path, overwrite=args.overwrite)

        if args.verify_zip:
            validate_npz_zip(output_path)

        logger.info(
            "Wrote training_data.npz from WandB artifact",
            output_path=str(output_path),
            size_bytes=output_path.stat().st_size,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main(parse(ScriptArgs))
