from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import structlog
import wandb

from tetris_bot.constants import (
    CHECKPOINT_DIRNAME,
    CHECKPOINT_FILENAME_PREFIX,
    INCUMBENT_ONNX_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.artifacts import copy_model_artifact_bundle

logger = structlog.get_logger()


@dataclass(frozen=True)
class WandbResumeSource:
    temp_dir: Path
    resume_dir: Path
    artifact_ref: str


def _normalize_wandb_reference(reference: str) -> str:
    normalized = reference.strip()
    if not normalized:
        raise ValueError("WandB reference cannot be empty")

    if normalized.startswith("wandb://"):
        normalized = normalized.removeprefix("wandb://")
    elif normalized.startswith("wandb.ai/"):
        normalized = normalized.removeprefix("wandb.ai/")
    elif "://" in normalized:
        parsed = urlparse(normalized)
        if parsed.netloc not in {"wandb.ai", "www.wandb.ai"}:
            raise ValueError(
                "WandB URL must point to wandb.ai "
                f"(got host {parsed.netloc!r} from {reference!r})"
            )
        normalized = parsed.path

    normalized = normalized.strip("/")
    if not normalized:
        raise ValueError(f"Failed to parse WandB reference: {reference!r}")
    return normalized


def resolve_wandb_model_artifact_reference(
    reference: str, default_alias: str = "latest"
) -> str:
    alias = default_alias.strip()
    if not alias:
        raise ValueError("default_alias cannot be empty")

    normalized = _normalize_wandb_reference(reference)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3:
        raise ValueError(
            "WandB reference must include entity/project/run_id or "
            f"entity/project/artifact (got {reference!r})"
        )

    # Run URL/path, e.g. entity/project/runs/abc123 or entity/project/abc123
    if len(parts) >= 4 and parts[2] == "runs":
        entity, project, run_id = parts[0], parts[1], parts[3]
        return f"{entity}/{project}/tetris-model-{run_id}:{alias}"
    if (
        len(parts) == 3
        and ":" not in parts[2]
        and not parts[2].startswith("tetris-model-")
    ):
        entity, project, run_id = parts
        return f"{entity}/{project}/tetris-model-{run_id}:{alias}"

    # Artifact URL, e.g. entity/project/artifacts/model/tetris-model-abc123/latest
    if len(parts) >= 6 and parts[2] == "artifacts" and parts[3] == "model":
        entity, project = parts[0], parts[1]
        artifact_name, artifact_alias = parts[4], parts[5]
        return f"{entity}/{project}/{artifact_name}:{artifact_alias}"

    artifact_ref = "/".join(parts)
    if ":" not in artifact_ref.rsplit("/", maxsplit=1)[-1]:
        artifact_ref = f"{artifact_ref}:{alias}"
    return artifact_ref


def _parse_checkpoint_step(path: Path) -> int | None:
    prefix = f"{CHECKPOINT_FILENAME_PREFIX}_"
    if path.suffix != ".pt" or not path.stem.startswith(prefix):
        return None
    step_text = path.stem.removeprefix(prefix)
    if not step_text.isdigit():
        return None
    return int(step_text)


def _choose_resume_checkpoint(artifact_dir: Path) -> Path:
    latest_checkpoint = artifact_dir / LATEST_CHECKPOINT_FILENAME
    if latest_checkpoint.exists():
        return latest_checkpoint

    checkpoint_candidates: list[tuple[int, Path]] = []
    for path in artifact_dir.glob(f"{CHECKPOINT_FILENAME_PREFIX}_*.pt"):
        step = _parse_checkpoint_step(path)
        if step is not None:
            checkpoint_candidates.append((step, path))
    if checkpoint_candidates:
        return max(checkpoint_candidates, key=lambda item: item[0])[1]

    all_pt_files = sorted(artifact_dir.glob("*.pt"))
    if len(all_pt_files) == 1:
        return all_pt_files[0]
    if len(all_pt_files) > 1:
        raise FileNotFoundError(
            "WandB artifact has multiple .pt files but none are "
            f"named {CHECKPOINT_FILENAME_PREFIX}_<step>.pt: "
            f"{[path.name for path in all_pt_files]}"
        )

    raise FileNotFoundError(
        "No checkpoint .pt file found in WandB artifact download "
        f"directory: {artifact_dir}"
    )


def stage_resume_directory_from_wandb_artifact(
    artifact_dir: Path, destination_dir: Path
) -> Path:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact download directory does not exist: {artifact_dir}")
    if not artifact_dir.is_dir():
        raise NotADirectoryError(
            f"Artifact download path is not a directory: {artifact_dir}"
        )

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_checkpoint_dir = destination_dir / CHECKPOINT_DIRNAME
    destination_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_source = _choose_resume_checkpoint(artifact_dir)
    checkpoint_destination = destination_checkpoint_dir / LATEST_CHECKPOINT_FILENAME
    shutil.copy2(checkpoint_source, checkpoint_destination)

    source_training_data = artifact_dir / TRAINING_DATA_FILENAME
    if source_training_data.exists():
        shutil.copy2(source_training_data, destination_dir / TRAINING_DATA_FILENAME)
    else:
        logger.warning(
            "WandB artifact has no replay buffer snapshot",
            expected_path=str(source_training_data),
        )

    source_incumbent = artifact_dir / INCUMBENT_ONNX_FILENAME
    if source_incumbent.exists():
        destination_incumbent = destination_checkpoint_dir / INCUMBENT_ONNX_FILENAME
        try:
            copy_model_artifact_bundle(source_incumbent, destination_incumbent)
        except (FileNotFoundError, RuntimeError, OSError) as error:
            for stale_incumbent_path in destination_checkpoint_dir.glob("incumbent*"):
                stale_incumbent_path.unlink(missing_ok=True)
            logger.warning(
                "Failed to stage incumbent model artifact bundle from WandB; "
                "continuing without incumbent bundle",
                source=str(source_incumbent),
                destination=str(destination_incumbent),
                error=str(error),
            )
    else:
        logger.warning(
            "WandB artifact has no incumbent model artifact bundle",
            expected_path=str(source_incumbent),
        )

    logger.info(
        "Staged resume directory from WandB artifact download",
        artifact_dir=str(artifact_dir),
        destination_dir=str(destination_dir),
        checkpoint_source=str(checkpoint_source),
        checkpoint_destination=str(checkpoint_destination),
    )
    return destination_dir


def prepare_wandb_resume_source(
    reference: str, default_alias: str = "latest"
) -> WandbResumeSource:
    artifact_ref = resolve_wandb_model_artifact_reference(
        reference, default_alias=default_alias
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="tetris-wandb-resume-"))

    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_ref, type="model")
        downloaded_dir = Path(artifact.download(root=str(temp_dir / "download")))
        resume_dir = stage_resume_directory_from_wandb_artifact(
            downloaded_dir, temp_dir / "resume_source"
        )
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    logger.info(
        "Prepared WandB artifact resume source",
        reference=reference,
        artifact_ref=artifact_ref,
        downloaded_dir=str(downloaded_dir),
        resume_dir=str(resume_dir),
    )
    return WandbResumeSource(
        temp_dir=temp_dir, resume_dir=resume_dir, artifact_ref=artifact_ref
    )
