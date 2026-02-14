from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import zipfile

import numpy as np
from numpy.lib import format as npy_format
from simple_parsing import parse
import structlog

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    data_path: Path  # Path to replay buffer NPZ file
    dry_run: bool = False  # Analyze and report only, do not modify file
    create_backup: bool = False  # Save backup before in-place rewrite
    backup_suffix: str = ".pre_combo_normalize.bak"  # Backup filename suffix
    epsilon: float = 1e-6  # Tolerance used for normalization checks
    temp_suffix: str = ".tmp_combo_patch"  # Temporary filename suffix


@dataclass(frozen=True)
class ComboStats:
    min_value: float
    max_value: float
    has_intermediate_values: bool
    only_zero_or_one: bool
    appears_normalized: bool


def validate_args(args: ScriptArgs) -> None:
    if args.epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if not args.backup_suffix:
        raise ValueError("backup_suffix must be non-empty")
    if not args.temp_suffix:
        raise ValueError("temp_suffix must be non-empty")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {args.data_path}")
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.data_path}")


def load_combo_array(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, mmap_mode="r") as data:
        if "combos" not in data:
            raise KeyError(f"NPZ is missing required key 'combos': {npz_path}")
        combos = np.asarray(data["combos"], dtype=np.float32)

    if combos.ndim != 1:
        raise ValueError(f"Expected combos to be 1D (N,), got shape: {combos.shape}")
    if combos.size == 0:
        raise ValueError("combos array is empty")
    if not np.all(np.isfinite(combos)):
        raise ValueError("combos contains non-finite values")
    return combos


def analyze_combo_values(combos: np.ndarray, epsilon: float) -> ComboStats:
    min_value = float(np.min(combos))
    max_value = float(np.max(combos))

    near_zero = np.isclose(combos, 0.0, atol=epsilon)
    near_one = np.isclose(combos, 1.0, atol=epsilon)
    only_zero_or_one = bool(np.all(near_zero | near_one))

    has_intermediate_values = bool(
        np.any((combos > epsilon) & (combos < (1.0 - epsilon)))
    )

    appears_normalized = (
        min_value >= -epsilon
        and max_value <= (1.0 + epsilon)
        and has_intermediate_values
    )

    return ComboStats(
        min_value=min_value,
        max_value=max_value,
        has_intermediate_values=has_intermediate_values,
        only_zero_or_one=only_zero_or_one,
        appears_normalized=appears_normalized,
    )


def should_normalize(stats: ComboStats) -> bool:
    if stats.max_value > 1.0:
        return True
    if stats.only_zero_or_one and not stats.has_intermediate_values:
        return True
    return not stats.appears_normalized


def normalize_combos(combos: np.ndarray) -> np.ndarray:
    return (combos / 12.0).astype(np.float32, copy=False)


def rewrite_npz_with_combos(
    source_path: Path, target_path: Path, combos: np.ndarray
) -> None:
    combos_member_name = "combos.npy"
    replaced = False
    with zipfile.ZipFile(source_path, "r") as source_zip:
        if combos_member_name not in source_zip.namelist():
            raise KeyError(f"NPZ zip entry missing: {combos_member_name}")

        with zipfile.ZipFile(
            target_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as target_zip:
            for source_info in source_zip.infolist():
                member_name = source_info.filename
                if member_name == combos_member_name:
                    with target_zip.open(member_name, mode="w", force_zip64=True) as f:
                        npy_format.write_array(
                            f, np.asarray(combos), allow_pickle=False
                        )
                    replaced = True
                    continue

                with source_zip.open(member_name, mode="r") as source_file:
                    with target_zip.open(
                        member_name,
                        mode="w",
                        force_zip64=True,
                    ) as target_file:
                        shutil.copyfileobj(source_file, target_file, length=1024 * 1024)

    if not replaced:
        raise RuntimeError("Failed to replace combos.npy while rewriting NPZ")


def verify_combos(npz_path: Path, expected: np.ndarray, epsilon: float) -> None:
    actual = load_combo_array(npz_path)
    if actual.shape != expected.shape:
        raise ValueError(
            f"Patched combos shape mismatch: expected {expected.shape}, got {actual.shape}"
        )
    if not np.allclose(actual, expected, atol=epsilon, rtol=0.0):
        raise ValueError(
            "Patched combos values do not match expected normalized values"
        )


def build_backup_path(npz_path: Path, backup_suffix: str) -> Path:
    return npz_path.parent / f"{npz_path.name}{backup_suffix}"


def build_temp_path(npz_path: Path, temp_suffix: str) -> Path:
    return npz_path.parent / f"{npz_path.name}{temp_suffix}"


def main(args: ScriptArgs) -> None:
    validate_args(args)

    combos = load_combo_array(args.data_path)
    pre_stats = analyze_combo_values(combos, args.epsilon)
    normalize_needed = should_normalize(pre_stats)

    logger.info(
        "Combo feature analysis",
        path=str(args.data_path),
        min_value=pre_stats.min_value,
        max_value=pre_stats.max_value,
        has_intermediate_values=pre_stats.has_intermediate_values,
        only_zero_or_one=pre_stats.only_zero_or_one,
        appears_normalized=pre_stats.appears_normalized,
        normalize_needed=normalize_needed,
    )

    if not normalize_needed:
        logger.info(
            "No changes made because combos already appears normalized under current rules",
            path=str(args.data_path),
        )
        return

    normalized = normalize_combos(combos)
    post_stats = analyze_combo_values(normalized, args.epsilon)
    logger.info(
        "Post-normalization combo stats",
        min_value=post_stats.min_value,
        max_value=post_stats.max_value,
        has_intermediate_values=post_stats.has_intermediate_values,
        only_zero_or_one=post_stats.only_zero_or_one,
        appears_normalized=post_stats.appears_normalized,
    )

    if args.dry_run:
        logger.info("Dry-run enabled; skipping NPZ rewrite", path=str(args.data_path))
        return

    backup_path = build_backup_path(args.data_path, args.backup_suffix)
    if args.create_backup:
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        shutil.copy2(args.data_path, backup_path)
        logger.info("Created backup", backup_path=str(backup_path))

    temp_path = build_temp_path(args.data_path, args.temp_suffix)
    if temp_path.exists():
        raise FileExistsError(f"Temporary output path already exists: {temp_path}")

    try:
        rewrite_npz_with_combos(args.data_path, temp_path, normalized)
        temp_path.replace(args.data_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    verify_combos(args.data_path, normalized, args.epsilon)
    logger.info("Patched combo normalization in-place", path=str(args.data_path))


if __name__ == "__main__":
    main(parse(ScriptArgs))
