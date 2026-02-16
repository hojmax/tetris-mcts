"""
Weight Export/Import Pipeline for Tetris AlphaZero

Supports:
- PyTorch checkpoint save/load
- ONNX export for Rust inference
"""

import contextlib
import io
import json
import logging
import struct
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import structlog

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_FILENAME_PREFIX,
    LATEST_CHECKPOINT_FILENAME,
    LATEST_METADATA_FILENAME,
    LATEST_ONNX_FILENAME,
)
from tetris_mcts.ml.network import AUX_FEATURES, PIECE_AUX_FEATURES, ConvBackbone, HeadsModel, TetrisNet

logger = structlog.get_logger()


def save_checkpoint(
    model: TetrisNet,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    step: int,
    filepath: str | Path,
    **extra_state,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The TetrisNet model
        optimizer: Optimizer state (optional)
        step: Training step number
        filepath: Path to save to
        **extra_state: Additional state to save
    """
    state = {
        "model_state_dict": model.state_dict(),
        "step": step,
        **extra_state,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | Path,
    model: Optional[TetrisNet] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> dict:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)

    Returns:
        Checkpoint state dict
    """
    state = torch.load(filepath, map_location="cpu", weights_only=True)

    if model is not None:
        model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        if "scheduler_state_dict" not in state:
            logger.warning(
                "Checkpoint missing scheduler state; using fresh scheduler state",
                checkpoint=str(filepath),
            )
        else:
            scheduler.load_state_dict(state["scheduler_state_dict"])

    return state


def export_onnx(
    model: TetrisNet,
    filepath: str | Path,
    opset_version: int = 18,
) -> bool:
    """
    Export model to ONNX format for Rust inference (tract-onnx).

    Args:
        model: The TetrisNet model
        filepath: Output path (should end in .onnx)
        opset_version: ONNX opset version (17 needed for LayerNorm)

    Returns:
        True if export succeeded, False if ONNX dependencies are missing
    """
    # ONNX export must happen on CPU
    original_device = next(model.parameters()).device
    if original_device.type == "cuda":
        torch.cuda.empty_cache()
    onnx_logger = logging.getLogger("torch.onnx")
    old_level = onnx_logger.level
    try:
        model.cpu()
        model.eval()

        # Create dummy inputs (batch size 1, tract handles this)
        dummy_board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        dummy_aux = torch.zeros(1, AUX_FEATURES)

        # Suppress ONNX export warnings
        onnx_logger.setLevel(logging.ERROR)

        # Export without dynamic_axes for tract compatibility
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                model,
                (dummy_board, dummy_aux),
                filepath,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["board", "aux_features"],
                output_names=["policy_logits", "value"],
                verbose=False,
            )
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("ONNX export skipped; missing dependencies", error=str(e))
        return False
    finally:
        onnx_logger.setLevel(old_level)
        model.to(original_device)


def _split_paths(onnx_path: str | Path) -> tuple[Path, Path, Path]:
    """Derive split model paths from the base ONNX path."""
    base = Path(onnx_path).with_suffix("")
    return (
        base.with_suffix(".conv.onnx"),
        base.with_suffix(".heads.onnx"),
        base.with_suffix(".fc.bin"),
    )


def split_model_paths(onnx_path: str | Path) -> tuple[Path, Path, Path]:
    """Return paths for the split-model artifacts associated with an ONNX file."""
    return _split_paths(onnx_path)


def export_split_models(
    model: TetrisNet,
    onnx_path: str | Path,
    opset_version: int = 18,
) -> bool:
    """Export split models (conv.onnx, heads.onnx, fc.bin) for cached Rust inference."""
    conv_path, heads_path, fc_path = _split_paths(onnx_path)
    original_device = next(model.parameters()).device
    if original_device.type == "cuda":
        torch.cuda.empty_cache()
    onnx_logger = logging.getLogger("torch.onnx")
    old_level = onnx_logger.level
    try:
        model.cpu()
        model.eval()
        onnx_logger.setLevel(logging.ERROR)

        conv_backbone = ConvBackbone(model)
        conv_backbone.eval()
        dummy_board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                conv_backbone,
                (dummy_board,),
                str(conv_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["board"],
                output_names=["conv_out"],
                verbose=False,
            )

        heads_model = HeadsModel(model)
        heads_model.eval()
        board_hidden = model.board_proj.out_features
        dummy_board_h = torch.zeros(1, board_hidden)
        dummy_aux = torch.zeros(1, PIECE_AUX_FEATURES)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                heads_model,
                (dummy_board_h, dummy_aux),
                str(heads_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["board_h", "piece_aux"],
                output_names=["policy_logits", "value"],
                verbose=False,
            )

        # Export board projection weight and bias as raw f32 binary
        # Format: [rows u32 LE][cols u32 LE][weight row-major f32][bias f32]
        weight = model.board_proj.weight.detach().numpy()  # (hidden, conv_flat)
        bias = model.board_proj.bias.detach().numpy()  # (hidden,)
        rows, cols = weight.shape
        with open(fc_path, "wb") as f:
            f.write(struct.pack("<II", rows, cols))
            f.write(weight.astype(np.float32).tobytes())
            f.write(bias.astype(np.float32).tobytes())

        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("Split model export skipped; missing dependencies", error=str(e))
        return False
    finally:
        onnx_logger.setLevel(old_level)
        model.to(original_device)


def export_metadata(
    filepath: str | Path,
    step: int,
    eval_metrics: Optional[dict] = None,
    config: Optional[dict] = None,
) -> None:
    """
    Export metadata alongside weights for tracking.

    Args:
        filepath: Path to JSON file
        step: Training step
        eval_metrics: Optional evaluation metrics
        config: Optional training config
    """
    metadata = {
        "step": step,
        "eval_metrics": eval_metrics or {},
        "config": config or {},
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


class WeightManager:
    """
    Manages weight files for training/inference coordination.

    Handles:
    - Saving checkpoints at intervals
    - Exporting weights for Rust self-play
    - Loading latest weights
    """

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: TetrisNet,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        step: int,
        eval_metrics: Optional[dict] = None,
        export_for_rust: bool = True,
        extra_checkpoint_state: dict[str, object] | None = None,
    ) -> dict[str, Path]:
        """
        Save checkpoint and optionally export for Rust.

        Returns paths to saved files.
        """
        paths = {}

        # Save PyTorch checkpoint
        ckpt_path = self.checkpoint_dir / f"{CHECKPOINT_FILENAME_PREFIX}_{step}.pt"
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            step,
            ckpt_path,
            **(extra_checkpoint_state or {}),
        )
        paths["checkpoint"] = ckpt_path

        if export_for_rust:
            # Export ONNX (full model — used as watch sentinel for hot-swap)
            onnx_path = self.checkpoint_dir / LATEST_ONNX_FILENAME
            onnx_export_ok = export_onnx(model, onnx_path)
            if not onnx_export_ok:
                raise RuntimeError(
                    "ONNX export failed due to missing dependencies while saving checkpoint"
                )
            paths["onnx"] = onnx_path

            # Export split models for cached inference
            split_export_ok = export_split_models(model, onnx_path)
            if not split_export_ok:
                raise RuntimeError(
                    "Split-model export failed due to missing dependencies while saving checkpoint"
                )

        # Save metadata
        meta_path = self.checkpoint_dir / LATEST_METADATA_FILENAME
        export_metadata(meta_path, step, eval_metrics)
        paths["metadata"] = meta_path

        # Update symlink to latest checkpoint
        latest_path = self.checkpoint_dir / LATEST_CHECKPOINT_FILENAME
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(ckpt_path.name)
        paths["latest"] = latest_path

        return paths

    def load_latest(
        self,
        model: TetrisNet,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Optional[int]:
        """
        Load the latest checkpoint.

        Returns:
            Training step of loaded checkpoint, or None if no checkpoint exists
        """
        latest_path = self.checkpoint_dir / LATEST_CHECKPOINT_FILENAME
        if not latest_path.exists():
            return None

        state = load_checkpoint(latest_path, model, optimizer, scheduler)
        return state.get("step", 0)

    def get_checkpoints(self) -> list[Path]:
        """Get list of all checkpoint files, sorted by step."""
        checkpoints = list(
            self.checkpoint_dir.glob(f"{CHECKPOINT_FILENAME_PREFIX}_*.pt")
        )
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints
