"""
Weight Export/Import Pipeline for Tetris AlphaZero

Supports:
- PyTorch checkpoint save/load
- ONNX export for Rust inference
- Simple binary format for manual Rust loading
"""

import math
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import struct
import json

from tetris_mcts.ml.network import TetrisNet, BOARD_HEIGHT, BOARD_WIDTH, AUX_FEATURES


def save_checkpoint(
    model: TetrisNet,
    optimizer: Optional[torch.optim.Optimizer],
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

    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | Path,
    model: Optional[TetrisNet] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
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
    import warnings
    import logging

    try:
        # ONNX export must happen on CPU
        original_device = next(model.parameters()).device
        model_cpu = model.cpu()
        model_cpu.eval()

        # Create dummy inputs (batch size 1, tract handles this)
        dummy_board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        dummy_aux = torch.zeros(1, AUX_FEATURES)

        # Suppress ONNX export warnings
        onnx_logger = logging.getLogger("torch.onnx")
        old_level = onnx_logger.level
        onnx_logger.setLevel(logging.ERROR)

        # Export without dynamic_axes for tract compatibility
        with warnings.catch_warnings():
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

        onnx_logger.setLevel(old_level)

        # Move model back to original device
        model.to(original_device)
        return True
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Warning: ONNX export skipped (missing dependencies): {e}")
        return False


def export_binary(
    model: TetrisNet,
    filepath: str | Path,
) -> None:
    """
    Export model weights in simple binary format for manual Rust loading.

    Format:
        - Header: "TNET" (4 bytes)
        - Version: u32 (4 bytes)
        - Num tensors: u32 (4 bytes)
        - For each tensor:
            - Name length: u32
            - Name: bytes
            - Num dimensions: u32
            - Dimensions: u32 * num_dims
            - Data: f32 * product(dims)
    """
    model.eval()
    state_dict = model.state_dict()

    with open(filepath, "wb") as f:
        # Header
        f.write(b"TNET")
        f.write(struct.pack("<I", 1))  # Version 1
        f.write(struct.pack("<I", len(state_dict)))  # Num tensors

        for name, tensor in state_dict.items():
            # Convert to numpy
            data = tensor.cpu().numpy().astype(np.float32)

            # Name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Dimensions
            f.write(struct.pack("<I", len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack("<I", dim))

            # Data (flattened, row-major)
            f.write(data.tobytes())


def load_binary(
    filepath: str | Path,
    model: TetrisNet,
) -> None:
    """
    Load model weights from binary format.

    Args:
        filepath: Path to binary weights file
        model: Model to load weights into
    """
    with open(filepath, "rb") as f:
        # Header
        header = f.read(4)
        assert header == b"TNET", f"Invalid header: {header}"

        version = struct.unpack("<I", f.read(4))[0]
        assert version == 1, f"Unsupported version: {version}"

        num_tensors = struct.unpack("<I", f.read(4))[0]

        state_dict = {}
        for _ in range(num_tensors):
            # Name
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")

            # Dimensions
            num_dims = struct.unpack("<I", f.read(4))[0]
            dims = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(num_dims))

            # Data
            num_elements = math.prod(dims)
            data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
            data = data.reshape(dims)

            state_dict[name] = torch.tensor(data)

        model.load_state_dict(state_dict)


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


def load_metadata(filepath: str | Path) -> dict:
    """Load metadata from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


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
        step: int,
        eval_metrics: Optional[dict] = None,
        export_for_rust: bool = True,
    ) -> dict[str, Path]:
        """
        Save checkpoint and optionally export for Rust.

        Returns paths to saved files.
        """
        paths = {}

        # Save PyTorch checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        save_checkpoint(model, optimizer, step, ckpt_path)
        paths["checkpoint"] = ckpt_path

        if export_for_rust:
            # Export ONNX
            onnx_path = self.checkpoint_dir / "latest.onnx"
            export_onnx(model, onnx_path)
            paths["onnx"] = onnx_path

            # Export binary
            bin_path = self.checkpoint_dir / "latest.bin"
            export_binary(model, bin_path)
            paths["binary"] = bin_path

        # Save metadata
        meta_path = self.checkpoint_dir / "latest_metadata.json"
        export_metadata(meta_path, step, eval_metrics)
        paths["metadata"] = meta_path

        # Update symlink to latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(ckpt_path.name)
        paths["latest"] = latest_path

        return paths

    def load_latest(
        self,
        model: TetrisNet,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[int]:
        """
        Load the latest checkpoint.

        Returns:
            Training step of loaded checkpoint, or None if no checkpoint exists
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return None

        state = load_checkpoint(latest_path, model, optimizer)
        return state.get("step", 0)

    def get_checkpoints(self) -> list[Path]:
        """Get list of all checkpoint files, sorted by step."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints

    def cleanup_old_checkpoints(self, keep: int = 5) -> None:
        """Keep only the N most recent checkpoints."""
        checkpoints = self.get_checkpoints()
        for ckpt in checkpoints[:-keep]:
            ckpt.unlink()


if __name__ == "__main__":
    import tempfile
    import os

    print("Testing weight export/import...")

    # Create model
    model = TetrisNet()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test checkpoint save/load
    print("\nTesting checkpoint save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        optimizer = torch.optim.Adam(model.parameters())

        save_checkpoint(model, optimizer, step=100, filepath=ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        model2 = TetrisNet()
        state = load_checkpoint(ckpt_path, model2)
        print(f"Loaded checkpoint at step {state['step']}")

        # Verify weights match
        for k in model.state_dict():
            assert torch.allclose(model.state_dict()[k], model2.state_dict()[k])
        print("Weights match!")

    # Test ONNX export
    print("\nTesting ONNX export...")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    try:
        export_onnx(model, onnx_path)
        print(f"Exported ONNX to {onnx_path}")
        print(f"File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"ONNX export failed (may need onnx package): {e}")
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

    # Test binary export
    print("\nTesting binary export/load...")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    export_binary(model, bin_path)
    print(f"Exported binary to {bin_path}")
    print(f"File size: {os.path.getsize(bin_path) / 1024 / 1024:.2f} MB")

    model3 = TetrisNet()
    load_binary(bin_path, model3)
    print("Loaded binary weights")

    # Verify weights match
    for k in model.state_dict():
        assert torch.allclose(model.state_dict()[k], model3.state_dict()[k], atol=1e-6)
    print("Weights match!")

    os.unlink(bin_path)

    # Test WeightManager
    print("\nTesting WeightManager...")
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = WeightManager(tmpdir)
        optimizer = torch.optim.Adam(model.parameters())

        paths = manager.save(model, optimizer, step=100, export_for_rust=True)
        print(f"Saved files: {list(paths.keys())}")

        model4 = TetrisNet()
        step = manager.load_latest(model4)
        print(f"Loaded latest at step {step}")

    print("\nAll tests passed!")
