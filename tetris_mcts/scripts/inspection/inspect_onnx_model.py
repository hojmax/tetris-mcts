"""
Inspect ONNX model architecture.

Shows layer shapes and parameter counts to verify which architecture is loaded.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from pathlib import Path
from simple_parsing import parse

logger = structlog.get_logger()


@dataclass
class InspectArgs:
    """Arguments for ONNX inspection."""

    model_path: Path


def inspect_onnx(model_path: Path) -> None:
    """Inspect ONNX model structure."""
    try:
        import onnx
    except ImportError:
        logger.error("onnx package not installed. Install with: pip install onnx")
        return

    if not model_path.exists():
        logger.error("Model not found", path=model_path)
        return

    logger.info("Loading ONNX model", path=model_path)
    model = onnx.load(str(model_path))

    logger.info("=" * 80)
    logger.info("ONNX Model Architecture")
    logger.info("=" * 80)

    # Count parameters
    total_params = 0
    for initializer in model.graph.initializer:
        dims = initializer.dims
        params = 1
        for d in dims:
            params *= d
        total_params += params

        # Show key layers
        name = initializer.name
        if any(
            key in name
            for key in [
                "conv_initial.weight",
                "conv_reduce.weight",
                "res_blocks.",
                "bn_initial",
                "bn_reduce",
                "board_proj.weight",
                "aux_fc.weight",
                "gate_fc.weight",
                "aux_proj.weight",
                "policy_head.weight",
                "value_head.weight",
            ]
        ):
            logger.info(f"  {name}: {list(dims)}")

    logger.info("=" * 80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info("=" * 80)

    # Infer architecture from conv layer shapes
    init_channels = None
    reduce_channels = None
    num_res_blocks = 0
    for initializer in model.graph.initializer:
        if "conv_initial.weight" in initializer.name:
            init_channels = initializer.dims[0]
        elif "conv_reduce.weight" in initializer.name:
            reduce_channels = initializer.dims[0]
        elif "res_blocks." in initializer.name and ".conv1.weight" in initializer.name:
            num_res_blocks += 1

    has_gating = any(
        "gate_fc.weight" in init.name for init in model.graph.initializer
    )

    if init_channels is not None:
        logger.info(
            "Detected deep conv backbone" + (" with gated-fusion" if has_gating else ""),
            trunk_channels=init_channels,
            num_res_blocks=num_res_blocks,
            reduction_channels=reduce_channels,
        )
    else:
        logger.warning("Could not detect conv backbone architecture from initializer names")


def main() -> None:
    args = parse(InspectArgs)
    inspect_onnx(args.model_path)


if __name__ == "__main__":
    main()
