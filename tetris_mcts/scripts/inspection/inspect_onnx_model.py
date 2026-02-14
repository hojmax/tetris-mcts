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
            for key in ["conv1.weight", "conv2.weight", "fc1.weight", "bn1", "bn2"]
        ):
            logger.info(f"  {name}: {list(dims)}")
        if any(
            key in name
            for key in [
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

    # Infer architecture from conv1.weight shape
    conv1_weight = None
    for initializer in model.graph.initializer:
        if "conv1.weight" in initializer.name:
            conv1_weight = list(initializer.dims)
            break

    if conv1_weight:
        out_channels = conv1_weight[0]  # [out, in, h, w]
        logger.info(
            "Detected architecture",
            conv1_filters=out_channels,
            note="Check conv2.weight for second layer",
        )

        # Find conv2
        conv2_filters = None
        for initializer in model.graph.initializer:
            if "conv2.weight" in initializer.name:
                conv2_weight = list(initializer.dims)
                conv2_filters = conv2_weight[0]
                logger.info(f"  Conv filters: [{out_channels}, {conv2_filters}]")
                break

        # Determine model family
        has_gating = any(
            "gate_fc.weight" in init.name for init in model.graph.initializer
        )
        has_concat_fc = any(
            "fc1.weight" in init.name for init in model.graph.initializer
        )

        if conv2_filters is None:
            logger.warning("Could not locate conv2.weight in model initializers")
        elif has_gating:
            logger.info(
                "✓ Detected gated-fusion architecture",
                conv_filters=[out_channels, conv2_filters],
            )
        elif has_concat_fc:
            logger.info(
                "✓ Detected concat+fc architecture",
                conv_filters=[out_channels, conv2_filters],
            )
        else:
            logger.info(
                f"Unrecognized architecture family: [{out_channels}, {conv2_filters}] filters"
            )


def main() -> None:
    args = parse(InspectArgs)
    inspect_onnx(args.model_path)


if __name__ == "__main__":
    main()
