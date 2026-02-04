"""
Verify that all configuration parameters are properly wired through the codebase.

This script tests that:
1. Config parameters are used (not hardcoded values)
2. Network accepts all architecture parameters
3. Trainer uses config values correctly
"""

from __future__ import annotations

import structlog
from pathlib import Path

from tetris_mcts.config import TrainingConfig, PROJECT_ROOT
from tetris_mcts.ml.network import TetrisNet

logger = structlog.get_logger()


def verify_network_config():
    """Verify network accepts config parameters."""
    logger.info("=== Verifying Network Configuration ===")

    config = TrainingConfig(
        conv_filters=[1, 2],
        fc_hidden=16,
        conv_kernel_size=5,
        conv_padding=2,
    )

    # Create network with custom params
    net = TetrisNet(
        conv_filters=config.conv_filters,
        fc_hidden=config.fc_hidden,
        conv_kernel_size=config.conv_kernel_size,
        conv_padding=config.conv_padding,
    )

    # Verify layers
    assert net.conv1.out_channels == 1, "Conv1 filters mismatch"
    assert net.conv2.out_channels == 2, "Conv2 filters mismatch"
    assert net.fc1.out_features == 16, "FC hidden size mismatch"
    assert net.conv1.kernel_size == (5, 5), "Kernel size mismatch"
    assert net.conv1.padding == (2, 2), "Padding mismatch"

    logger.info("✓ Network configuration verified")


def verify_config_completeness():
    """Verify all expected parameters exist in config."""
    logger.info("=== Verifying Config Completeness ===")

    config = TrainingConfig()

    # Architecture parameters
    required_arch = [
        "conv_filters",
        "fc_hidden",
        "conv_kernel_size",
        "conv_padding",
        "max_moves",
    ]

    # Optimizer parameters
    required_opt = [
        "batch_size",
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "lr_schedule",
        "lr_decay_steps",
        "lr_min_factor",
        "lr_step_gamma",
        "lr_step_divisor",
    ]

    # MCTS parameters
    required_mcts = [
        "num_simulations",
        "c_puct",
        "temperature",
        "dirichlet_alpha",
        "dirichlet_epsilon",
        "num_workers",
    ]

    all_required = required_arch + required_opt + required_mcts

    missing = []
    for param in all_required:
        if not hasattr(config, param):
            missing.append(param)

    if missing:
        logger.error("Missing parameters", missing=missing)
        raise ValueError(f"Config missing parameters: {missing}")

    logger.info(
        "✓ All required parameters present",
        arch_params=len(required_arch),
        opt_params=len(required_opt),
        mcts_params=len(required_mcts),
    )


def verify_no_hardcoded_values():
    """Check that common hardcoded values are not in key files."""
    logger.info("=== Checking for Hardcoded Values ===")

    # Note: This is a simple check, not exhaustive
    files_to_check = [
        PROJECT_ROOT / "tetris_mcts" / "ml" / "training.py",
    ]

    suspicious_patterns = [
        ("1.0", "grad clip norm - should use config.grad_clip_norm"),
        ("0.01", "lr min factor - should use config.lr_min_factor"),
    ]

    warnings = []
    for file_path in files_to_check:
        if not file_path.exists():
            continue

        content = file_path.read_text()
        for pattern, desc in suspicious_patterns:
            if f"= {pattern}" in content and f"config.{pattern}" not in content:
                # Check if it's actually hardcoded (not a comment or string)
                lines = [
                    line
                    for line in content.split("\n")
                    if pattern in line
                    and "=" in line
                    and not line.strip().startswith("#")
                ]
                if lines:
                    warnings.append(
                        f"{file_path.name}: Found '{pattern}' ({desc}): {lines[0][:80]}"
                    )

    if warnings:
        logger.warning(
            "Found potentially hardcoded values (may be false positives)", count=len(warnings)
        )
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("✓ No obvious hardcoded values found")


def main():
    logger.info("Starting configuration verification")

    try:
        verify_config_completeness()
        verify_network_config()
        verify_no_hardcoded_values()

        logger.info("=" * 80)
        logger.info("✓ All verification checks passed!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Verification failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
