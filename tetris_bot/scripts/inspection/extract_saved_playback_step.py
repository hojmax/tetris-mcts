"""Extract one exact step from a saved full-game playback into a smaller playback file.

The output keeps the target step's full tree, but collapses earlier steps to
root-only stubs so the visualizer can reconstruct the exact state without
loading the whole stitched game graph.

Load the result with:

    make viz VIZ_ARGS="--saved_playback <path-to-generated-json>"
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import structlog
from simple_parsing import parse

from tetris_bot.constants import NUM_ACTIONS, PROJECT_ROOT
from tetris_bot.scripts.inspection.tree_playback_artifact import (
    load_tree_playback_artifact,
)

logger = structlog.get_logger()

HOLD_ACTION_INDEX = NUM_ACTIONS - 1


@dataclass
class ScriptArgs:
    saved_playback: Path
    step_index: int = 0
    output_path: Path | None = None


def collapse_step_to_root_only(step: dict[str, Any]) -> dict[str, Any]:
    tree = step["tree"]
    root_id = int(tree["root_id"])
    root_node = next(
        (node for node in tree["nodes"] if int(node["id"]) == root_id),
        None,
    )
    if root_node is None:
        raise ValueError(f"Step tree is missing root node {root_id}")

    collapsed_root = {
        **root_node,
        "children": [],
    }
    return {
        **step,
        "tree": {
            **tree,
            "nodes": [collapsed_root],
        },
    }


def default_output_path(saved_playback: Path, step_index: int) -> Path:
    return saved_playback.with_name(
        f"{saved_playback.stem}.step_{step_index:03d}.json"
    )


def build_step_slice_payload(
    payload: dict[str, Any],
    *,
    step_index: int,
) -> dict[str, Any]:
    steps = payload["steps"]
    replay_moves = payload["replay_moves"]
    if step_index < 0 or step_index >= len(steps):
        raise ValueError(
            f"step_index out of range: {step_index} (valid 0-{len(steps) - 1})"
        )

    sliced_steps = [
        steps[idx] if idx == step_index else collapse_step_to_root_only(steps[idx])
        for idx in range(step_index + 1)
    ]
    sliced_replay_moves = replay_moves[: step_index + 1]

    metadata = {
        **payload["metadata"],
        "source": "saved_playback_step_extract",
        "slice_step_index": int(step_index),
        "slice_collapsed_prefix_steps": int(step_index),
    }

    return {
        **payload,
        "metadata": metadata,
        "replay_moves": sliced_replay_moves,
        "steps": sliced_steps,
        "total_attack": sum(int(move["attack"]) for move in sliced_replay_moves),
        "num_moves": sum(
            1 for move in sliced_replay_moves if int(move["action"]) != HOLD_ACTION_INDEX
        ),
        "num_frames": len(sliced_replay_moves),
        "tree_reuse_hits": 0,
        "tree_reuse_misses": 0,
    }


def main(args: ScriptArgs) -> None:
    saved_playback = args.saved_playback.resolve()
    if not saved_playback.exists():
        raise FileNotFoundError(f"Saved playback file not found: {saved_playback}")

    payload = load_tree_playback_artifact(saved_playback)
    sliced = build_step_slice_payload(payload, step_index=args.step_index)

    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else default_output_path(saved_playback, args.step_index)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sliced, separators=(",", ":")))

    logger.info(
        "Wrote saved playback step extract",
        input_path=str(saved_playback),
        output_path=str(output_path),
        step_index=args.step_index,
        output_frames=int(sliced["num_frames"]),
        output_moves=int(sliced["num_moves"]),
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
