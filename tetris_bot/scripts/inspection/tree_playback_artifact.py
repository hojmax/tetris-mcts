"""Save and load compact full-game MCTS tree playback artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from tetris_core.tetris_core import MCTSConfig, TetrisEnv
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    QUEUE_SIZE,
)

logger = structlog.get_logger()

TREE_PLAYBACK_FORMAT_VERSION = 1
HOLD_ACTION_INDEX = NUM_ACTIONS - 1
NO_CHANCE_OUTCOME = NUM_PIECE_TYPES


def _action_reveals_new_visible_piece(env: TetrisEnv, action_idx: int) -> bool:
    return action_idx != HOLD_ACTION_INDEX or env.get_hold_piece() is None


def _search_config_payload(config: MCTSConfig) -> dict[str, Any]:
    return {
        "num_simulations": int(config.num_simulations),
        "c_puct": float(config.c_puct),
        "temperature": float(config.temperature),
        "dirichlet_alpha": float(config.dirichlet_alpha),
        "dirichlet_epsilon": float(config.dirichlet_epsilon),
        "visit_sampling_epsilon": float(config.visit_sampling_epsilon),
        "max_placements": int(config.max_placements),
        "reuse_tree": bool(config.reuse_tree),
        "use_parent_value_for_unvisited_q": bool(
            config.use_parent_value_for_unvisited_q
        ),
        "nn_value_weight": float(config.nn_value_weight),
        "death_penalty": float(config.death_penalty),
        "overhang_penalty_weight": float(config.overhang_penalty_weight),
        "mcts_seed": int(config.seed) if config.seed is not None else None,
    }


def _serialize_tree_node(node: object) -> dict[str, Any]:
    return {
        "id": int(node.id),
        "node_type": str(node.node_type),
        "visit_count": int(node.visit_count),
        "value_sum": float(node.value_sum),
        "mean_value": float(node.mean_value),
        "value_history": [float(value) for value in node.value_history],
        "nn_value": float(node.nn_value),
        "unvisited_child_value_estimate": float(node.unvisited_child_value_estimate),
        "is_terminal": bool(node.is_terminal),
        "move_number": int(node.move_number),
        "attack": int(node.attack),
        "cumulative_attack": int(node.state.attack),
        "parent_id": int(node.parent_id) if node.parent_id is not None else None,
        "edge_from_parent": (
            int(node.edge_from_parent) if node.edge_from_parent is not None else None
        ),
        "children": [int(child_id) for child_id in node.children],
        "valid_actions": [int(action) for action in node.valid_actions],
        "action_priors": [float(prior) for prior in node.action_priors],
    }


def _saved_tree_q_bounds(saved_tree: dict[str, Any]) -> dict[str, float | str] | None:
    q_min = saved_tree.get("q_min")
    q_max = saved_tree.get("q_max")
    if q_min is not None and q_max is not None:
        return {
            "q_min": float(q_min),
            "q_max": float(q_max),
            "source": "export",
        }

    root_id = int(saved_tree["root_id"])
    root_node = next(
        (node for node in saved_tree["nodes"] if int(node["id"]) == root_id),
        None,
    )
    if root_node is None:
        return None

    value_history = [float(value) for value in root_node.get("value_history", [])]
    if not value_history:
        return None
    return {
        "q_min": min(value_history),
        "q_max": max(value_history),
        "source": "root_value_history",
    }


def save_tree_playback_artifact(
    playback: object,
    output_path: Path,
    *,
    initial_seed: int,
    config: MCTSConfig,
    add_noise: bool,
    model_path: Path | None = None,
    source: str = "python_export",
    candidate_step: int = 0,
    promoted: bool = False,
    candidate_avg_attack: float = 0.0,
    evaluation_seconds: float = 0.0,
) -> dict[str, Any]:
    """Write a compact full-game tree playback artifact to JSON."""
    payload = {
        "metadata": {
            "format_version": TREE_PLAYBACK_FORMAT_VERSION,
            "source": source,
            "initial_seed": int(initial_seed),
            "board_width": BOARD_WIDTH,
            "board_height": BOARD_HEIGHT,
            "add_noise": bool(add_noise),
            "model_path": str(model_path) if model_path is not None else "",
            "candidate_step": int(candidate_step),
            "promoted": bool(promoted),
            "candidate_avg_attack": float(candidate_avg_attack),
            "evaluation_seconds": float(evaluation_seconds),
            "search_config": _search_config_payload(config),
        },
        "replay_moves": [
            {"action": int(move.action), "attack": int(move.attack)}
            for move in playback.replay_moves
        ],
        "steps": [
            {
                "frame_index": int(step.frame_index),
                "placement_count": int(step.placement_count),
                "selected_action": int(step.selected_action),
                "selected_chance_outcome": int(step.selected_chance_outcome),
                "attack": int(step.attack),
                "tree": {
                    "nodes": [_serialize_tree_node(node) for node in step.tree.nodes],
                    "root_id": int(step.tree.root_id),
                    "num_simulations": int(step.tree.num_simulations),
                    "selected_action": int(step.tree.selected_action),
                    "policy": [float(prob) for prob in step.tree.policy],
                    "q_min": float(step.tree.q_min),
                    "q_max": float(step.tree.q_max),
                },
            }
            for step in playback.steps
        ],
        "total_attack": int(playback.total_attack),
        "num_moves": int(playback.num_moves),
        "num_frames": int(playback.num_frames),
        "tree_reuse_hits": int(playback.tree_reuse_hits),
        "tree_reuse_misses": int(playback.tree_reuse_misses),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    logger.info("Saved tree playback artifact", path=str(output_path))
    return payload


def load_tree_playback_artifact(path: Path) -> dict[str, Any]:
    """Load a compact full-game tree playback artifact from JSON."""
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    version = metadata.get("format_version")
    if version != TREE_PLAYBACK_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported tree playback format_version {version} in {path}; "
            f"expected {TREE_PLAYBACK_FORMAT_VERSION}"
        )
    return payload


def _possible_chance_outcomes(env: TetrisEnv) -> list[int]:
    if env.get_queue_len() >= QUEUE_SIZE:
        return [NO_CHANCE_OUTCOME]
    outcomes = [int(piece) for piece in env.get_possible_next_pieces()]
    outcomes.sort()
    return outcomes


def _reconstruct_step_tree(
    saved_tree: dict[str, Any],
    step_root_env: TetrisEnv,
    *,
    search_step: int,
    is_reuse_root: bool,
) -> tuple[list[dict[str, Any]], dict[int, TetrisEnv]]:
    nodes = saved_tree["nodes"]
    env_cache: dict[int, TetrisEnv] = {}
    node_type_by_id = {int(node["id"]): str(node["node_type"]) for node in nodes}
    node_rows: list[dict[str, Any]] = []

    for node in nodes:
        node_id = int(node["id"])
        parent_id = node.get("parent_id")
        edge_from_parent = node.get("edge_from_parent")

        if parent_id is None:
            env = step_root_env.clone_state()
        else:
            parent_env = env_cache[int(parent_id)].clone_state()
            parent_type = node_type_by_id[int(parent_id)]
            if parent_type == "decision":
                action_idx = int(edge_from_parent)
                reveals_new_visible_piece = _action_reveals_new_visible_piece(
                    parent_env, action_idx
                )
                result = parent_env.execute_action_index(action_idx)
                if result is None:
                    raise ValueError(
                        f"Failed to reconstruct decision edge action {action_idx} "
                        f"from parent node {parent_id}"
                    )
                visible_queue_len = (
                    QUEUE_SIZE - 1 if reveals_new_visible_piece else QUEUE_SIZE
                )
                parent_env.truncate_queue(visible_queue_len)
            else:
                chance_outcome = int(edge_from_parent)
                if chance_outcome < NUM_PIECE_TYPES:
                    parent_env.push_queue_piece(chance_outcome)
                elif chance_outcome != NO_CHANCE_OUTCOME:
                    raise ValueError(
                        f"Invalid chance outcome {chance_outcome} for node {node_id}"
                    )
            env = parent_env

        env_cache[node_id] = env
        current_piece = env.get_current_piece()
        hold_piece = env.get_hold_piece()
        node_rows.append(
            {
                **node,
                "board": list(env.get_board()),
                "board_piece_types": list(env.get_board_piece_types()),
                "current_piece": (
                    int(current_piece.piece_type) if current_piece is not None else None
                ),
                "hold_piece": (
                    int(hold_piece.piece_type) if hold_piece is not None else None
                ),
                "cumulative_attack": int(env.attack),
                "queue": [int(piece) for piece in env.get_queue(QUEUE_SIZE)],
                "possible_chance_outcomes": (
                    _possible_chance_outcomes(env)
                    if str(node["node_type"]) == "chance"
                    else []
                ),
                "search_step": search_step,
                "is_reuse_root": is_reuse_root
                and node_id == int(saved_tree["root_id"]),
            }
        )

    return node_rows, env_cache


def build_tree_dict_from_saved_playback(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, TetrisEnv]]:
    """Reconstruct visualizer tree/env data from a saved playback artifact."""
    metadata = payload["metadata"]
    if (
        int(metadata.get("board_width", BOARD_WIDTH)) != BOARD_WIDTH
        or int(metadata.get("board_height", BOARD_HEIGHT)) != BOARD_HEIGHT
    ):
        raise ValueError(
            "Saved playback board size does not match current project constants "
            f"({metadata.get('board_width')}x{metadata.get('board_height')} vs "
            f"{BOARD_WIDTH}x{BOARD_HEIGHT})"
        )
    search_config = metadata["search_config"]
    initial_seed = int(metadata["initial_seed"])
    replay_moves = payload["replay_moves"]

    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, initial_seed)
    combined_nodes: list[dict[str, Any]] = []
    combined_env_cache: dict[int, TetrisEnv] = {}
    highlighted_node_ids: set[str] = set()
    highlighted_edge_keys: list[dict[str, str]] = []
    reuse_edges: list[dict[str, str]] = []
    search_step_q_bounds: dict[int, dict[str, float | str]] = {}
    root_id = 0
    previous_path_target: str | None = None
    replay_prefix_index = 0

    for step_index, step in enumerate(payload["steps"]):
        q_bounds = _saved_tree_q_bounds(step["tree"])
        if q_bounds is not None:
            search_step_q_bounds[step_index] = q_bounds
        step_root_env = env.clone_state()
        step_nodes, step_env_cache = _reconstruct_step_tree(
            step["tree"],
            step_root_env,
            search_step=step_index,
            is_reuse_root=step_index > 0,
        )
        node_offset = len(combined_nodes)
        step_root_id = int(step["tree"]["root_id"]) + node_offset
        if step_index == 0:
            root_id = step_root_id

        for node in step_nodes:
            node_id = int(node["id"]) + node_offset
            combined_nodes.append(
                {
                    **node,
                    "id": node_id,
                    "children": [
                        int(child_id) + node_offset for child_id in node["children"]
                    ],
                    "parent_id": (
                        int(node["parent_id"]) + node_offset
                        if node["parent_id"] is not None
                        else None
                    ),
                }
            )

        for node_id, node_env in step_env_cache.items():
            combined_env_cache[node_id + node_offset] = node_env

        highlighted_node_ids.add(str(step_root_id))

        selected_action = int(step["selected_action"])
        selected_chance_outcome = int(step["selected_chance_outcome"])
        action_target = f"v_{step_root_id}_{selected_action}"
        chance_target: str | None = None

        for node in combined_nodes:
            if (
                node["parent_id"] == step_root_id
                and node["edge_from_parent"] == selected_action
            ):
                action_target = str(node["id"])
                break

        highlighted_node_ids.add(action_target)
        highlighted_edge_keys.append(
            {"source": str(step_root_id), "target": action_target}
        )

        for node in combined_nodes:
            if (
                node["parent_id"] is not None
                and str(node["parent_id"]) == action_target
                and node["edge_from_parent"] == selected_chance_outcome
            ):
                chance_target = str(node["id"])
                break

        if chance_target is None:
            chance_target = f"vp_{action_target}_{selected_chance_outcome}"
        highlighted_node_ids.add(chance_target)
        if chance_target is not None:
            highlighted_edge_keys.append(
                {"source": action_target, "target": chance_target}
            )

        if previous_path_target is not None:
            reuse_edges.append(
                {
                    "source": previous_path_target,
                    "target": str(step_root_id),
                    "label": "reuse",
                    "classes": "reuse-edge chosen-edge",
                }
            )
            highlighted_edge_keys.append(
                {"source": previous_path_target, "target": str(step_root_id)}
            )
        previous_path_target = chance_target

        replay_move = replay_moves[replay_prefix_index]
        replay_prefix_index += 1
        result = env.execute_action_index(int(replay_move["action"]))
        if result is None:
            raise ValueError(
                f"Failed to replay action {replay_move['action']} for saved playback step {step_index}"
            )

    total_edges = payload["tree_reuse_hits"] + payload["tree_reuse_misses"]
    tree_dict = {
        "nodes": combined_nodes,
        "root_id": root_id,
        "selected_action": (
            int(payload["steps"][0]["selected_action"]) if payload["steps"] else None
        ),
        "num_simulations": search_config["num_simulations"],
        "c_puct": search_config["c_puct"],
        "use_parent_value_for_unvisited_q": search_config[
            "use_parent_value_for_unvisited_q"
        ],
        "mode": "full_game",
        "counter_label": (
            f"Full game: {payload['num_moves']} placements, {payload['num_frames']} frames, "
            f"Atk {payload['total_attack']}, reuse {payload['tree_reuse_hits']}/{total_edges}, "
            f"nodes {len(combined_nodes)}"
        ),
        "highlighted_node_ids": sorted(highlighted_node_ids),
        "highlighted_edge_keys": highlighted_edge_keys,
        "reuse_edges": reuse_edges,
        "total_attack": int(payload["total_attack"]),
        "num_moves": int(payload["num_moves"]),
        "num_frames": int(payload["num_frames"]),
        "tree_reuse_hits": int(payload["tree_reuse_hits"]),
        "tree_reuse_misses": int(payload["tree_reuse_misses"]),
        "search_step_q_bounds": search_step_q_bounds,
    }
    return tree_dict, combined_env_cache
