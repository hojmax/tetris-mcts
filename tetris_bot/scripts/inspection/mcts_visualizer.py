"""
MCTS Tree Visualizer for Tetris.

Interactive visualization of the MCTS search tree from the Rust implementation.
Uses Dash + Cytoscape for interactive graph exploration with:
- Drag and drop node positioning
- Zoom and pan
- Click to view node details (board state, value estimates, visit counts)
- Step-through simulation capability
"""

import base64
import io
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dash
from dash import (
    html,
    dcc,
    callback,
    Output,
    Input,
    State,
    ALL,
    clientside_callback,
)
import dash_cytoscape as cyto
from PIL import Image, ImageDraw
from simple_parsing import parse

from tetris_core.tetris_core import TetrisEnv, MCTSAgent, MCTSConfig
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    INCUMBENT_ONNX_FILENAME,
    NUM_PIECE_TYPES,
    PIECE_COLORS,
    PIECE_NAMES,
    PROJECT_ROOT,
    QUEUE_SIZE,
)
from tetris_bot.scripts.inspection.tree_playback_artifact import (
    build_tree_dict_from_saved_playback,
    load_tree_playback_artifact,
)
from tetris_bot.scripts.utils.run_search_config import load_effective_self_play_config

# Global cache for TetrisEnv states (keyed by node ID)
# This allows us to clone states and execute actions for visualization
_env_cache: dict[int, TetrisEnv] = {}
NO_CHANCE_OUTCOME = NUM_PIECE_TYPES
NO_CHANCE_OUTCOME_LABEL = "NOOP"


def _render_board_image(
    board: list[list[int]],
    board_piece_types: list[list[int | None]],
    cell_size: int = 12,
) -> str:
    height = len(board)
    width = len(board[0]) if board else 10
    # board_piece_types may be empty for MCTS lightweight-cloned states
    has_piece_types = len(board_piece_types) == height

    img = Image.new("RGB", (width * cell_size, height * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                color = (80, 80, 80)
                if has_piece_types:
                    color_idx = board_piece_types[y][x]
                    if color_idx is not None and color_idx < len(PIECE_COLORS):
                        color = PIECE_COLORS[color_idx]

                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

    for x in range(width + 1):
        draw.line(
            [(x * cell_size, 0), (x * cell_size, height * cell_size)], fill=(40, 40, 40)
        )
    for y in range(height + 1):
        draw.line(
            [(0, y * cell_size), (width * cell_size, y * cell_size)], fill=(40, 40, 40)
        )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@dataclass
class ScriptArgs:
    run_dir: Path = (  # Training run dir (default: training_runs/v41)
        PROJECT_ROOT / "training_runs" / "v41"
    )
    use_dummy_network: bool = False  # Use uniform-prior/zero-value bootstrap search
    state_preset: Path | None = (
        None  # Optional Python file exposing VIZ_STATE_PRESET dict
    )
    saved_playback: Path | None = (
        None  # Optional compact full-game tree playback artifact to preload
    )


PIECE_TOKEN_TO_INDEX = {
    "I": 0,
    "O": 1,
    "T": 2,
    "S": 3,
    "Z": 4,
    "J": 5,
    "L": 6,
}


def load_saved_playback_defaults(args: ScriptArgs) -> dict[str, Any] | None:
    if args.saved_playback is None:
        return None

    saved_path = args.saved_playback.resolve()
    if not saved_path.exists():
        raise ValueError(f"Saved playback file does not exist: {saved_path}")

    payload = load_tree_playback_artifact(saved_path)
    tree_dict, env_cache = build_tree_dict_from_saved_playback(payload)
    metadata = payload["metadata"]
    return {
        "path": saved_path,
        "payload": payload,
        "tree_dict": tree_dict,
        "env_cache": env_cache,
        "seed": int(metadata["initial_seed"]),
    }


def load_viz_defaults(
    args: ScriptArgs, saved_playback: dict[str, Any] | None
) -> dict[str, str | int | float | bool | None]:
    if saved_playback is not None:
        metadata = saved_playback["payload"]["metadata"]
        config = metadata["search_config"]
        return {
            "model_path": str(metadata.get("model_path", "")),
            "num_simulations": int(config["num_simulations"]),
            "c_puct": float(config["c_puct"]),
            "temperature": float(config["temperature"]),
            "dirichlet_alpha": float(config["dirichlet_alpha"]),
            "dirichlet_epsilon": float(config["dirichlet_epsilon"]),
            "max_placements": int(config["max_placements"]),
            "q_scale": (
                float(config["q_scale"]) if config.get("q_scale") is not None else None
            ),
            "use_dummy_network": args.use_dummy_network,
        }

    run_dir = args.run_dir
    config_path = run_dir / CONFIG_FILENAME
    model_path = run_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME

    if not run_dir.exists():
        raise ValueError(f"Run dir does not exist: {run_dir}")
    if not config_path.exists():
        raise ValueError(f"Missing run config: {config_path}")
    if not args.use_dummy_network and not model_path.exists():
        raise ValueError(f"Missing incumbent ONNX checkpoint: {model_path}")

    config = load_effective_self_play_config(run_dir)
    required_keys = [
        "num_simulations",
        "c_puct",
        "temperature",
        "dirichlet_alpha",
        "dirichlet_epsilon",
        "max_placements",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Run config missing required keys: {', '.join(sorted(missing_keys))}"
        )

    use_tanh = config.get("use_tanh_q_normalization", True)
    q_scale = float(config["q_scale"]) if use_tanh and "q_scale" in config else None

    return {
        "model_path": str(model_path),
        "num_simulations": int(config["num_simulations"]),
        "c_puct": float(config["c_puct"]),
        "temperature": float(config["temperature"]),
        "dirichlet_alpha": float(config["dirichlet_alpha"]),
        "dirichlet_epsilon": float(config["dirichlet_epsilon"]),
        "max_placements": int(config["max_placements"]),
        "q_scale": q_scale,
        "use_dummy_network": args.use_dummy_network,
    }


def _format_piece_for_input(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        if value < 0 or value >= NUM_PIECE_TYPES:
            raise ValueError(f"Piece index out of range: {value}")
        return PIECE_NAMES[value]
    if isinstance(value, str):
        token = value.strip().upper()
        if token == "":
            return ""
        if token.isdigit():
            piece_idx = int(token)
            if piece_idx < 0 or piece_idx >= NUM_PIECE_TYPES:
                raise ValueError(f"Piece index out of range: {piece_idx}")
            return PIECE_NAMES[piece_idx]
        if token not in PIECE_TOKEN_TO_INDEX:
            raise ValueError(
                f"Invalid piece token '{value}' (expected I,O,T,S,Z,J,L or 0-6)"
            )
        piece_idx = PIECE_TOKEN_TO_INDEX[token]
        return PIECE_NAMES[piece_idx]
    raise ValueError(f"Unsupported piece value type: {type(value).__name__}")


def _format_hold_used_for_input(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "":
            return ""
        if token in {"1", "true", "t", "yes", "y"}:
            return "true"
        if token in {"0", "false", "f", "no", "n"}:
            return "false"
        raise ValueError(f"Invalid hold_used token '{value}' (expected true/false)")
    raise ValueError(f"Unsupported hold_used value type: {type(value).__name__}")


def _format_queue_for_input(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        token = value.strip()
        if token == "":
            return ""
        raw_tokens = [item.strip() for item in token.replace("|", ",").split(",")]
        pieces = [_format_piece_for_input(item) for item in raw_tokens if item != ""]
        return ",".join(pieces)
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            pieces.append(_format_piece_for_input(item))
        return ",".join(pieces)
    raise ValueError(f"Unsupported queue value type: {type(value).__name__}")


def _format_board_for_input(value: object | None) -> str:
    if value is None:
        return ""
    allowed = set("._0IO TSZJL1234567")
    if isinstance(value, str):
        token = value.strip("\n")
        if token == "":
            return ""
        lines = [line.strip() for line in token.splitlines() if line.strip() != ""]
        if len(lines) != BOARD_HEIGHT:
            raise ValueError(
                f"Board must have exactly {BOARD_HEIGHT} non-empty rows, got {len(lines)}"
            )
        for row_idx, line in enumerate(lines):
            compact = line.replace(" ", "")
            if len(compact) != BOARD_WIDTH:
                raise ValueError(
                    f"Board row {row_idx + 1} must have {BOARD_WIDTH} cells, got {len(compact)}"
                )
            if any(char.upper() not in allowed for char in compact):
                raise ValueError(f"Invalid board row {row_idx + 1}: {line}")
        return token
    if isinstance(value, list):
        rows: list[str] = []
        for row in value:
            if not isinstance(row, str):
                raise ValueError(
                    "Board list values must be strings with one row per entry"
                )
            rows.append(row)
        board_text = "\n".join(rows)
        return _format_board_for_input(board_text)
    raise ValueError(f"Unsupported board value type: {type(value).__name__}")


def load_state_preset_defaults(
    args: ScriptArgs,
) -> dict[str, int | str | None]:
    defaults: dict[str, int | str | None] = {
        "seed": 42,
        "move_number": 0,
        "current_piece": "",
        "hold_piece": "",
        "hold_used": "",
        "queue": "",
        "board": "",
    }

    if args.state_preset is None:
        return defaults

    preset_path = args.state_preset.resolve()
    if not preset_path.exists():
        raise ValueError(f"State preset file does not exist: {preset_path}")

    suffix = preset_path.suffix.lower()
    if suffix == ".json":
        preset = json.loads(preset_path.read_text())
    else:
        module_name = f"_viz_state_preset_{preset_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, preset_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load state preset module: {preset_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preset = getattr(module, "VIZ_STATE_PRESET", None)

    if not isinstance(preset, dict):
        raise ValueError(
            (
                f"Invalid state preset payload in {preset_path}: "
                "expected a JSON object or a Python module defining VIZ_STATE_PRESET"
            )
        )

    if "seed" in preset:
        defaults["seed"] = int(preset["seed"])
    if "move_number" in preset:
        defaults["move_number"] = int(preset["move_number"])
    if "current_piece" in preset:
        defaults["current_piece"] = _format_piece_for_input(preset["current_piece"])
    if "hold_piece" in preset:
        defaults["hold_piece"] = _format_piece_for_input(preset["hold_piece"])
    if "hold_used" in preset:
        defaults["hold_used"] = _format_hold_used_for_input(preset["hold_used"])
    if "queue" in preset:
        defaults["queue"] = _format_queue_for_input(preset["queue"])
    if "board" in preset:
        defaults["board"] = _format_board_for_input(preset["board"])

    return defaults


SCRIPT_ARGS = parse(ScriptArgs)
SAVED_PLAYBACK_DEFAULTS = load_saved_playback_defaults(SCRIPT_ARGS)
VIZ_DEFAULTS = load_viz_defaults(SCRIPT_ARGS, SAVED_PLAYBACK_DEFAULTS)
STATE_PRESET_DEFAULTS = load_state_preset_defaults(SCRIPT_ARGS)
if SAVED_PLAYBACK_DEFAULTS is not None:
    STATE_PRESET_DEFAULTS["seed"] = SAVED_PLAYBACK_DEFAULTS["seed"]

Q_NORMALIZATION_EPSILON = 1e-6
VALUE_HISTORY_INLINE_LIMIT = 64
VALUE_HISTORY_EDGE_SAMPLE = 8


def normalize_q_value(q: float, q_min: float, q_max: float) -> float:
    q_range = q_max - q_min
    if abs(q_range) < Q_NORMALIZATION_EPSILON:
        return 0.5
    return (q - q_min) / q_range


def squash_q_value(q: float, q_scale: float) -> float:
    return math.tanh(q / q_scale)


def transform_q(
    raw_q: float, q_scale: float | None, q_min: float, q_max: float
) -> float:
    if q_scale is not None:
        return squash_q_value(raw_q, q_scale)
    return normalize_q_value(raw_q, q_min, q_max)


def format_chance_outcome(outcome_idx: int) -> str:
    if outcome_idx == NO_CHANCE_OUTCOME:
        return NO_CHANCE_OUTCOME_LABEL
    if 0 <= outcome_idx < NUM_PIECE_TYPES:
        return PIECE_NAMES[outcome_idx]
    return f"P{outcome_idx}"


def get_possible_chance_outcomes_for_env(env: TetrisEnv) -> list[int]:
    if env.get_queue_len() >= QUEUE_SIZE:
        return [NO_CHANCE_OUTCOME]
    outcomes = list(env.get_possible_next_pieces())
    outcomes.sort()
    return outcomes


def build_decision_action_stats(
    decision_node: dict, tree_dict: dict, c_puct: float, q_scale: float | None
) -> list[dict]:
    if not decision_node["valid_actions"]:
        return []

    action_to_prior = dict(
        zip(decision_node["valid_actions"], decision_node["action_priors"])
    )
    child_by_action: dict[int, dict] = {}
    for child_id in decision_node["children"]:
        child = tree_dict["nodes"][child_id]
        action_idx = child.get("edge_from_parent")
        if action_idx is not None:
            child_by_action[action_idx] = child

    raw_q_by_action: dict[int, float] = {}
    unvisited_raw_q = (
        float(decision_node.get("unvisited_child_value_estimate", 0.0))
        if tree_dict.get("use_parent_value_for_unvisited_q", False)
        else 0.0
    )
    for action_idx in decision_node["valid_actions"]:
        child = child_by_action.get(action_idx)
        raw_q_by_action[action_idx] = (
            child["mean_value"] if child is not None else unvisited_raw_q
        )

    q_min = min(raw_q_by_action.values())
    q_max = max(raw_q_by_action.values())
    # Rust MCTS increments visit_count BEFORE calling select_action, so the
    # next simulation will use sqrt(visit_count + 1) for PUCT computation.
    sqrt_parent = (decision_node["visit_count"] + 1) ** 0.5

    action_stats: list[dict] = []
    for action_idx in decision_node["valid_actions"]:
        prior = action_to_prior[action_idx]
        child = child_by_action.get(action_idx)
        visits = child["visit_count"] if child is not None else 0
        raw_q = raw_q_by_action[action_idx]
        transformed_q = transform_q(raw_q, q_scale, q_min, q_max)
        u_value = c_puct * prior * sqrt_parent / (1 + visits)
        puct_total = transformed_q + u_value

        parent_id = decision_node["id"]
        if child is not None:
            target_node_id = str(child["id"])
        else:
            target_node_id = f"v_{parent_id}_{action_idx}"

        action_stats.append(
            {
                "action": action_idx,
                "prior": prior,
                "visits": visits,
                "raw_q": raw_q,
                "q_transformed": transformed_q,
                "u": u_value,
                "puct": puct_total,
                "child": child,
                "target_node_id": target_node_id,
            }
        )

    return action_stats


def build_value_history_details(node: dict) -> list:
    value_history = node.get("value_history", [])
    if not value_history:
        return []

    details = [html.P(f"Backed-up values tracked: {len(value_history)}")]
    if len(value_history) <= VALUE_HISTORY_INLINE_LIMIT:
        terms = " + ".join(f"{value:.6f}" for value in value_history)
        details.append(
            html.P(
                f"Q = ({terms}) / {len(value_history)} = {node['mean_value']:.6f}",
                style={
                    "fontFamily": "monospace",
                    "fontSize": "11px",
                    "overflowWrap": "anywhere",
                },
            )
        )
        return details

    head_terms = " + ".join(
        f"{value:.6f}" for value in value_history[:VALUE_HISTORY_EDGE_SAMPLE]
    )
    tail_terms = " + ".join(
        f"{value:.6f}" for value in value_history[-VALUE_HISTORY_EDGE_SAMPLE:]
    )
    details.append(
        html.P(
            (
                "Q preview = "
                f"({head_terms} + ... + {tail_terms}) / {len(value_history)} = "
                f"{node['mean_value']:.6f}"
            ),
            style={
                "fontFamily": "monospace",
                "fontSize": "11px",
                "overflowWrap": "anywhere",
            },
        )
    )
    details.append(
        html.P(
            f"Showing first/last {VALUE_HISTORY_EDGE_SAMPLE} values only for performance.",
            style={"fontSize": "11px", "color": "#666"},
        )
    )
    return details


def build_action_stats_table(action_stats: list[dict], q_col: str):
    sorted_rows = sorted(action_stats, key=lambda row: row["puct"], reverse=True)
    header_cells = [
        html.Th("Action", style={"textAlign": "left", "padding": "4px 8px"}),
        html.Th("N", style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th("P", style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th(q_col, style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th("Qraw", style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th("U", style={"textAlign": "right", "padding": "4px 8px", "color": "#0066cc"}),
        html.Th(
            "PUCT",
            style={"textAlign": "right", "padding": "4px 8px", "fontWeight": "bold"},
        ),
        html.Th("Atk", style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th("NNval", style={"textAlign": "right", "padding": "4px 8px"}),
    ]

    body_rows = []
    for row in sorted_rows:
        action_label = f"a{row['action']}"
        target_node_id = row["target_node_id"]
        child = row["child"]
        body_rows.append(
            html.Tr(
                [
                    html.Td(
                        html.Button(
                            action_label,
                            id={"type": "nav-button", "target": target_node_id},
                            n_clicks=0,
                            style={
                                "background": "none",
                                "border": "none",
                                "padding": 0,
                                "cursor": "pointer",
                                "color": "#0066cc",
                                "textDecoration": "underline",
                                "fontFamily": "monospace",
                                "fontSize": "12px",
                            },
                        ),
                        style={"padding": "4px 8px", "textAlign": "left"},
                    ),
                    html.Td(f"{row['visits']}", style={"padding": "4px 8px", "textAlign": "right"}),
                    html.Td(f"{row['prior']:.4f}", style={"padding": "4px 8px", "textAlign": "right"}),
                    html.Td(
                        f"{row['q_transformed']:.4f}",
                        style={"padding": "4px 8px", "textAlign": "right"},
                    ),
                    html.Td(f"{row['raw_q']:.4f}", style={"padding": "4px 8px", "textAlign": "right"}),
                    html.Td(
                        f"{row['u']:.4f}",
                        style={"padding": "4px 8px", "textAlign": "right", "color": "#0066cc"},
                    ),
                    html.Td(
                        f"{row['puct']:.4f}",
                        style={"padding": "4px 8px", "textAlign": "right", "fontWeight": "bold"},
                    ),
                    html.Td(
                        "" if child is None else f"{child['attack']}",
                        style={"padding": "4px 8px", "textAlign": "right"},
                    ),
                    html.Td(
                        "" if child is None else f"{child['nn_value']:.4f}",
                        style={"padding": "4px 8px", "textAlign": "right"},
                    ),
                ]
            )
        )

    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr(header_cells)),
                html.Tbody(body_rows),
            ],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontFamily": "monospace",
                "fontSize": "12px",
            },
        ),
        style={"overflowX": "auto"},
    )


def build_chance_outcome_stats(chance_node: dict, tree_dict: dict) -> list[dict]:
    child_by_outcome: dict[int, dict] = {}
    for child_id in chance_node.get("children", []):
        child = _find_tree_node(tree_dict, child_id)
        if child is None:
            continue
        outcome_idx = child.get("edge_from_parent")
        if outcome_idx is not None:
            child_by_outcome[int(outcome_idx)] = child

    outcomes: list[dict] = []
    seen_outcomes: set[int] = set()
    for outcome_idx, child in child_by_outcome.items():
        seen_outcomes.add(outcome_idx)
        outcomes.append(
            {
                "outcome_idx": outcome_idx,
                "label": format_chance_outcome(outcome_idx),
                "visits": int(child["visit_count"]),
                "q": float(child["mean_value"]),
                "nn_value": float(child["nn_value"]),
                "target_node_id": str(child["id"]),
                "is_virtual": False,
            }
        )

    for outcome_idx in chance_node.get("possible_chance_outcomes", []):
        outcome_idx = int(outcome_idx)
        if outcome_idx in seen_outcomes:
            continue
        outcomes.append(
            {
                "outcome_idx": outcome_idx,
                "label": format_chance_outcome(outcome_idx),
                "visits": 0,
                "q": 0.0,
                "nn_value": None,
                "target_node_id": f"vp_{chance_node['id']}_{outcome_idx}",
                "is_virtual": True,
            }
        )

    outcomes.sort(key=lambda row: (row["visits"], not row["is_virtual"]), reverse=True)
    return outcomes


def build_chance_outcomes_table(outcome_stats: list[dict]):
    header_cells = [
        html.Th("Outcome", style={"textAlign": "left", "padding": "4px 8px"}),
        html.Th("N", style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th("Qraw", style={"textAlign": "right", "padding": "4px 8px"}),
        html.Th("NNval", style={"textAlign": "right", "padding": "4px 8px"}),
    ]

    body_rows = []
    for row in outcome_stats:
        body_rows.append(
            html.Tr(
                [
                    html.Td(
                        html.Button(
                            row["label"],
                            id={"type": "nav-button", "target": row["target_node_id"]},
                            n_clicks=0,
                            style={
                                "background": "none",
                                "border": "none",
                                "padding": 0,
                                "cursor": "pointer",
                                "color": "#0066cc",
                                "textDecoration": "underline",
                                "fontFamily": "monospace",
                                "fontSize": "12px",
                            },
                        ),
                        style={"padding": "4px 8px", "textAlign": "left"},
                    ),
                    html.Td(f"{row['visits']}", style={"padding": "4px 8px", "textAlign": "right"}),
                    html.Td(f"{row['q']:.4f}", style={"padding": "4px 8px", "textAlign": "right"}),
                    html.Td(
                        "" if row["nn_value"] is None else f"{row['nn_value']:.4f}",
                        style={"padding": "4px 8px", "textAlign": "right"},
                    ),
                ]
            )
        )

    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr(header_cells)),
                html.Tbody(body_rows),
            ],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontFamily": "monospace",
                "fontSize": "12px",
            },
        ),
        style={"overflowX": "auto"},
    )


def build_env_cache_for_tree(tree) -> dict[int, TetrisEnv]:
    env_cache: dict[int, TetrisEnv] = {}
    node_types = {n.id: n.node_type for n in tree.nodes}

    for node in tree.nodes:
        if node.parent_id is None:
            env_cache[node.id] = node.state.clone_state()
            continue

        if node.parent_id in env_cache and node.edge_from_parent is not None:
            parent_env = env_cache[node.parent_id].clone_state()
            if node_types[node.parent_id] == "decision":
                result = parent_env.execute_action_index(node.edge_from_parent)
                if result is None:
                    env_cache[node.id] = node.state.clone_state()
                    continue
                parent_env.truncate_queue(QUEUE_SIZE)
            elif node.edge_from_parent < NUM_PIECE_TYPES:
                parent_env.push_queue_piece(node.edge_from_parent)
            env_cache[node.id] = parent_env
            continue

        env_cache[node.id] = node.state.clone_state()

    return env_cache


def tree_export_to_dict(
    tree,
    c_puct: float,
    q_scale: float | None,
    use_parent_value_for_unvisited_q: bool,
    search_step: int = 0,
    is_reuse_root: bool = False,
) -> dict:
    env_cache = build_env_cache_for_tree(tree)

    return {
        "nodes": [
            {
                "id": node.id,
                "node_type": node.node_type,
                "visit_count": node.visit_count,
                "mean_value": node.mean_value,
                "value_sum": node.value_sum,
                "value_history": list(node.value_history),
                "nn_value": node.nn_value,
                "unvisited_child_value_estimate": node.unvisited_child_value_estimate,
                "attack": node.attack,
                "cumulative_attack": int(node.state.attack),
                "is_terminal": node.is_terminal,
                "move_number": node.move_number,
                "valid_actions": list(node.valid_actions),
                "action_priors": list(node.action_priors),
                "children": list(node.children),
                "parent_id": node.parent_id,
                "edge_from_parent": node.edge_from_parent,
                "board": list(node.state.get_board()),
                "board_piece_types": list(
                    env_cache[node.id].get_board_piece_types()
                    if node.id in env_cache
                    else node.state.get_board_piece_types()
                ),
                "current_piece": node.state.get_current_piece().piece_type
                if node.state.get_current_piece()
                else None,
                "hold_piece": node.state.get_hold_piece().piece_type
                if node.state.get_hold_piece()
                else None,
                "queue": list(node.state.get_queue(5)),
                "possible_chance_outcomes": (
                    get_possible_chance_outcomes_for_env(node.state)
                    if node.node_type == "chance"
                    else []
                ),
                "search_step": search_step,
                "is_reuse_root": is_reuse_root and node.id == tree.root_id,
            }
            for node in tree.nodes
        ],
        "root_id": tree.root_id,
        "selected_action": tree.selected_action,
        "num_simulations": tree.num_simulations,
        "c_puct": c_puct,
        "q_scale": q_scale,
        "use_parent_value_for_unvisited_q": use_parent_value_for_unvisited_q,
        "mode": "single_search",
        "counter_label": f"Sims: {tree.num_simulations}",
        "highlighted_node_ids": [],
        "highlighted_edge_keys": [],
        "reuse_edges": [],
    }


def offset_tree_dict(tree_dict: dict, node_offset: int) -> dict:
    offset_nodes: list[dict] = []
    for node in tree_dict["nodes"]:
        offset_nodes.append(
            {
                **node,
                "id": node["id"] + node_offset,
                "children": [child_id + node_offset for child_id in node["children"]],
                "parent_id": (
                    node["parent_id"] + node_offset
                    if node["parent_id"] is not None
                    else None
                ),
            }
        )

    return {
        **tree_dict,
        "nodes": offset_nodes,
        "root_id": tree_dict["root_id"] + node_offset,
    }


def find_selected_path_targets(
    tree_dict: dict,
    selected_action: int,
    selected_chance_outcome: int,
) -> tuple[str, str | None]:
    nodes_by_id = {node["id"]: node for node in tree_dict["nodes"]}
    root = nodes_by_id.get(tree_dict["root_id"])
    if root is None:
        raise ValueError(f"Root node not found: {tree_dict['root_id']}")

    action_target = f"v_{root['id']}_{selected_action}"
    selected_chance_node: dict | None = None

    for child_id in root["children"]:
        child = nodes_by_id.get(child_id)
        if child is None:
            continue
        if child.get("edge_from_parent") == selected_action:
            action_target = str(child["id"])
            selected_chance_node = child
            break

    if selected_chance_node is None:
        return action_target, None

    chance_target = f"vp_{selected_chance_node['id']}_{selected_chance_outcome}"
    for child_id in selected_chance_node["children"]:
        child = nodes_by_id.get(child_id)
        if child is None:
            continue
        if child.get("edge_from_parent") == selected_chance_outcome:
            chance_target = str(child["id"])
            break

    return action_target, chance_target


def build_full_game_tree_dict(
    playback,
    c_puct: float,
    q_scale: float | None,
    use_parent_value_for_unvisited_q: bool,
) -> dict:
    combined_nodes: list[dict] = []
    highlighted_node_ids: set[str] = set()
    highlighted_edge_keys: list[dict[str, str]] = []
    reuse_edges: list[dict[str, str]] = []
    root_id = 0
    previous_path_target: str | None = None

    for step_index, step in enumerate(playback.steps):
        step_tree = tree_export_to_dict(
            step.tree,
            c_puct=c_puct,
            q_scale=q_scale,
            use_parent_value_for_unvisited_q=use_parent_value_for_unvisited_q,
            search_step=step_index,
            is_reuse_root=step_index > 0,
        )
        step_tree = offset_tree_dict(step_tree, len(combined_nodes))

        if step_index == 0:
            root_id = step_tree["root_id"]

        combined_nodes.extend(step_tree["nodes"])
        step_root_id = str(step_tree["root_id"])
        highlighted_node_ids.add(step_root_id)

        action_target, chance_target = find_selected_path_targets(
            step_tree,
            selected_action=step.selected_action,
            selected_chance_outcome=step.selected_chance_outcome,
        )
        highlighted_node_ids.add(action_target)
        highlighted_edge_keys.append({"source": step_root_id, "target": action_target})

        if chance_target is not None:
            highlighted_node_ids.add(chance_target)
            highlighted_edge_keys.append(
                {"source": action_target, "target": chance_target}
            )

        if previous_path_target is not None:
            reuse_edge = {
                "source": previous_path_target,
                "target": step_root_id,
                "label": "reuse",
                "classes": "reuse-edge chosen-edge",
            }
            reuse_edges.append(reuse_edge)
            highlighted_edge_keys.append(
                {
                    "source": previous_path_target,
                    "target": step_root_id,
                }
            )

        previous_path_target = chance_target

    return {
        "nodes": combined_nodes,
        "root_id": root_id,
        "selected_action": playback.steps[0].selected_action
        if playback.steps
        else None,
        "num_simulations": playback.steps[0].tree.num_simulations
        if playback.steps
        else 0,
        "c_puct": c_puct,
        "q_scale": q_scale,
        "mode": "full_game",
        "counter_label": (
            f"Full game: {playback.num_moves} placements, {playback.num_frames} frames, "
            f"Atk {playback.total_attack}, reuse {playback.tree_reuse_hits}/"
            f"{playback.tree_reuse_hits + playback.tree_reuse_misses}, "
            f"nodes {len(combined_nodes)}"
        ),
        "highlighted_node_ids": sorted(highlighted_node_ids),
        "highlighted_edge_keys": highlighted_edge_keys,
        "reuse_edges": reuse_edges,
        "total_attack": playback.total_attack,
        "num_moves": playback.num_moves,
        "num_frames": playback.num_frames,
        "tree_reuse_hits": playback.tree_reuse_hits,
        "tree_reuse_misses": playback.tree_reuse_misses,
    }


def build_full_game_env_cache(playback) -> dict[int, TetrisEnv]:
    combined_env_cache: dict[int, TetrisEnv] = {}
    node_offset = 0

    for step in playback.steps:
        step_env_cache = build_env_cache_for_tree(step.tree)
        for node_id, env in step_env_cache.items():
            combined_env_cache[node_id + node_offset] = env
        node_offset += len(step.tree.nodes)

    return combined_env_cache


def parse_piece_token(token: str) -> int:
    value = token.strip().upper()
    if value == "":
        raise ValueError("Piece token is empty")
    if value.isdigit():
        piece_idx = int(value)
        if piece_idx < 0 or piece_idx >= NUM_PIECE_TYPES:
            raise ValueError(f"Invalid piece index '{value}' (expected 0-6)")
        return piece_idx
    if value not in PIECE_TOKEN_TO_INDEX:
        raise ValueError(
            f"Invalid piece token '{token}' (expected I,O,T,S,Z,J,L or 0-6)"
        )
    return PIECE_TOKEN_TO_INDEX[value]


def parse_optional_piece(piece_text: str | None) -> int | None:
    if piece_text is None or piece_text.strip() == "":
        return None
    if piece_text.strip().lower() in {"none", "null", "-"}:
        return None
    return parse_piece_token(piece_text)


def parse_optional_hold_used(hold_used_text: str | None) -> bool | None:
    if hold_used_text is None or hold_used_text.strip() == "":
        return None
    normalized = hold_used_text.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(
        f"Invalid hold-used value '{hold_used_text}' (expected true/false)"
    )


def parse_optional_queue(queue_text: str | None) -> list[int] | None:
    if queue_text is None or queue_text.strip() == "":
        return None
    raw_tokens = [token.strip() for token in queue_text.replace("|", ",").split(",")]
    tokens = [token for token in raw_tokens if token != ""]
    if len(tokens) == 0:
        return None
    return [parse_piece_token(token) for token in tokens]


def parse_optional_board(
    board_text: str | None,
) -> tuple[list[list[int]], list[list[int | None]]] | None:
    if board_text is None or board_text.strip() == "":
        return None

    board_rows: list[list[int]] = []
    piece_rows: list[list[int | None]] = []
    lines = [line.strip() for line in board_text.splitlines() if line.strip() != ""]
    if len(lines) != BOARD_HEIGHT:
        raise ValueError(
            f"Board must have exactly {BOARD_HEIGHT} non-empty rows, got {len(lines)}"
        )

    for row_idx, line in enumerate(lines):
        compact = line.replace(" ", "")
        if len(compact) != BOARD_WIDTH:
            raise ValueError(
                f"Board row {row_idx + 1} must have {BOARD_WIDTH} cells, got {len(compact)}"
            )

        board_row: list[int] = []
        piece_row: list[int | None] = []
        for char in compact:
            upper_char = char.upper()
            if upper_char in {".", "_", "0"}:
                board_row.append(0)
                piece_row.append(None)
            elif upper_char in PIECE_TOKEN_TO_INDEX:
                board_row.append(1)
                piece_row.append(PIECE_TOKEN_TO_INDEX[upper_char])
            elif upper_char in {"1", "2", "3", "4", "5", "6", "7"}:
                board_row.append(1)
                piece_row.append(int(upper_char) - 1)
            else:
                raise ValueError(
                    f"Invalid board character '{char}' (expected .,0,I,O,T,S,Z,J,L,1-7)"
                )
        board_rows.append(board_row)
        piece_rows.append(piece_row)

    return board_rows, piece_rows


def apply_custom_state(
    env: TetrisEnv,
    current_piece_text: str | None,
    hold_piece_text: str | None,
    hold_used_text: str | None,
    queue_text: str | None,
    board_text: str | None,
) -> str | None:
    try:
        board_data = parse_optional_board(board_text)
        if board_data is not None:
            board_rows, piece_rows = board_data
            env.set_board(board_rows)
            env.set_board_piece_types(piece_rows)

        queue = parse_optional_queue(queue_text)
        if queue is not None:
            env.set_queue(queue)

        current_piece = parse_optional_piece(current_piece_text)
        if current_piece is not None:
            env.set_current_piece_type(current_piece)

        hold_piece = parse_optional_piece(hold_piece_text)
        if hold_piece_text is not None and hold_piece_text.strip() != "":
            env.set_hold_piece_type(hold_piece)

        hold_used = parse_optional_hold_used(hold_used_text)
        if hold_used is not None:
            env.set_hold_used(hold_used)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to apply custom state: {e}"

    return None


def display_virtual_node(node_data, tree_dict, c_puct, q_scale):
    """Display details for a virtual (unvisited) node."""
    # Parse virtual node ID: v_parentid_actionidx
    parts = node_data["id"].split("_")
    if len(parts) != 3:
        return "Invalid virtual node", "", ""

    parent_id = int(parts[1])
    action_idx = int(parts[2])

    if parent_id >= len(tree_dict["nodes"]):
        return "Parent not found", "", ""

    parent = tree_dict["nodes"][parent_id]
    action_stats = build_decision_action_stats(parent, tree_dict, c_puct, q_scale)
    stats_by_action = {row["action"]: row for row in action_stats}
    if action_idx not in stats_by_action:
        return "Action not found in parent valid actions", "", ""
    action_row = stats_by_action[action_idx]
    q_min = min(row["raw_q"] for row in action_stats)
    q_max = max(row["raw_q"] for row in action_stats)

    # Try to compute the resulting board by executing the action
    attack = None
    env_copy = None
    if parent_id in _env_cache:
        env_copy = _env_cache[parent_id].clone_state()
        attack = env_copy.execute_action_index(action_idx)

    # Format details
    details = [
        html.H4(
            "Unvisited Action",
            style={"marginTop": 0, "marginBottom": "10px", "color": "#ff8844"},
        ),
        html.P(f"Action Index: {action_idx}"),
        html.P(f"Parent Node: D{parent_id}"),
        html.P(f"Attack: {attack if attack is not None else '?'}"),
        html.P(
            f"Cumulative Attack: {env_copy.attack if env_copy is not None else parent.get('cumulative_attack', '?')}"
        ),
        html.Hr(),
        html.H4("PUCT Components"),
        html.P(f"Prior (P): {action_row['prior']:.4f}", style={"fontWeight": "bold"}),
        html.P(f"Q (raw): {action_row['raw_q']:.6f}"),
        html.P(
            f"Q_tanh: tanh({action_row['raw_q']:.4f}/{q_scale:.1f}) = {action_row['q_transformed']:.6f}"
            if q_scale is not None
            else f"Q_norm: {action_row['q_transformed']:.6f} (sibling q_min={q_min:.6f}, q_max={q_max:.6f})"
        ),
        html.P(
            f"Exploration (U): {action_row['u']:.3f}",
            style={"color": "#0066cc"},
        ),
        html.P(
            (
                f"PUCT = Q_{'tanh' if q_scale is not None else 'norm'} + U = "
                f"{action_row['q_transformed']:.6f} + {action_row['u']:.6f} = {action_row['puct']:.6f}"
            ),
            style={"fontWeight": "bold"},
        ),
    ]

    # Render the resulting board if we were able to compute it
    if attack is not None and env_copy is not None:
        board = env_copy.get_board()
        board_piece_types = env_copy.get_board_piece_types()
        details.append(html.Hr())
        details.append(
            html.P(
                "Board after executing this action:",
                style={"fontSize": "11px", "color": "#666", "fontStyle": "italic"},
            )
        )
    else:
        # Fall back to parent's board
        board = parent["board"]
        board_piece_types = parent["board_piece_types"]
        details.append(html.Hr())
        details.append(
            html.P(
                "(Could not compute resulting board - showing parent state)",
                style={"fontSize": "11px", "color": "#666", "fontStyle": "italic"},
            )
        )

    img_b64 = _render_board_image(board, board_piece_types)

    # State info - from resulting state if computed, otherwise from parent
    if attack is not None and env_copy is not None:
        current = env_copy.get_current_piece()
        hold = env_copy.get_hold_piece()
        queue = env_copy.get_queue(5)
        state_info = [
            html.P(
                f"Current Piece: {PIECE_NAMES[current.piece_type] if current else 'None'}"
            ),
            html.P(f"Hold Piece: {PIECE_NAMES[hold.piece_type] if hold else 'None'}"),
            html.P(f"Queue: {[PIECE_NAMES[p] for p in queue]}"),
            html.P(
                "(State after action)",
                style={"fontSize": "10px", "color": "#888"},
            ),
        ]
    else:
        state_info = [
            html.P(
                f"Current Piece: {PIECE_NAMES[parent['current_piece']] if parent['current_piece'] is not None else 'None'}"
            ),
            html.P(
                f"Hold Piece: {PIECE_NAMES[parent['hold_piece']] if parent['hold_piece'] is not None else 'None'}"
            ),
            html.P(f"Queue: {[PIECE_NAMES[p] for p in parent['queue']]}"),
            html.P(
                "(Parent state - could not compute result)",
                style={"fontSize": "10px", "color": "#888"},
            ),
        ]

    return details, f"data:image/png;base64,{img_b64}", state_info


def display_virtual_piece_node(node_data, tree_dict):
    """Display details for a virtual (unvisited) decision node from a chance node."""
    # Parse virtual node ID: vp_parentid_piecetype
    parts = node_data["id"].split("_")
    if len(parts) != 3:
        return "Invalid virtual piece node", "", ""

    parent_id = int(parts[1])
    piece_type = int(parts[2])

    if parent_id >= len(tree_dict["nodes"]):
        return "Parent not found", "", ""

    parent = tree_dict["nodes"][parent_id]  # This is the chance node
    piece_name = format_chance_outcome(piece_type)

    # Format details
    details = [
        html.H4(
            f"Unvisited Piece: {piece_name}",
            style={"marginTop": 0, "marginBottom": "10px", "color": "#4488ff"},
        ),
        html.P(f"Piece Type: {piece_name} ({piece_type})"),
        html.P(f"Parent Chance Node: C{parent_id}"),
        html.P(f"Cumulative Attack: {parent.get('cumulative_attack', '?')}"),
        html.Hr(),
        html.P(
            (
                "This outcome has not been explored yet. "
                "The board shown is the state after the parent's action was executed."
            ),
            style={"fontSize": "11px", "color": "#666", "fontStyle": "italic"},
        ),
    ]

    # Render the chance node's board (state after action, before piece spawn)
    board = parent["board"]
    board_piece_types = parent["board_piece_types"]

    img_b64 = _render_board_image(board, board_piece_types)

    state_info = [
        html.P(
            f"Current Piece: {PIECE_NAMES[parent['current_piece']] if parent.get('current_piece') is not None else 'None'}"
        ),
        html.P(
            f"Hold Piece: {PIECE_NAMES[parent['hold_piece']] if parent.get('hold_piece') is not None else 'None'}"
        ),
        html.P(f"Queue: {[PIECE_NAMES[p] for p in parent.get('queue', [])]}"),
        html.P(
            f"+ New piece: {piece_name}",
            style={"color": "#4488ff", "fontWeight": "bold"},
        ),
    ]

    return details, f"data:image/png;base64,{img_b64}", state_info


def _node_color(attack: int | str, node_type: str) -> str:
    """Background color based on cumulative attack: cool base -> fire gradient."""
    try:
        atk = int(attack)
    except (TypeError, ValueError):
        atk = 0
    if atk <= 0:
        return "#203560" if node_type == "decision" else "#1a4040"
    # Fire gradient capped at 8 attack: dark brick red -> pure red
    t = min(atk / 8.0, 1.0)
    r = int(139 + 116 * t)
    g = int(24 * (1 - t))
    return f"#{r:02x}{g:02x}00"


def build_cytoscape_elements(
    tree_dict: dict,
    max_nodes: int | None = None,
    show_unvisited: bool = True,
):
    """Convert a serialized tree dict into Cytoscape elements."""
    if tree_dict is None or not tree_dict.get("nodes"):
        return []

    elements = []
    nodes = tree_dict["nodes"]
    c_puct = tree_dict.get("c_puct", 1.0)
    highlighted_node_ids = set(tree_dict.get("highlighted_node_ids", []))
    highlighted_edge_keys = {
        (edge["source"], edge["target"])
        for edge in tree_dict.get("highlighted_edge_keys", [])
    }

    # Limit nodes only if explicitly requested
    nodes_to_show = (
        len(nodes)
        if max_nodes is None or max_nodes <= 0
        else min(len(nodes), max_nodes)
    )

    # Sort nodes by visit count to show most important ones
    indexed_nodes = [(node["id"], node["visit_count"]) for node in nodes]
    sorted_indices = [
        i for i, _ in sorted(indexed_nodes, key=lambda x: x[1], reverse=True)
    ][:nodes_to_show]
    shown_ids = set(sorted_indices)

    # Always include root
    shown_ids.add(tree_dict["root_id"])
    for highlighted_node_id in highlighted_node_ids:
        if highlighted_node_id.isdigit():
            shown_ids.add(int(highlighted_node_id))

    # Track which actions/pieces from nodes already have children
    visited_actions = {}  # decision node_id -> set of action indices with children
    visited_pieces = {}  # chance node_id -> set of piece types with children
    for node in nodes:
        if node["node_type"] == "decision":
            visited_actions[node["id"]] = set()
            for child_id in node["children"]:
                child = nodes[child_id]
                if child["edge_from_parent"] is not None:
                    visited_actions[node["id"]].add(child["edge_from_parent"])
        elif node["node_type"] == "chance":
            visited_pieces[node["id"]] = set()
            for child_id in node["children"]:
                child = nodes[child_id]
                if child["edge_from_parent"] is not None:
                    visited_pieces[node["id"]].add(child["edge_from_parent"])

    for node in nodes:
        if node["id"] not in shown_ids:
            continue

        # Node data
        is_decision = node["node_type"] == "decision"
        node_classes = ["decision" if is_decision else "chance"]
        if node.get("is_reuse_root"):
            node_classes.append("reused-root")
        if str(node["id"]) in highlighted_node_ids:
            node_classes.append("chosen-node")

        # Label based on type
        if is_decision:
            label = (
                f"D{node['id']}\nV:{node['visit_count']}\nQ:{node['mean_value']:.1f}"
            )
        else:
            label = f"C{node['id']}\nV:{node['visit_count']}\nA:{node['attack']}"

        elements.append(
            {
                "data": {
                    "id": str(node["id"]),
                    "label": label,
                    "node_type": node["node_type"],
                    "visit_count": node["visit_count"],
                    "mean_value": node["mean_value"],
                    "value_sum": node["value_sum"],
                            "attack": node["attack"],
                            "cumulative_attack": node.get("cumulative_attack", 0),
                            "attack_color": _node_color(node["attack"], node["node_type"]),
                    "is_terminal": node["is_terminal"],
                    "move_number": node["move_number"],
                    "edge_from_parent": node["edge_from_parent"],
                    "parent_id": node["parent_id"],
                    "search_step": node.get("search_step", 0),
                },
                "classes": " ".join(node_classes),
            }
        )

        # Edges to children
        for child_id in node["children"]:
            if child_id in shown_ids:
                child = nodes[child_id]
                edge_label = ""
                if child["edge_from_parent"] is not None:
                    if is_decision:
                        edge_label = f"a{child['edge_from_parent']}"
                    else:
                        edge_label = format_chance_outcome(child["edge_from_parent"])

                source_id = str(node["id"])
                target_id = str(child_id)
                edge_classes = []
                if (source_id, target_id) in highlighted_edge_keys:
                    edge_classes.append("chosen-edge")

                elements.append(
                    {
                        "data": {
                            "source": source_id,
                            "target": target_id,
                            "label": edge_label,
                        },
                        "classes": " ".join(edge_classes),
                    }
                )

        # Add virtual (unvisited) chance nodes for decision nodes
        if is_decision and node["valid_actions"]:
            sqrt_parent = (node["visit_count"] + 1) ** 0.5
            action_to_prior = dict(zip(node["valid_actions"], node["action_priors"]))

            for action_idx in node["valid_actions"]:
                # Skip if this action already has a child
                if action_idx in visited_actions.get(node["id"], set()):
                    continue

                prior = action_to_prior.get(action_idx, 0.0)
                # U = c_puct * P * sqrt(N_parent) / (1 + 0) for unvisited
                u_value = c_puct * prior * sqrt_parent
                virtual_id = f"v_{node['id']}_{action_idx}"
                edge_key = (str(node["id"]), virtual_id)
                is_highlighted = (
                    virtual_id in highlighted_node_ids
                    or edge_key in highlighted_edge_keys
                )
                if not show_unvisited and not is_highlighted:
                    continue

                virtual_classes = ["chance", "unvisited"]
                edge_classes = ["unvisited-edge"]
                if virtual_id in highlighted_node_ids:
                    virtual_classes.append("chosen-node")
                if edge_key in highlighted_edge_keys:
                    edge_classes.append("chosen-edge")

                elements.append(
                    {
                        "data": {
                            "id": virtual_id,
                            "label": f"a{action_idx}\nP:{prior:.2f}\nU:{u_value:.2f}",
                            "node_type": "virtual",
                            "visit_count": 0,
                            "mean_value": 0.0,
                            "value_sum": 0.0,
                            "attack": "?",
                            "cumulative_attack": node.get("cumulative_attack", 0),
                            "attack_color": _node_color(0, "chance"),
                            "is_terminal": False,
                            "move_number": node["move_number"],
                            "edge_from_parent": action_idx,
                            "parent_id": node["id"],
                            "prior": prior,
                            "u_value": u_value,
                            "search_step": node.get("search_step", 0),
                        },
                        "classes": " ".join(virtual_classes),
                    }
                )

                elements.append(
                    {
                        "data": {
                            "source": str(node["id"]),
                            "target": virtual_id,
                            "label": f"a{action_idx}",
                        },
                        "classes": " ".join(edge_classes),
                    }
                )

        # Add virtual (unvisited) decision nodes for chance nodes
        if not is_decision:
            possible_outcomes = node.get("possible_chance_outcomes", [])
            for outcome_idx in possible_outcomes:
                # Skip if this outcome already has a child
                if outcome_idx in visited_pieces.get(node["id"], set()):
                    continue

                virtual_id = f"vp_{node['id']}_{outcome_idx}"
                outcome_name = format_chance_outcome(outcome_idx)
                edge_key = (str(node["id"]), virtual_id)
                is_highlighted = (
                    virtual_id in highlighted_node_ids
                    or edge_key in highlighted_edge_keys
                )
                if not show_unvisited and not is_highlighted:
                    continue

                virtual_classes = ["decision", "unvisited"]
                edge_classes = ["unvisited-edge"]
                if virtual_id in highlighted_node_ids:
                    virtual_classes.append("chosen-node")
                if edge_key in highlighted_edge_keys:
                    edge_classes.append("chosen-edge")

                elements.append(
                    {
                        "data": {
                            "id": virtual_id,
                            "label": f"{outcome_name}\n(unvisited)",
                            "node_type": "virtual_decision",
                            "visit_count": 0,
                            "mean_value": 0.0,
                            "value_sum": 0.0,
                            "attack": 0,
                            "cumulative_attack": node.get("cumulative_attack", 0),
                            "attack_color": _node_color(0, "decision"),
                            "is_terminal": False,
                            "move_number": node["move_number"],
                            "edge_from_parent": outcome_idx,
                            "parent_id": node["id"],
                            "search_step": node.get("search_step", 0),
                        },
                        "classes": " ".join(virtual_classes),
                    }
                )

                elements.append(
                    {
                        "data": {
                            "source": str(node["id"]),
                            "target": virtual_id,
                            "label": outcome_name,
                        },
                        "classes": " ".join(edge_classes),
                    }
                )

    for reuse_edge in tree_dict.get("reuse_edges", []):
        elements.append(
            {
                "data": {
                    "source": reuse_edge["source"],
                    "target": reuse_edge["target"],
                    "label": reuse_edge.get("label", "reuse"),
                },
                "classes": reuse_edge.get("classes", "reuse-edge"),
            }
        )

    return elements


def _find_tree_node(tree_dict: dict, node_id: int) -> dict | None:
    if node_id < 0:
        return None
    nodes = tree_dict.get("nodes", [])
    if node_id >= len(nodes):
        return None
    node = nodes[node_id]
    if int(node["id"]) != node_id:
        return None
    return node


def _virtual_action_node_data(tree_dict: dict, node_id_str: str) -> dict | None:
    parts = node_id_str.split("_")
    if len(parts) != 3:
        return None
    parent_id = int(parts[1])
    action_idx = int(parts[2])
    parent = _find_tree_node(tree_dict, parent_id)
    if parent is None or parent.get("node_type") != "decision":
        return None

    c_puct = tree_dict.get("c_puct", 1.0)
    q_scale = tree_dict.get("q_scale")
    action_stats = build_decision_action_stats(parent, tree_dict, c_puct, q_scale)
    stats_by_action = {row["action"]: row for row in action_stats}
    action_row = stats_by_action.get(action_idx)
    if action_row is None:
        return None

    return {
        "id": node_id_str,
        "label": f"a{action_idx}\nP:{action_row['prior']:.2f}\nU:{action_row['u']:.2f}",
        "node_type": "virtual",
        "visit_count": 0,
        "mean_value": 0.0,
        "value_sum": 0.0,
        "attack": "?",
        "attack_color": _node_color(0, "chance"),
        "cumulative_attack": parent.get("cumulative_attack", 0),
        "is_terminal": False,
        "move_number": parent["move_number"],
        "edge_from_parent": action_idx,
        "parent_id": parent_id,
        "prior": action_row["prior"],
        "u_value": action_row["u"],
        "search_step": parent.get("search_step", 0),
    }


def _virtual_chance_node_data(tree_dict: dict, node_id_str: str) -> dict | None:
    parts = node_id_str.split("_")
    if len(parts) != 3:
        return None
    parent_id = int(parts[1])
    outcome_idx = int(parts[2])
    parent = _find_tree_node(tree_dict, parent_id)
    if parent is None or parent.get("node_type") != "chance":
        return None

    return {
        "id": node_id_str,
        "label": f"{format_chance_outcome(outcome_idx)}\n(unvisited)",
        "node_type": "virtual_decision",
        "visit_count": 0,
        "mean_value": 0.0,
        "value_sum": 0.0,
        "attack": 0,
        "attack_color": _node_color(0, "decision"),
        "cumulative_attack": parent.get("cumulative_attack", 0),
        "is_terminal": False,
        "move_number": parent["move_number"],
        "edge_from_parent": outcome_idx,
        "parent_id": parent_id,
        "search_step": parent.get("search_step", 0),
    }


def resolve_node_data_from_tree(tree_dict: dict, node_id_str: str) -> dict | None:
    if node_id_str.startswith("v_") and not node_id_str.startswith("vp_"):
        return _virtual_action_node_data(tree_dict, node_id_str)
    if node_id_str.startswith("vp_"):
        return _virtual_chance_node_data(tree_dict, node_id_str)
    if not node_id_str.isdigit():
        return None
    node = _find_tree_node(tree_dict, int(node_id_str))
    if node is None:
        return None
    return {
        "id": str(node["id"]),
        "edge_from_parent": node.get("edge_from_parent"),
        "parent_id": node.get("parent_id"),
    }


def siblings_for_parent(tree_dict: dict, parent_id: int) -> list[str]:
    parent = _find_tree_node(tree_dict, parent_id)
    if parent is None:
        return []

    siblings: list[str] = []
    for child_id in parent.get("children", []):
        child = _find_tree_node(tree_dict, child_id)
        if child is not None:
            siblings.append(str(child["id"]))

    if parent["node_type"] == "decision":
        visited_actions = {
            _find_tree_node(tree_dict, child_id).get("edge_from_parent")
            for child_id in parent.get("children", [])
            if _find_tree_node(tree_dict, child_id) is not None
        }
        for action_idx in parent.get("valid_actions", []):
            if action_idx not in visited_actions:
                siblings.append(f"v_{parent_id}_{action_idx}")
    else:
        visited_outcomes = {
            _find_tree_node(tree_dict, child_id).get("edge_from_parent")
            for child_id in parent.get("children", [])
            if _find_tree_node(tree_dict, child_id) is not None
        }
        for outcome_idx in parent.get("possible_chance_outcomes", []):
            if outcome_idx not in visited_outcomes:
                siblings.append(f"vp_{parent_id}_{outcome_idx}")

    def get_edge(sid: str) -> int:
        if sid.startswith("v_") or sid.startswith("vp_"):
            return int(sid.split("_")[2])
        if sid.isdigit():
            node = _find_tree_node(tree_dict, int(sid))
            if node is not None and node.get("edge_from_parent") is not None:
                return int(node["edge_from_parent"])
        return 0

    siblings.sort(key=get_edge)
    return siblings


def resolve_parent_target(tree_dict: dict, node_id_str: str) -> str | None:
    node_data = resolve_node_data_from_tree(tree_dict, node_id_str)
    if node_data is not None and node_data.get("parent_id") is not None:
        return str(node_data["parent_id"])

    for reuse_edge in tree_dict.get("reuse_edges", []):
        if reuse_edge.get("target") == node_id_str:
            return str(reuse_edge["source"])
    return None


def resolve_best_child_target(tree_dict: dict, node_id_str: str) -> str | None:
    if node_id_str.startswith("v_") or node_id_str.startswith("vp_"):
        return None
    if not node_id_str.isdigit():
        return None

    node = _find_tree_node(tree_dict, int(node_id_str))
    if node is None:
        return None

    children = node.get("children", [])
    if children:
        best = max(
            children,
            key=lambda cid: (
                tree_dict["nodes"][cid].get("visit_count", 0)
                if 0 <= cid < len(tree_dict["nodes"])
                else 0
            ),
        )
        return str(best)

    if node.get("node_type") == "decision" and node.get("valid_actions"):
        action_stats = build_decision_action_stats(
            node,
            tree_dict,
            tree_dict.get("c_puct", 1.0),
            tree_dict.get("q_scale"),
        )
        if action_stats:
            best_action = max(action_stats, key=lambda row: (row["visits"], row["puct"]))
            return str(best_action["target_node_id"])

    if node.get("node_type") == "chance" and node.get("possible_chance_outcomes"):
        outcome_stats = build_chance_outcome_stats(node, tree_dict)
        if outcome_stats:
            best_outcome = max(
                outcome_stats, key=lambda row: (row["visits"], not row["is_virtual"])
            )
            return str(best_outcome["target_node_id"])

    for reuse_edge in tree_dict.get("reuse_edges", []):
        if reuse_edge.get("source") == node_id_str:
            return str(reuse_edge["target"])
    return None


if SAVED_PLAYBACK_DEFAULTS is not None:
    _env_cache.clear()
    _env_cache.update(SAVED_PLAYBACK_DEFAULTS["env_cache"])
    INITIAL_TREE_DATA = SAVED_PLAYBACK_DEFAULTS["tree_dict"]
    INITIAL_ENV_DATA: dict[str, int | str | None] | None = {
        "seed": SAVED_PLAYBACK_DEFAULTS["seed"],
        "move_number": 0,
        "mode": "full_game",
        "saved_playback": str(SAVED_PLAYBACK_DEFAULTS["path"]),
    }
    INITIAL_SELECTED_NODE = (
        str(INITIAL_TREE_DATA["root_id"]) if INITIAL_TREE_DATA.get("nodes") else None
    )
    INITIAL_ELEMENTS = build_cytoscape_elements(INITIAL_TREE_DATA, None, False)
    INITIAL_COUNTER_LABEL = INITIAL_TREE_DATA["counter_label"]
    INITIAL_ADD_NOISE_VALUE = (
        ["noise"]
        if SAVED_PLAYBACK_DEFAULTS["payload"]["metadata"].get("add_noise")
        else []
    )
else:
    INITIAL_TREE_DATA = None
    INITIAL_ENV_DATA = None
    INITIAL_SELECTED_NODE = None
    INITIAL_ELEMENTS = []
    INITIAL_COUNTER_LABEL = "Sims: 0"
    INITIAL_ADD_NOISE_VALUE = ["noise"]


# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body { margin: 0; padding: 0; overflow: hidden; height: 100%; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Cytoscape stylesheet
stylesheet = [
    # Decision nodes — navy base, attack heatmap via data(attack_color)
    {
        "selector": ".decision",
        "style": {
            "background-color": "data(attack_color)",
            "label": "data(label)",
            "text-wrap": "wrap",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            "color": "white",
            "width": 80,
            "height": 60,
            "shape": "round-rectangle",
        },
    },
    # Chance nodes — teal base, attack heatmap via data(attack_color)
    {
        "selector": ".chance",
        "style": {
            "background-color": "data(attack_color)",
            "label": "data(label)",
            "text-wrap": "wrap",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            "color": "white",
            "width": 70,
            "height": 50,
            "shape": "diamond",
        },
    },
    # Edges
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "line-color": "#666",
            "target-arrow-color": "#666",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "label": "data(label)",
            "font-size": "8px",
            "text-rotation": "autorotate",
        },
    },
    # Selected node
    {
        "selector": ":selected",
        "style": {
            "border-width": 3,
            "border-color": "#ff0",
        },
    },
    # Unvisited (virtual) nodes - semi-transparent, color from data(attack_color)
    {
        "selector": ".unvisited",
        "style": {
            "opacity": 0.4,
        },
    },
    # Edges to unvisited nodes - dashed and semi-transparent
    {
        "selector": ".unvisited-edge",
        "style": {
            "opacity": 0.4,
            "line-style": "dashed",
        },
    },
    {
        "selector": ".chosen-node",
        "style": {
            "border-width": 4,
            "border-color": "#f4d35e",
        },
    },
    {
        "selector": ".chosen-edge",
        "style": {
            "width": 4,
            "line-color": "#f4d35e",
            "target-arrow-color": "#f4d35e",
        },
    },
    {
        "selector": ".reuse-edge",
        "style": {
            "line-style": "dotted",
            "curve-style": "unbundled-bezier",
        },
    },
    {
        "selector": ".reused-root",
        "style": {
            "border-width": 3,
            "border-color": "#ff8844",
        },
    },
    # Highlighted node (keyboard navigation) - just adds border, keeps other styles
    {
        "selector": ".highlighted",
        "style": {
            "border-width": 3,
            "border-color": "#ffff00",
        },
    },
]

app.layout = html.Div(
    [
        # Compact controls bar
        html.Div(
            [
                html.Label("Model:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="model-path",
                    type="text",
                    placeholder="Path to ONNX model",
                    value=VIZ_DEFAULTS["model_path"],
                    style={"width": "250px", "marginRight": "15px"},
                ),
                dcc.Checklist(
                    id="use-dummy-network",
                    options=[{"label": " Dummy network", "value": "dummy"}],
                    value=["dummy"] if VIZ_DEFAULTS["use_dummy_network"] else [],
                    style={"marginRight": "15px"},
                ),
                html.Label("Sims:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="num-simulations",
                    type="number",
                    value=VIZ_DEFAULTS["num_simulations"],
                    min=1,
                    style={"width": "60px", "marginRight": "5px"},
                ),
                html.Label("c_puct:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="c-puct",
                    type="number",
                    value=VIZ_DEFAULTS["c_puct"],
                    step=0.1,
                    style={"width": "70px", "marginRight": "15px"},
                ),
                html.Label("q_scale:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="q-scale",
                    type="number",
                    value=VIZ_DEFAULTS["q_scale"],
                    step=1.0,
                    style={"width": "70px", "marginRight": "15px"},
                ),
                html.Label("Seed:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="seed",
                    type="number",
                    value=STATE_PRESET_DEFAULTS["seed"],
                    style={"width": "60px", "marginRight": "15px"},
                ),
                html.Label("Placement #:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="move-number",
                    type="number",
                    value=STATE_PRESET_DEFAULTS["move_number"],
                    min=0,
                    step=1,
                    style={"width": "70px", "marginRight": "15px"},
                ),
                dcc.Checklist(
                    id="add-noise",
                    options=[{"label": " Root noise", "value": "noise"}],
                    value=INITIAL_ADD_NOISE_VALUE,
                    style={"marginRight": "15px"},
                ),
                html.Label("Temp:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="temperature",
                    type="number",
                    value=VIZ_DEFAULTS["temperature"],
                    step=0.1,
                    style={"width": "70px", "marginRight": "15px"},
                ),
                html.Label("Dir α:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="dirichlet-alpha",
                    type="number",
                    value=VIZ_DEFAULTS["dirichlet_alpha"],
                    step=0.01,
                    style={"width": "70px", "marginRight": "15px"},
                ),
                html.Label("Dir ε:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="dirichlet-epsilon",
                    type="number",
                    value=VIZ_DEFAULTS["dirichlet_epsilon"],
                    step=0.01,
                    style={"width": "70px", "marginRight": "15px"},
                ),
                html.Label("Max Placements:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="max-placements",
                    type="number",
                    value=VIZ_DEFAULTS["max_placements"],
                    min=1,
                    step=1,
                    style={"width": "80px", "marginRight": "15px"},
                ),
                html.Label("Current:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="current-piece",
                    type="text",
                    placeholder="I/O/T/S/Z/J/L or 0-6",
                    value=STATE_PRESET_DEFAULTS["current_piece"],
                    style={"width": "150px", "marginRight": "10px"},
                ),
                html.Label("Hold:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="hold-piece",
                    type="text",
                    placeholder="optional piece",
                    value=STATE_PRESET_DEFAULTS["hold_piece"],
                    style={"width": "110px", "marginRight": "10px"},
                ),
                html.Label("Hold Used:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="hold-used",
                    type="text",
                    placeholder="true/false",
                    value=STATE_PRESET_DEFAULTS["hold_used"],
                    style={"width": "90px", "marginRight": "10px"},
                ),
                html.Label("Queue:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="queue-input",
                    type="text",
                    placeholder="comma-separated, e.g. I,T,L,S,O",
                    value=STATE_PRESET_DEFAULTS["queue"],
                    style={"width": "230px", "marginRight": "10px"},
                ),
                html.Button(
                    "Step (+1)",
                    id="step-button",
                    n_clicks=0,
                    style={"marginLeft": "10px"},
                ),
                html.Button(
                    "Step (+100)",
                    id="step-100-button",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                html.Button(
                    "Step (+1000)",
                    id="step-1000-button",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                html.Button(
                    "Step (-1)",
                    id="step-back-button",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                html.Button(
                    "Play Full Game",
                    id="play-full-game-button",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                html.Div(
                    dcc.Loading(
                        html.Span(id="sim-counter", children=INITIAL_COUNTER_LABEL),
                        type="dot",
                    ),
                    style={"marginLeft": "15px", "fontWeight": "bold"},
                ),
                dcc.Checklist(
                    id="show-unvisited",
                    options=[{"label": " Show unvisited", "value": "show"}],
                    value=[],  # Default to unchecked
                    style={"marginLeft": "20px"},
                ),
                dcc.Textarea(
                    id="board-input",
                    value=(
                        STATE_PRESET_DEFAULTS["board"]
                        if isinstance(STATE_PRESET_DEFAULTS["board"], str)
                        else ""
                    ),
                    placeholder=(
                        "Optional board override: 20 rows x 10 cols.\n"
                        "Use '.' or '0' for empty, I/O/T/S/Z/J/L or 1-7 for filled."
                    ),
                    style={
                        "width": "420px",
                        "height": "130px",
                        "marginLeft": "10px",
                        "fontFamily": "monospace",
                        "fontSize": "11px",
                    },
                ),
            ],
            style={
                "padding": "5px 10px",
                "backgroundColor": "#f0f0f0",
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "5px",
            },
        ),
        html.Div(
            dcc.Loading(
                html.Div(
                    [
                        # Tree visualization (left)
                        html.Div(
                            [
                                cyto.Cytoscape(
                                    id="cytoscape-tree",
                                    elements=INITIAL_ELEMENTS,
                                    style={
                                        "width": "100%",
                                        "height": "100%",
                                    },
                                    layout={
                                        "name": "dagre",
                                        "rankDir": "TB",
                                        "spacingFactor": 1.5,
                                    },
                                    stylesheet=stylesheet,
                                    zoom=1,
                                    pan={"x": 0, "y": 0},
                                ),
                            ],
                            style={"flex": "1", "minWidth": "0"},
                        ),
                        # Right panel: Board + State info (top) and Node details (bottom)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Button(
                                            "Back",
                                            id="nav-back-button",
                                            n_clicks=0,
                                            style={
                                                "padding": "4px 12px",
                                                "cursor": "pointer",
                                            },
                                        ),
                                        html.Button(
                                            "Root",
                                            id="nav-root-button",
                                            n_clicks=0,
                                            style={
                                                "padding": "4px 12px",
                                                "cursor": "pointer",
                                                "marginLeft": "6px",
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "6px"},
                                ),
                                # Top row: Board image + State info side by side
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="board-image",
                                                    style={"border": "1px solid #ccc"},
                                                ),
                                            ],
                                            style={"marginRight": "15px"},
                                        ),
                                        html.Div(
                                            id="state-info",
                                            style={"fontSize": "13px"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "flexDirection": "row",
                                        "alignItems": "flex-start",
                                        "marginBottom": "10px",
                                        "paddingBottom": "10px",
                                        "borderBottom": "1px solid #ddd",
                                    },
                                ),
                                # Node details below
                                html.Div(
                                    id="node-details",
                                    children="Click a node to see details",
                                    style={
                                        "overflowY": "auto",
                                        "flex": "1",
                                        "minHeight": "0",
                                    },
                                ),
                            ],
                            style={
                                "width": "480px",
                                "padding": "10px",
                                "backgroundColor": "#f8f8f8",
                                "marginLeft": "5px",
                                "height": "100%",
                                "display": "flex",
                                "flexDirection": "column",
                                "overflow": "hidden",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "flex": "1",
                        "minHeight": "0",
                    },
                ),
                type="circle",
                delay_show=250,
            ),
            style={"flex": "1", "minHeight": "0"},
        ),
        # Hidden storage for tree data
        dcc.Store(id="tree-store", data=INITIAL_TREE_DATA),
        dcc.Store(id="env-store", data=INITIAL_ENV_DATA),
        dcc.Store(id="sims-done-store", data=0),
        # Store for current selection and navigation
        dcc.Store(id="selected-node-store", data=INITIAL_SELECTED_NODE),
        dcc.Store(id="nav-history-store", data=[]),
        dcc.Store(id="siblings-store", data=[]),
        # Hidden div to capture keyboard events
        html.Div(id="keyboard-target", tabIndex="0", style={"outline": "none"}),
        dcc.Store(id="keyboard-event", data={"key": "", "timestamp": 0}),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "0",
        "margin": "0",
        "display": "flex",
        "flexDirection": "column",
        "overflowY": "auto",
        "height": "100vh",
    },
)


def build_error_elements(error_label: str) -> list[dict]:
    return [
        {
            "data": {"id": "error", "label": error_label},
            "classes": "decision",
        }
    ]


@callback(
    Output("tree-store", "data"),
    Output("env-store", "data"),
    Output("sims-done-store", "data"),
    Output("selected-node-store", "data", allow_duplicate=True),
    Output("siblings-store", "data", allow_duplicate=True),
    Output("cytoscape-tree", "elements"),
    Output("sim-counter", "children"),
    Input("step-button", "n_clicks"),
    Input("step-100-button", "n_clicks"),
    Input("step-1000-button", "n_clicks"),
    Input("step-back-button", "n_clicks"),
    Input("play-full-game-button", "n_clicks"),
    Input("show-unvisited", "value"),
    State("model-path", "value"),
    State("use-dummy-network", "value"),
    State("num-simulations", "value"),
    State("c-puct", "value"),
    State("q-scale", "value"),
    State("seed", "value"),
    State("move-number", "value"),
    State("add-noise", "value"),
    State("temperature", "value"),
    State("dirichlet-alpha", "value"),
    State("dirichlet-epsilon", "value"),
    State("max-placements", "value"),
    State("current-piece", "value"),
    State("hold-piece", "value"),
    State("hold-used", "value"),
    State("queue-input", "value"),
    State("board-input", "value"),
    State("tree-store", "data"),
    State("env-store", "data"),
    State("sims-done-store", "data"),
    prevent_initial_call=True,
)
def run_mcts(
    step_clicks,
    step_100_clicks,
    step_1000_clicks,
    step_back_clicks,
    _play_full_game_clicks,
    show_unvisited_value,
    model_path,
    use_dummy_network_value,
    num_sims,
    c_puct,
    q_scale_input,
    seed,
    move_number,
    add_noise_value,
    temperature,
    dirichlet_alpha,
    dirichlet_epsilon,
    max_placements,
    current_piece_text,
    hold_piece_text,
    hold_used_text,
    queue_text,
    board_text,
    tree_data,
    env_data,
    sims_done,
):
    """Run MCTS search and update the tree."""
    global _env_cache

    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            [],
            dash.no_update,
        )

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    show_unvisited = "show" in (show_unvisited_value or [])

    if triggered_id == "show-unvisited":
        if tree_data is None:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        return (
            tree_data,
            env_data,
            sims_done,
            dash.no_update,
            dash.no_update,
            build_cytoscape_elements(tree_data, None, show_unvisited),
            tree_data.get("counter_label", f"Sims: {sims_done or 0}"),
        )

    if num_sims is None:
        raise ValueError("num_simulations is required")
    if c_puct is None:
        raise ValueError("c_puct is required")
    if temperature is None:
        raise ValueError("temperature is required")
    if dirichlet_alpha is None:
        raise ValueError("dirichlet_alpha is required")
    if dirichlet_epsilon is None:
        raise ValueError("dirichlet_epsilon is required")
    if max_placements is None:
        raise ValueError("max_placements is required")

    current_sims = sims_done or 0

    if triggered_id == "play-full-game-button":
        sims_to_run = int(num_sims)
    elif triggered_id == "step-button":
        sims_to_run = current_sims + 1
    elif triggered_id == "step-100-button":
        sims_to_run = current_sims + 100
    elif triggered_id == "step-1000-button":
        sims_to_run = current_sims + 1000
    elif triggered_id == "step-back-button":
        sims_to_run = max(current_sims - 1, 0)
    else:
        sims_to_run = current_sims

    if triggered_id != "play-full-game-button" and sims_to_run == 0:
        _env_cache.clear()
        return (
            None,
            {
                "seed": seed,
                "move_number": int(move_number) if move_number is not None else 0,
            },
            0,
            None,
            [],
            [],
            "Sims: 0",
        )

    # Create config with current number of simulations
    q_scale = float(q_scale_input) if q_scale_input is not None else None
    config = MCTSConfig()
    config.num_simulations = sims_to_run
    config.c_puct = c_puct
    config.q_scale = q_scale
    config.temperature = temperature
    config.dirichlet_alpha = dirichlet_alpha
    config.dirichlet_epsilon = dirichlet_epsilon
    config.max_placements = max_placements
    config.seed = int(seed) if seed is not None else None
    config.track_value_history = True
    config.reuse_tree = True
    agent = MCTSAgent(config)

    use_dummy_network = "dummy" in (use_dummy_network_value or [])

    # Load model unless running uniform-prior/zero-value bootstrap mode.
    if not use_dummy_network:
        if model_path is None or str(model_path).strip() == "":
            error_label = "Model path is required unless dummy network is enabled"
            return (
                None,
                None,
                0,
                None,
                [],
                build_error_elements(error_label),
                error_label,
            )

        model_path_str = str(model_path)
        if not Path(model_path_str).exists():
            return (
                None,
                None,
                0,
                None,
                [],
                build_error_elements(f"Model not found: {model_path_str}"),
                "Error: Model not found",
            )

        if not agent.load_model(model_path_str):
            model_suffix = Path(model_path_str).suffix.lower()
            if model_suffix == ".pt":
                error_label = (
                    "Failed to load model: .pt checkpoint provided. "
                    "Use an ONNX file (e.g., training_runs/.../checkpoints/incumbent.onnx)."
                )
            else:
                error_label = "Failed to load model: expected an ONNX file"
            return (
                None,
                None,
                0,
                None,
                [],
                build_error_elements(error_label),
                error_label,
            )

    # Create env from seed, then apply optional custom overrides
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed or 42)
    custom_state_error = apply_custom_state(
        env=env,
        current_piece_text=current_piece_text,
        hold_piece_text=hold_piece_text,
        hold_used_text=hold_used_text,
        queue_text=queue_text,
        board_text=board_text,
    )
    if custom_state_error is not None:
        return (
            None,
            None,
            0,
            None,
            [],
            build_error_elements(f"Invalid custom state: {custom_state_error}"),
            f"Error: {custom_state_error}",
        )

    add_noise = "noise" in (add_noise_value or [])
    placement_count_int = int(move_number) if move_number is not None else 0
    env.placement_count = placement_count_int

    if triggered_id == "play-full-game-button":
        playback = agent.play_game_with_trees(
            env,
            max_placements=max_placements,
            add_noise=add_noise,
        )
        if playback is None:
            _env_cache.clear()
            return (
                None,
                None,
                0,
                None,
                [],
                build_error_elements("Full-game playback failed"),
                "Error: Full-game playback failed",
            )

        _env_cache.clear()
        _env_cache.update(build_full_game_env_cache(playback))
        tree_dict = build_full_game_tree_dict(
            playback,
            config.c_puct,
            config.q_scale,
            config.use_parent_value_for_unvisited_q,
        )
        elements = build_cytoscape_elements(tree_dict, None, show_unvisited)
        root_selection = str(tree_dict["root_id"]) if tree_dict.get("nodes") else None
        return (
            tree_dict,
            {"seed": seed, "move_number": placement_count_int, "mode": "full_game"},
            0,
            root_selection,
            [],
            elements,
            tree_dict["counter_label"],
        )

    result = agent.search_with_tree(
        env,
        add_noise=add_noise,
    )
    if result is None:
        _env_cache.clear()
        return (
            None,
            None,
            0,
            None,
            [],
            build_error_elements("MCTS failed (game over?)"),
            "Error: MCTS failed",
        )

    _, tree = result
    _env_cache.clear()
    _env_cache.update(build_env_cache_for_tree(tree))
    tree_dict = tree_export_to_dict(
        tree,
        config.c_puct,
        config.q_scale,
        config.use_parent_value_for_unvisited_q,
    )
    tree_dict["counter_label"] = f"Sims: {sims_to_run}"
    elements = build_cytoscape_elements(tree_dict, None, show_unvisited)

    return (
        tree_dict,
        {"seed": seed, "move_number": placement_count_int, "mode": "single_search"},
        sims_to_run,
        None,
        [],
        elements,
        tree_dict["counter_label"],
    )


@callback(
    Output("node-details", "children"),
    Output("board-image", "src"),
    Output("state-info", "children"),
    Input("cytoscape-tree", "tapNodeData"),
    Input("selected-node-store", "data"),
    State("tree-store", "data"),
)
def display_node_details(tap_node_data, selected_node_id, tree_dict):
    """Display details for clicked node or keyboard-navigated node."""
    if tree_dict is None:
        return "Click a node to see details", "", ""

    # Determine which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Get node data based on trigger source
    if triggered_id == "selected-node-store" and selected_node_id:
        node_data = resolve_node_data_from_tree(tree_dict, str(selected_node_id))
        if node_data is None:
            return "Node not found", "", ""
    elif tap_node_data is not None:
        node_data = tap_node_data
    else:
        return "Click a node to see details", "", ""

    c_puct = tree_dict.get("c_puct", 1.0)
    q_scale = tree_dict.get("q_scale")
    node_id_str = str(node_data["id"])

    # Handle virtual (unvisited) chance nodes (from decision node actions)
    if node_id_str.startswith("v_") and not node_id_str.startswith("vp_"):
        return display_virtual_node(node_data, tree_dict, c_puct, q_scale)

    # Handle virtual (unvisited) decision nodes (from chance node piece outcomes)
    if node_id_str.startswith("vp_"):
        return display_virtual_piece_node(node_data, tree_dict)

    node_id = int(node_id_str)
    if node_id >= len(tree_dict["nodes"]):
        return "Node not found", "", ""

    node = tree_dict["nodes"][node_id]

    # Format details
    details = [
        html.H4("Node Info", style={"marginTop": 0, "marginBottom": "10px"}),
        html.P(f"Node ID: {node['id']}"),
        html.P(f"Search Step: {node.get('search_step', 0)}"),
        html.P(f"Type: {node['node_type']}"),
        html.P(f"Visit Count (N): {node['visit_count']}"),
        html.P(
            f"NN Value: {node['nn_value']:.3f}",
            style={"fontWeight": "bold", "color": "#0066cc"},
        ),
        html.P(f"Cumulative Attack: {node.get('cumulative_attack', 0)}"),
        html.P(f"MCTS Q-Value: {node['mean_value']:.3f}"),
        html.P(f"Value Sum: {node['value_sum']:.3f}"),
    ]
    details.extend(build_value_history_details(node))

    if node["node_type"] == "decision":
        details.extend(
            [
                html.P(f"Placement: {node['move_number']}"),
                html.P(f"Terminal: {node['is_terminal']}"),
                html.P(f"Valid Actions: {len(node['valid_actions'])}"),
            ]
        )
        action_stats = build_decision_action_stats(node, tree_dict, c_puct, q_scale)
        q_min = min(row["raw_q"] for row in action_stats) if action_stats else 0.0
        q_max = max(row["raw_q"] for row in action_stats) if action_stats else 0.0
        q_col = "Qtanh" if q_scale is not None else "Qnorm"

        if action_stats:
            details.append(html.Hr())
            q_desc = (
                f"PUCT = Q_tanh + U, where Q_tanh = tanh(Q/{q_scale:.1f})"
                if q_scale is not None
                else f"PUCT = Q_norm + U, where Q_norm is min-max normalized "
                f"(q_min={q_min:.4f}, q_max={q_max:.4f})"
            )
            details.append(
                html.P(
                    f"{q_desc}, U = c_puct * P * sqrt(N_parent + 1) / (1 + N_child)",
                    style={"fontSize": "11px", "color": "#666", "marginBottom": "10px"},
                )
            )
            details.append(build_action_stats_table(action_stats, q_col))

        elif node["action_priors"]:
            # No children yet, show top priors
            details.append(html.Hr())
            details.append(html.H4("Top Priors (no children yet)"))
            top_priors = sorted(
                zip(node["valid_actions"], node["action_priors"]),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for action, prior in top_priors:
                details.append(html.P(f"  a{action}: P={prior:.4f}"))
    else:
        # Chance node
        details.append(html.P(f"Attack: {node['attack']}"))
        details.append(html.P(f"Placement: {node['move_number']}"))
        outcome_stats = build_chance_outcome_stats(node, tree_dict)
        if outcome_stats:
            details.append(html.Hr())
            details.append(
                html.P(
                    "Outcome children",
                    style={"fontSize": "11px", "color": "#666", "marginBottom": "10px"},
                )
            )
            details.append(build_chance_outcomes_table(outcome_stats))

        # Show parent info if available
        parent_id = node.get("parent_id")
        if parent_id is not None:
            parent = tree_dict["nodes"][parent_id]
            edge = node.get("edge_from_parent")
            if edge is not None and parent["action_priors"]:
                parent_action_stats = build_decision_action_stats(
                    parent, tree_dict, c_puct, q_scale
                )
                selection_row = next(
                    (row for row in parent_action_stats if row["action"] == edge), None
                )
                if selection_row is not None:
                    q_min = min(row["raw_q"] for row in parent_action_stats)
                    q_max = max(row["raw_q"] for row in parent_action_stats)
                    n_parent = parent["visit_count"]
                    n_child = node["visit_count"]
                    details.append(html.Hr())
                    details.append(html.H4("Selection Info"))
                    details.append(
                        html.P(
                            f"Prior (P): {selection_row['prior']:.4f}",
                        )
                    )
                    details.append(html.P(f"Parent visits (N_parent): {n_parent}"))
                    details.append(html.P(f"Child visits (N_child): {n_child}"))
                    details.append(
                        html.P(
                            f"Q (raw): {selection_row['raw_q']:.6f}",
                        )
                    )
                    details.append(
                        html.P(
                            f"Q_tanh: tanh({selection_row['raw_q']:.4f}/{q_scale:.1f}) = {selection_row['q_transformed']:.6f}"
                            if q_scale is not None
                            else f"Q_norm: {selection_row['q_transformed']:.6f} (sibling q_min={q_min:.6f}, q_max={q_max:.6f})",
                        )
                    )
                    details.append(
                        html.P(
                            f"Exploration (U): {selection_row['u']:.3f}",
                            style={"color": "#0066cc"},
                        )
                    )
                    details.append(
                        html.P(
                            (
                                "U = c_puct * P * sqrt(N_parent + 1) / (1 + N_child) = "
                                f"{c_puct:.3f} * {selection_row['prior']:.6f} * sqrt({n_parent} + 1) "
                                f"/ (1 + {n_child}) = {selection_row['u']:.6f}"
                            ),
                            style={"fontFamily": "monospace", "fontSize": "12px"},
                        )
                    )
                    q_label = "Q_tanh" if q_scale is not None else "Q_norm"
                    details.append(
                        html.P(
                            (
                                f"PUCT = {q_label} + U = "
                                f"{selection_row['q_transformed']:.6f} + {selection_row['u']:.6f} = "
                                f"{selection_row['puct']:.6f}"
                            ),
                            style={"fontFamily": "monospace", "fontSize": "12px"},
                        )
                    )
                    details.append(
                        html.P(
                            f"Argmax score used at parent: {selection_row['puct']:.6f}",
                            style={"fontWeight": "bold"},
                        )
                    )

    # Render board
    board = node["board"]
    board_piece_types = node["board_piece_types"]

    img_b64 = _render_board_image(board, board_piece_types)

    # State info
    state_info = [
        html.P(
            f"Current Piece: {PIECE_NAMES[node['current_piece']] if node['current_piece'] is not None else 'None'}"
        ),
        html.P(
            f"Hold Piece: {PIECE_NAMES[node['hold_piece']] if node['hold_piece'] is not None else 'None'}"
        ),
        html.P(f"Queue: {[PIECE_NAMES[p] for p in node['queue']]}"),
    ]

    return details, f"data:image/png;base64,{img_b64}", state_info


# Clientside callback to capture keyboard events
clientside_callback(
    """
    function(n) {
        // Set up keyboard listener on first call
        if (!window._keyboardListenerSet) {
            window._keyboardListenerSet = true;
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    window._lastKeyEvent = {key: e.key, timestamp: Date.now()};
                    document.getElementById('keyboard-target').click();
                }
            });
        }
        return window._lastKeyEvent || {key: '', timestamp: 0};
    }
    """,
    Output("keyboard-event", "data"),
    Input("keyboard-target", "n_clicks"),
)

# Pan the Cytoscape view to center on the selected node
clientside_callback(
    """
    function(selectedNodeId, rootClicks, treeData) {
        var cyEl = document.getElementById('cytoscape-tree');
        if (!cyEl || !cyEl._cyreg || !cyEl._cyreg.cy) return window.dash_clientside.no_update;
        var cy = cyEl._cyreg.cy;

        var ctx = window.dash_clientside.callback_context;
        var trigger = ctx.triggered.length ? ctx.triggered[0].prop_id : '';

        if (trigger === 'nav-root-button.n_clicks') {
            if (!rootClicks || !treeData) return window.dash_clientside.no_update;
            var rootId = String(treeData.root_id);
            var root = cy.getElementById(rootId);
            if (root.length > 0) {
                cy.stop();
                cy.batch(function() {
                    cy.nodes(':selected').unselect();
                    root.select();
                });
                cy.fit(root, 120);
                cy.center(root);
            }
            return window.dash_clientside.no_update;
        }

        if (!selectedNodeId) return window.dash_clientside.no_update;
        var node = cy.getElementById(selectedNodeId);
        if (node.length > 0) {
            cy.batch(function() {
                cy.nodes(':selected').unselect();
                node.select();
            });
            cy.animate({center: {eles: node}, duration: 200});
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("keyboard-target", "title"),  # dummy output (pan)
    Input("selected-node-store", "data"),
    Input("nav-root-button", "n_clicks"),
    State("tree-store", "data"),
)

# Manage navigation history: push previous node on forward nav, pop on back click
clientside_callback(
    """
    function(selectedNodeId, backClicks, history) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered.length) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        var trigger = ctx.triggered[0].prop_id;
        history = history || [];

        if (trigger === 'nav-back-button.n_clicks') {
            if (history.length === 0) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            var prev = history[history.length - 1];
            var newHistory = history.slice(0, -1);
            // Set a flag so the next selected-node-store trigger knows it was a back nav
            window._navBackInProgress = true;
            return [prev, newHistory];
        }

        // Forward navigation — push previous node if not triggered by back
        if (window._navBackInProgress) {
            window._navBackInProgress = false;
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        if (selectedNodeId && window._lastSelectedNode && window._lastSelectedNode !== selectedNodeId) {
            history = history.concat([window._lastSelectedNode]);
        }
        window._lastSelectedNode = selectedNodeId;
        return [window.dash_clientside.no_update, history];
    }
    """,
    Output("selected-node-store", "data", allow_duplicate=True),
    Output("nav-history-store", "data"),
    Input("selected-node-store", "data"),
    Input("nav-back-button", "n_clicks"),
    State("nav-history-store", "data"),
    prevent_initial_call=True,
)


@callback(
    Output("selected-node-store", "data", allow_duplicate=True),
    Input("nav-root-button", "n_clicks"),
    State("tree-store", "data"),
    prevent_initial_call=True,
)
def navigate_to_root(_, tree_dict):
    if not tree_dict:
        return dash.no_update
    return str(tree_dict["root_id"])


@callback(
    Output("selected-node-store", "data"),
    Input("cytoscape-tree", "tapNodeData"),
)
def update_selection_info(node_data):
    """Track selected node from clicks."""
    if node_data is None:
        return None
    return str(node_data["id"])


@callback(
    Output("siblings-store", "data"),
    Input("selected-node-store", "data"),
    State("tree-store", "data"),
)
def update_siblings(selected_node_id, tree_dict):
    """Track siblings for keyboard navigation after any selection change."""
    if selected_node_id is None or tree_dict is None:
        return []

    parent_target = resolve_parent_target(tree_dict, str(selected_node_id))
    if parent_target is None or not str(parent_target).isdigit():
        return []
    return siblings_for_parent(tree_dict, int(parent_target))


@callback(
    Output("selected-node-store", "data", allow_duplicate=True),
    Input("keyboard-event", "data"),
    State("selected-node-store", "data"),
    State("siblings-store", "data"),
    State("tree-store", "data"),
    prevent_initial_call=True,
)
def navigate_siblings(keyboard_event, selected_node, siblings, tree_dict):
    """Navigate the tree with arrow keys: left/right = siblings, up = parent, down = best child."""
    if not keyboard_event or not keyboard_event.get("key"):
        return dash.no_update

    if selected_node is None:
        return dash.no_update

    key = keyboard_event["key"]
    if key not in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"):
        return dash.no_update

    # ArrowUp: go to parent
    if key == "ArrowUp":
        if not tree_dict:
            return dash.no_update
        parent_target = resolve_parent_target(tree_dict, str(selected_node))
        return parent_target if parent_target is not None else dash.no_update

    # ArrowDown: go to most-visited child
    if key == "ArrowDown":
        if not tree_dict:
            return dash.no_update
        child_target = resolve_best_child_target(tree_dict, str(selected_node))
        return child_target if child_target is not None else dash.no_update

    # Extract parent_id from current selection
    parent_target = (
        resolve_parent_target(tree_dict, str(selected_node)) if tree_dict else None
    )

    all_siblings = (
        siblings_for_parent(tree_dict, int(parent_target))
        if parent_target is not None and str(parent_target).isdigit()
        else []
    )
    if not all_siblings:
        return dash.no_update

    if selected_node not in all_siblings:
        return dash.no_update

    current_idx = all_siblings.index(selected_node)

    if key == "ArrowLeft":
        new_idx = (current_idx - 1) % len(all_siblings)
    else:  # ArrowRight
        new_idx = (current_idx + 1) % len(all_siblings)

    return all_siblings[new_idx]


@callback(
    Output("selected-node-store", "data", allow_duplicate=True),
    Input({"type": "nav-button", "target": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def navigate_to_action_node(_):
    """Navigate to a child node when an action row button is clicked."""
    if not dash.callback_context.triggered:
        return dash.no_update

    trigger = dash.callback_context.triggered[0]
    if int(trigger.get("value") or 0) <= 0:
        return dash.no_update

    trigger_prop = trigger["prop_id"]
    trigger_id = trigger_prop.split(".")[0]
    if not trigger_id:
        return dash.no_update

    try:
        target = json.loads(trigger_id).get("target")
    except json.JSONDecodeError:
        return dash.no_update
    if target is None:
        return dash.no_update
    return target


# Load dagre layout extension
cyto.load_extra_layouts()


def main():
    """Run the MCTS visualizer."""
    print("Starting MCTS Visualizer...")
    if SAVED_PLAYBACK_DEFAULTS is not None:
        print(f"Loaded saved playback: {SAVED_PLAYBACK_DEFAULTS['path']}")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
