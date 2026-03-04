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

import dash
from dash import (
    html,
    dcc,
    callback,
    Output,
    Input,
    State,
    clientside_callback,
    dash_table,
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
    NUM_PIECE_TYPES,
    PARALLEL_ONNX_FILENAME,
    PIECE_COLORS,
    PIECE_NAMES,
    PROJECT_ROOT,
    QUEUE_SIZE,
)

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
    run_dir: Path = (  # Training run dir (default: training_runs/v11)
        PROJECT_ROOT / "training_runs" / "v41"
    )
    use_dummy_network: bool = False  # Use uniform-prior/zero-value bootstrap search
    state_preset: Path | None = (
        None  # Optional Python file exposing VIZ_STATE_PRESET dict
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


def load_viz_defaults(args: ScriptArgs) -> dict[str, str | int | float | bool | None]:
    run_dir = args.run_dir
    config_path = run_dir / CONFIG_FILENAME
    model_path = run_dir / CHECKPOINT_DIRNAME / PARALLEL_ONNX_FILENAME

    if not run_dir.exists():
        raise ValueError(f"Run dir does not exist: {run_dir}")
    if not config_path.exists():
        raise ValueError(f"Missing run config: {config_path}")
    if not args.use_dummy_network and not model_path.exists():
        raise ValueError(f"Missing latest ONNX checkpoint: {model_path}")

    config = json.loads(config_path.read_text())
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
        raise ValueError(
            f"Invalid hold_used token '{value}' (expected true/false)"
        )
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
VIZ_DEFAULTS = load_viz_defaults(SCRIPT_ARGS)
STATE_PRESET_DEFAULTS = load_state_preset_defaults(SCRIPT_ARGS)

Q_NORMALIZATION_EPSILON = 1e-6


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


def get_possible_chance_outcomes(chance_node) -> list[int]:
    queue_len = chance_node.state.get_queue_len()
    if queue_len >= QUEUE_SIZE:
        return [NO_CHANCE_OUTCOME]
    outcomes = list(chance_node.state.get_possible_next_pieces())
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
    for action_idx in decision_node["valid_actions"]:
        child = child_by_action.get(action_idx)
        raw_q_by_action[action_idx] = child["mean_value"] if child is not None else 0.0

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


def build_cytoscape_elements(
    tree, max_nodes: int | None = None, show_unvisited: bool = True, c_puct: float = 1.0
):
    """Convert MCTSTreeExport to Cytoscape elements."""
    elements = []

    # Limit nodes only if explicitly requested
    nodes_to_show = (
        len(tree.nodes)
        if max_nodes is None or max_nodes <= 0
        else min(len(tree.nodes), max_nodes)
    )

    # Sort nodes by visit count to show most important ones
    indexed_nodes = [(i, node.visit_count) for i, node in enumerate(tree.nodes)]
    sorted_indices = [
        i for i, _ in sorted(indexed_nodes, key=lambda x: x[1], reverse=True)
    ][:nodes_to_show]
    shown_ids = set(sorted_indices)

    # Always include root
    shown_ids.add(tree.root_id)

    # Track which actions/pieces from nodes already have children
    visited_actions = {}  # decision node_id -> set of action indices with children
    visited_pieces = {}  # chance node_id -> set of piece types with children
    for node in tree.nodes:
        if node.node_type == "decision":
            visited_actions[node.id] = set()
            for child_id in node.children:
                child = tree.nodes[child_id]
                if child.edge_from_parent is not None:
                    visited_actions[node.id].add(child.edge_from_parent)
        elif node.node_type == "chance":
            visited_pieces[node.id] = set()
            for child_id in node.children:
                child = tree.nodes[child_id]
                if child.edge_from_parent is not None:
                    visited_pieces[node.id].add(child.edge_from_parent)

    for node in tree.nodes:
        if node.id not in shown_ids:
            continue

        # Node data
        is_decision = node.node_type == "decision"
        node_class = "decision" if is_decision else "chance"

        # Label based on type
        if is_decision:
            label = f"D{node.id}\nV:{node.visit_count}\nQ:{node.mean_value:.1f}"
        else:
            label = f"C{node.id}\nV:{node.visit_count}\nA:{node.attack}"

        elements.append(
            {
                "data": {
                    "id": str(node.id),
                    "label": label,
                    "node_type": node.node_type,
                    "visit_count": node.visit_count,
                    "mean_value": node.mean_value,
                    "value_sum": node.value_sum,
                    "attack": node.attack,
                    "is_terminal": node.is_terminal,
                    "move_number": node.move_number,
                    "edge_from_parent": node.edge_from_parent,
                    "parent_id": node.parent_id,
                },
                "classes": node_class,
            }
        )

        # Edges to children
        for child_id in node.children:
            if child_id in shown_ids:
                child = tree.nodes[child_id]
                edge_label = ""
                if child.edge_from_parent is not None:
                    if is_decision:
                        edge_label = f"a{child.edge_from_parent}"
                    else:
                        edge_label = format_chance_outcome(child.edge_from_parent)

                elements.append(
                    {
                        "data": {
                            "source": str(node.id),
                            "target": str(child_id),
                            "label": edge_label,
                        }
                    }
                )

        # Add virtual (unvisited) chance nodes for decision nodes
        if show_unvisited and is_decision and node.valid_actions:
            sqrt_parent = (node.visit_count + 1) ** 0.5
            action_to_prior = dict(zip(node.valid_actions, node.action_priors))

            for action_idx in node.valid_actions:
                # Skip if this action already has a child
                if action_idx in visited_actions.get(node.id, set()):
                    continue

                prior = action_to_prior.get(action_idx, 0.0)
                # U = c_puct * P * sqrt(N_parent) / (1 + 0) for unvisited
                u_value = c_puct * prior * sqrt_parent
                virtual_id = f"v_{node.id}_{action_idx}"

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
                            "is_terminal": False,
                            "move_number": node.move_number,
                            "edge_from_parent": action_idx,
                            "parent_id": node.id,
                            "prior": prior,
                            "u_value": u_value,
                        },
                        "classes": "chance unvisited",
                    }
                )

                elements.append(
                    {
                        "data": {
                            "source": str(node.id),
                            "target": virtual_id,
                            "label": f"a{action_idx}",
                        },
                        "classes": "unvisited-edge",
                    }
                )

        # Add virtual (unvisited) decision nodes for chance nodes
        if show_unvisited and not is_decision:
            possible_outcomes = get_possible_chance_outcomes(node)
            for outcome_idx in possible_outcomes:
                # Skip if this outcome already has a child
                if outcome_idx in visited_pieces.get(node.id, set()):
                    continue

                virtual_id = f"vp_{node.id}_{outcome_idx}"
                outcome_name = format_chance_outcome(outcome_idx)

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
                            "is_terminal": False,
                            "move_number": node.move_number,
                            "edge_from_parent": outcome_idx,
                            "parent_id": node.id,
                        },
                        "classes": "decision unvisited",
                    }
                )

                elements.append(
                    {
                        "data": {
                            "source": str(node.id),
                            "target": virtual_id,
                            "label": outcome_name,
                        },
                        "classes": "unvisited-edge",
                    }
                )

    return elements


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
    # Decision nodes (blue)
    {
        "selector": ".decision",
        "style": {
            "background-color": "#4488ff",
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
    # Chance nodes (orange)
    {
        "selector": ".chance",
        "style": {
            "background-color": "#ff8844",
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
    # Unvisited (virtual) chance nodes - semi-transparent
    {
        "selector": ".unvisited",
        "style": {
            "opacity": 0.4,
            "background-color": "#ffaa44",
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
                    max=1000,
                    style={"width": "60px", "marginRight": "15px"},
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
                html.Label("Move #:", style={"marginRight": "5px"}),
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
                    value=["noise"],
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
                html.Label("Max Nodes:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="max-nodes-slider",
                    type="number",
                    value=0,
                    min=0,
                    step=100,
                    style={"width": "80px", "marginRight": "15px"},
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
                    "Step (-1)",
                    id="step-back-button",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                html.Span(
                    id="sim-counter",
                    children="Sims: 0",
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
                    value=STATE_PRESET_DEFAULTS["board"],
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
            [
                # Tree visualization (left)
                html.Div(
                    [
                        cyto.Cytoscape(
                            id="cytoscape-tree",
                            elements=[],
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
                        html.Button(
                            "Back",
                            id="nav-back-button",
                            n_clicks=0,
                            style={
                                "marginBottom": "6px",
                                "padding": "4px 12px",
                                "cursor": "pointer",
                            },
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
                            style={"overflowY": "auto", "flex": "1", "minHeight": "0"},
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
        # Hidden storage for tree data
        dcc.Store(id="tree-store"),
        dcc.Store(id="env-store"),
        dcc.Store(id="sims-done-store", data=0),
        # Store for current selection and navigation
        dcc.Store(id="selected-node-store", data=None),
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
    Input("step-back-button", "n_clicks"),
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
    State("max-nodes-slider", "value"),
    State("tree-store", "data"),
    State("env-store", "data"),
    State("sims-done-store", "data"),
    prevent_initial_call=True,
)
def run_mcts(
    step_clicks,
    step_100_clicks,
    step_back_clicks,
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
    max_nodes,
    tree_data,
    env_data,
    sims_done,
):
    """Run MCTS search and update the tree."""
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

    max_sims = num_sims
    current_sims = sims_done or 0

    if triggered_id == "step-button":
        sims_to_run = min(current_sims + 1, max_sims)
    elif triggered_id == "step-100-button":
        sims_to_run = min(current_sims + 100, max_sims)
    elif triggered_id == "step-back-button":
        sims_to_run = max(current_sims - 1, 0)
    elif triggered_id == "show-unvisited":
        sims_to_run = current_sims
    else:
        sims_to_run = current_sims

    if sims_to_run == 0:
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
            f"Sims: 0/{max_sims}",
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
                [
                    {
                        "data": {"id": "error", "label": error_label},
                        "classes": "decision",
                    }
                ],
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
                [
                    {
                        "data": {
                            "id": "error",
                            "label": f"Model not found: {model_path_str}",
                        },
                        "classes": "decision",
                    }
                ],
                "Error: Model not found",
            )

        if not agent.load_model(model_path_str):
            model_suffix = Path(model_path_str).suffix.lower()
            if model_suffix == ".pt":
                error_label = (
                    "Failed to load model: .pt checkpoint provided. "
                    "Use an ONNX file (e.g., training_runs/.../checkpoints/parallel.onnx)."
                )
            else:
                error_label = "Failed to load model: expected an ONNX file"
            return (
                None,
                None,
                0,
                None,
                [],
                [
                    {
                        "data": {"id": "error", "label": error_label},
                        "classes": "decision",
                    }
                ],
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
            [
                {
                    "data": {
                        "id": "error",
                        "label": f"Invalid custom state: {custom_state_error}",
                    },
                    "classes": "decision",
                }
            ],
            f"Error: {custom_state_error}",
        )

    # Run MCTS with current number of simulations
    add_noise = "noise" in (add_noise_value or [])
    placement_count_int = int(move_number) if move_number is not None else 0
    result = agent.search_with_tree(
        env,
        add_noise=add_noise,
        placement_count=placement_count_int,
    )
    if result is None:
        return (
            None,
            None,
            0,
            None,
            [],
            [
                {
                    "data": {"id": "error", "label": "MCTS failed (game over?)"},
                    "classes": "decision",
                }
            ],
            "Error: MCTS failed",
        )

    mcts_result, tree = result

    # Build elements
    show_unvisited = "show" in (show_unvisited_value or [])
    max_nodes_limit = None if max_nodes is None or max_nodes <= 0 else int(max_nodes)
    elements = build_cytoscape_elements(
        tree, max_nodes_limit, show_unvisited, config.c_puct
    )

    # Cache TetrisEnv states by replaying actions from the root.
    # Tree nodes from mcts_clone() have empty board_piece_types (performance opt).
    # Replaying from root preserves board_piece_types so child boards render with
    # correct piece colors instead of uniform gray.
    global _env_cache
    _env_cache.clear()

    # Build a node_type lookup from tree.nodes (needed before tree_dict is built)
    _node_types = {n.id: n.node_type for n in tree.nodes}

    for n in tree.nodes:
        if n.parent_id is None:
            # Root: full clone preserves board_piece_types from the original env
            _env_cache[n.id] = n.state.clone_state()
        elif n.parent_id in _env_cache and n.edge_from_parent is not None:
            parent_env = _env_cache[n.parent_id].clone_state()
            if _node_types[n.parent_id] == "decision":
                # Chance node child: parent executed this action
                result = parent_env.execute_action_index(n.edge_from_parent)
                if result is None:
                    _env_cache[n.id] = n.state.clone_state()
                    continue
                parent_env.truncate_queue(QUEUE_SIZE)
            else:
                # Decision node child: chance outcome added this piece to queue
                if n.edge_from_parent < NUM_PIECE_TYPES:
                    parent_env.push_queue_piece(n.edge_from_parent)
            _env_cache[n.id] = parent_env
        else:
            # Fallback: use tree node state directly (no board_piece_types)
            _env_cache[n.id] = n.state.clone_state()

    # Store tree data for click handling (nn_value now comes from Rust)
    tree_dict = {
        "nodes": [
            {
                "id": n.id,
                "node_type": n.node_type,
                "visit_count": n.visit_count,
                "mean_value": n.mean_value,
                "value_sum": n.value_sum,
                "value_history": list(n.value_history),
                "nn_value": n.nn_value,  # Now stored in Rust tree export
                "attack": n.attack,
                "is_terminal": n.is_terminal,
                "move_number": n.move_number,
                "valid_actions": list(n.valid_actions),
                "action_priors": list(n.action_priors),
                "children": list(n.children),
                "parent_id": n.parent_id,
                "edge_from_parent": n.edge_from_parent,
                "board": list(n.state.get_board()),
                "board_piece_types": list(
                    _env_cache[n.id].get_board_piece_types()
                    if n.id in _env_cache
                    else n.state.get_board_piece_types()
                ),
                "current_piece": n.state.get_current_piece().piece_type
                if n.state.get_current_piece()
                else None,
                "hold_piece": n.state.get_hold_piece().piece_type
                if n.state.get_hold_piece()
                else None,
                "queue": list(n.state.get_queue(5)),
            }
            for n in tree.nodes
        ],
        "root_id": tree.root_id,
        "selected_action": tree.selected_action,
        "num_simulations": tree.num_simulations,
        "c_puct": config.c_puct,
        "q_scale": config.q_scale,
    }

    return (
        tree_dict,
        {"seed": seed, "move_number": placement_count_int},
        sims_to_run,
        None,
        [],
        elements,
        f"Sims: {sims_to_run}/{max_sims}",
    )


@callback(
    Output("node-details", "children"),
    Output("board-image", "src"),
    Output("state-info", "children"),
    Input("cytoscape-tree", "tapNodeData"),
    Input("selected-node-store", "data"),
    State("tree-store", "data"),
    State("cytoscape-tree", "elements"),
)
def display_node_details(tap_node_data, selected_node_id, tree_dict, elements):
    """Display details for clicked node or keyboard-navigated node."""
    if tree_dict is None:
        return "Click a node to see details", "", ""

    # Determine which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Get node data based on trigger source
    if triggered_id == "selected-node-store" and selected_node_id:
        # Keyboard navigation - find node data from elements
        node_data = None
        for elem in elements:
            if "data" in elem and str(elem["data"].get("id", "")) == selected_node_id:
                node_data = elem["data"]
                break
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
        html.P(f"Type: {node['node_type']}"),
        html.P(f"Visit Count (N): {node['visit_count']}"),
        html.P(
            f"NN Value: {node['nn_value']:.3f}",
            style={"fontWeight": "bold", "color": "#0066cc"},
        ),
        html.P(f"MCTS Q-Value: {node['mean_value']:.3f}"),
        html.P(f"Value Sum: {node['value_sum']:.3f}"),
    ]
    value_history = node.get("value_history", [])
    if value_history:
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

    if node["node_type"] == "decision":
        details.extend(
            [
                html.P(f"Move Number: {node['move_number']}"),
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

            table_data = []
            nav_map = {}
            for row in action_stats:
                action_label = f"a{row['action']}"
                nav_map[action_label] = row["target_node_id"]
                entry: dict[str, str | int | float] = {
                    "Action": action_label,
                    "N": row["visits"],
                    "P": round(row["prior"], 4),
                    q_col: round(row["q_transformed"], 4),
                    "Qraw": round(row["raw_q"], 4),
                    "U": round(row["u"], 4),
                    "PUCT": round(row["puct"], 4),
                }
                if row["child"] is not None:
                    entry["Atk"] = row["child"]["attack"]
                    entry["NNval"] = round(row["child"]["nn_value"], 4)
                else:
                    entry["Atk"] = ""
                    entry["NNval"] = ""
                table_data.append(entry)

            columns = [
                {"name": "Action", "id": "Action", "type": "text"},
                {"name": "N", "id": "N", "type": "numeric"},
                {"name": "P", "id": "P", "type": "numeric"},
                {"name": q_col, "id": q_col, "type": "numeric"},
                {"name": "Qraw", "id": "Qraw", "type": "numeric"},
                {"name": "U", "id": "U", "type": "numeric"},
                {"name": "PUCT", "id": "PUCT", "type": "numeric"},
                {"name": "Atk", "id": "Atk", "type": "numeric"},
                {"name": "NNval", "id": "NNval", "type": "numeric"},
            ]

            details.append(
                dcc.Store(id="action-nav-map", data=nav_map),
            )
            details.append(
                dash_table.DataTable(
                    id="action-stats-table",
                    columns=columns,  # type: ignore[arg-type]
                    data=table_data,
                    sort_action="native",
                    sort_by=[{"column_id": "PUCT", "direction": "desc"}],
                    style_table={"overflowX": "auto", "fontSize": "12px"},
                    style_cell={
                        "padding": "4px 8px",
                        "textAlign": "right",
                        "fontFamily": "monospace",
                        "minWidth": "50px",
                    },
                    style_cell_conditional=[  # type: ignore[arg-type]
                        {
                            "if": {"column_id": "Action"},
                            "textAlign": "left",
                            "cursor": "pointer",
                            "color": "#0066cc",
                            "textDecoration": "underline",
                        },
                    ],
                    style_header={
                        "fontWeight": "bold",
                        "cursor": "pointer",
                    },
                    style_data_conditional=[  # type: ignore[arg-type]
                        {"if": {"column_id": "U"}, "color": "#0066cc"},
                        {"if": {"column_id": "PUCT"}, "fontWeight": "bold"},
                    ],
                    page_size=50,
                ),
            )

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
        details.append(html.P(f"Move Number: {node['move_number']}"))

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
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    window._lastKeyEvent = {key: e.key, timestamp: Date.now()};
                    // Trigger a dummy update by dispatching a custom event
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
    function(selectedNodeId) {
        if (!selectedNodeId) return window.dash_clientside.no_update;
        var cyEl = document.getElementById('cytoscape-tree');
        if (!cyEl || !cyEl._cyreg || !cyEl._cyreg.cy) return window.dash_clientside.no_update;
        var cy = cyEl._cyreg.cy;
        var node = cy.getElementById(selectedNodeId);
        if (node.length > 0) {
            cy.animate({center: {eles: node}, duration: 200});
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("keyboard-target", "title"),  # dummy output (pan)
    Input("selected-node-store", "data"),
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
    Output("selected-node-store", "data"),
    Output("siblings-store", "data"),
    Input("cytoscape-tree", "tapNodeData"),
    State("tree-store", "data"),
    State("cytoscape-tree", "elements"),
)
def update_selection_info(node_data, tree_dict, elements):
    """Track selected node and its siblings for keyboard navigation."""
    if node_data is None or tree_dict is None:
        return None, []

    node_id = str(node_data["id"])

    # Find parent_id from node_data or tree_dict
    parent_id = node_data.get("parent_id")
    if parent_id is None and node_id.isdigit():
        nid = int(node_id)
        if nid < len(tree_dict["nodes"]):
            parent_id = tree_dict["nodes"][nid].get("parent_id")

    siblings = []

    if parent_id is not None:
        # Find all siblings from elements (includes both visited and virtual nodes)
        for elem in elements:
            if "data" not in elem:
                continue
            elem_id = str(elem["data"].get("id", ""))
            elem_parent = elem["data"].get("parent_id")
            # Check if this is a sibling (same parent, is a node not an edge)
            if elem_parent == parent_id and "source" not in elem["data"]:
                siblings.append(elem_id)

    # Sort siblings by edge_from_parent for consistent ordering
    def get_edge(sid):
        # Check elements for edge info
        for elem in elements:
            if "data" in elem and str(elem["data"].get("id", "")) == sid:
                edge = elem["data"].get("edge_from_parent")
                if edge is not None:
                    return edge
        if sid.startswith("v_"):
            return int(sid.split("_")[2])
        elif sid.startswith("vp_"):
            return int(sid.split("_")[2])
        return 0

    siblings.sort(key=get_edge)

    return node_id, siblings


@callback(
    Output("selected-node-store", "data", allow_duplicate=True),
    Input("keyboard-event", "data"),
    State("selected-node-store", "data"),
    State("siblings-store", "data"),
    State("cytoscape-tree", "elements"),
    State("tree-store", "data"),
    prevent_initial_call=True,
)
def navigate_siblings(keyboard_event, selected_node, siblings, elements, tree_dict):
    """Navigate to sibling node on left/right arrow key."""
    if not keyboard_event or not keyboard_event.get("key"):
        return dash.no_update

    if selected_node is None:
        return dash.no_update

    key = keyboard_event["key"]
    if key not in ("ArrowLeft", "ArrowRight"):
        return dash.no_update

    # Build full siblings list including virtual nodes from elements
    all_siblings = list(siblings) if siblings else []

    # Extract parent_id from current selection
    parent_id = None
    if selected_node.startswith("v_"):
        parent_id = int(selected_node.split("_")[1])
    elif selected_node.startswith("vp_"):
        parent_id = int(selected_node.split("_")[1])
    elif tree_dict and selected_node.isdigit():
        nid = int(selected_node)
        if nid < len(tree_dict["nodes"]):
            parent_id = tree_dict["nodes"][nid].get("parent_id")

    # Add virtual siblings from elements
    if parent_id is not None:
        for elem in elements:
            if "data" not in elem:
                continue
            elem_id = str(elem["data"].get("id", ""))
            elem_parent = elem["data"].get("parent_id")
            if elem_parent == parent_id and elem_id not in all_siblings:
                all_siblings.append(elem_id)

    if not all_siblings:
        return dash.no_update

    # Sort by edge_from_parent
    def get_edge(sid):
        # Check elements for edge info
        for elem in elements:
            if "data" in elem and str(elem["data"].get("id", "")) == sid:
                edge = elem["data"].get("edge_from_parent")
                if edge is not None:
                    return edge
        if sid.startswith("v_"):
            return int(sid.split("_")[2])
        elif sid.startswith("vp_"):
            return int(sid.split("_")[2])
        return 0

    all_siblings.sort(key=get_edge)

    if selected_node not in all_siblings:
        return dash.no_update

    current_idx = all_siblings.index(selected_node)

    if key == "ArrowLeft":
        new_idx = (current_idx - 1) % len(all_siblings)
    else:  # ArrowRight
        new_idx = (current_idx + 1) % len(all_siblings)

    return all_siblings[new_idx]


@callback(
    Output("cytoscape-tree", "stylesheet"),
    Input("selected-node-store", "data"),
    Input("cytoscape-tree", "tapNodeData"),
)
def update_highlight_stylesheet(selected_node_id, tap_node_data):
    """Update stylesheet to highlight the selected node."""
    # Start with base stylesheet
    styles = list(stylesheet)  # Copy the base stylesheet

    # Add highlight rule for selected node
    if selected_node_id:
        styles.append(
            {
                "selector": f'node[id = "{selected_node_id}"]',
                "style": {
                    "border-width": 3,
                    "border-color": "#ffff00",
                },
            }
        )

    return styles


@callback(
    Output("selected-node-store", "data", allow_duplicate=True),
    Input("action-stats-table", "active_cell"),
    State("action-stats-table", "derived_virtual_data"),
    State("action-nav-map", "data"),
    prevent_initial_call=True,
)
def navigate_to_action_node(active_cell, virtual_data, nav_map):
    """Navigate to a child node when an Action cell is clicked in the stats table."""
    if not active_cell or not virtual_data or not nav_map:
        return dash.no_update
    if active_cell.get("column_id") != "Action":
        return dash.no_update
    row_idx = active_cell["row"]
    action_label = virtual_data[row_idx]["Action"]
    target = nav_map.get(action_label)
    if target is None:
        return dash.no_update
    return target


# Load dagre layout extension
cyto.load_extra_layouts()


def main():
    """Run the MCTS visualizer."""
    print("Starting MCTS Visualizer...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
