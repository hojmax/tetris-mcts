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
import json
from dataclasses import dataclass
from pathlib import Path

import dash
from dash import html, dcc, callback, Output, Input, State, clientside_callback
import dash_cytoscape as cyto
from PIL import Image, ImageDraw
from simple_parsing import parse

from tetris_core import TetrisEnv, MCTSAgent, MCTSConfig
from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    LATEST_ONNX_FILENAME,
    NUM_PIECE_TYPES,
    PIECE_COLORS,
    PIECE_NAMES,
)

# Global cache for TetrisEnv states (keyed by node ID)
# This allows us to clone states and execute actions for visualization
_env_cache: dict[int, TetrisEnv] = {}


@dataclass
class ScriptArgs:
    run_dir: Path = Path("training_runs/v3")  # Training run dir (default: training_runs/v3)


def load_viz_defaults(args: ScriptArgs) -> dict[str, str | int | float]:
    defaults: dict[str, str | int | float] = {
        "model_path": "training_runs/v3/checkpoints/latest.onnx",
        "num_simulations": 100,
        "c_puct": 1.0,
    }
    run_dir = args.run_dir
    config_path = run_dir / CONFIG_FILENAME
    model_path = run_dir / CHECKPOINT_DIRNAME / LATEST_ONNX_FILENAME

    if not run_dir.exists():
        raise ValueError(f"Run dir does not exist: {run_dir}")
    if not config_path.exists():
        raise ValueError(f"Missing run config: {config_path}")
    if not model_path.exists():
        raise ValueError(f"Missing latest ONNX checkpoint: {model_path}")

    config = json.loads(config_path.read_text())
    defaults["model_path"] = str(model_path)
    defaults["num_simulations"] = int(config["num_simulations"])
    defaults["c_puct"] = float(config["c_puct"])
    return defaults


SCRIPT_ARGS = parse(ScriptArgs)
VIZ_DEFAULTS = load_viz_defaults(SCRIPT_ARGS)

PIECE_TOKEN_TO_INDEX = {
    "I": 0,
    "O": 1,
    "T": 2,
    "S": 3,
    "Z": 4,
    "J": 5,
    "L": 6,
}


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


def display_virtual_node(node_data, tree_dict, c_puct):
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
    prior = node_data.get("prior", 0.0)
    u_value = node_data.get("u_value", 0.0)

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
        html.P(f"Prior (P): {prior:.4f}", style={"fontWeight": "bold"}),
        html.P("Q-Value: 0.0 (unvisited)"),
        html.P(f"Exploration (U): {u_value:.3f}", style={"color": "#0066cc"}),
        html.P(f"PUCT Total: {u_value:.3f}", style={"fontWeight": "bold"}),
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

    cell_size = 12
    height = len(board)
    width = len(board[0]) if board else 10

    img = Image.new("RGB", (width * cell_size, height * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                color_idx = board_piece_types[y][x]
                if color_idx is not None and color_idx < len(PIECE_COLORS):
                    color = PIECE_COLORS[color_idx]
                else:
                    color = (80, 80, 80)

                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

    # Grid
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
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

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
    piece_name = (
        PIECE_NAMES[piece_type] if piece_type < len(PIECE_NAMES) else f"P{piece_type}"
    )

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
            "This piece outcome has not been explored yet. "
            "The board shown is the state after the parent's action was executed, "
            "before this piece would be added to the queue.",
            style={"fontSize": "11px", "color": "#666", "fontStyle": "italic"},
        ),
    ]

    # Render the chance node's board (state after action, before piece spawn)
    board = parent["board"]
    board_piece_types = parent["board_piece_types"]

    cell_size = 12
    height = len(board)
    width = len(board[0]) if board else 10

    img = Image.new("RGB", (width * cell_size, height * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                color_idx = board_piece_types[y][x]
                if color_idx is not None and color_idx < len(PIECE_COLORS):
                    color = PIECE_COLORS[color_idx]
                else:
                    color = (80, 80, 80)

                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

    # Grid
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
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

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
    tree, max_nodes: int = 500, show_unvisited: bool = True, c_puct: float = 1.0
):
    """Convert MCTSTreeExport to Cytoscape elements."""
    elements = []

    # Limit nodes for performance
    nodes_to_show = min(len(tree.nodes), max_nodes)

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
                        edge_label = (
                            PIECE_NAMES[child.edge_from_parent]
                            if child.edge_from_parent < NUM_PIECE_TYPES
                            else str(child.edge_from_parent)
                        )

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
            sqrt_parent = max(node.visit_count, 1) ** 0.5
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
            # For chance nodes, show unvisited piece outcomes
            # All 7 pieces are potentially possible (simplification - ignores bag constraints)
            for piece_type in range(NUM_PIECE_TYPES):
                # Skip if this piece already has a child
                if piece_type in visited_pieces.get(node.id, set()):
                    continue

                virtual_id = f"vp_{node.id}_{piece_type}"
                piece_name = PIECE_NAMES[piece_type]

                elements.append(
                    {
                        "data": {
                            "id": virtual_id,
                            "label": f"{piece_name}\n(unvisited)",
                            "node_type": "virtual_decision",
                            "visit_count": 0,
                            "mean_value": 0.0,
                            "value_sum": 0.0,
                            "attack": 0,
                            "is_terminal": False,
                            "move_number": node.move_number,
                            "edge_from_parent": piece_type,
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
                            "label": piece_name,
                        },
                        "classes": "unvisited-edge",
                    }
                )

    return elements


# Create Dash app
app = dash.Dash(__name__)
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
                html.Label("Seed:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="seed",
                    type="number",
                    value=42,
                    style={"width": "60px", "marginRight": "15px"},
                ),
                html.Label("Current:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="current-piece",
                    type="text",
                    placeholder="I/O/T/S/Z/J/L or 0-6",
                    style={"width": "150px", "marginRight": "10px"},
                ),
                html.Label("Hold:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="hold-piece",
                    type="text",
                    placeholder="optional piece",
                    style={"width": "110px", "marginRight": "10px"},
                ),
                html.Label("Hold Used:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="hold-used",
                    type="text",
                    placeholder="true/false",
                    style={"width": "90px", "marginRight": "10px"},
                ),
                html.Label("Queue:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="queue-input",
                    type="text",
                    placeholder="comma-separated, e.g. I,T,L,S,O",
                    style={"width": "230px", "marginRight": "10px"},
                ),
                html.Label("Max Nodes:", style={"marginRight": "5px"}),
                dcc.Input(
                    id="max-nodes-slider",
                    type="number",
                    value=200,
                    min=50,
                    max=1000,
                    step=50,
                    style={"width": "60px", "marginRight": "15px"},
                ),
                html.Button("Run (All Sims)", id="run-button", n_clicks=0),
                html.Button(
                    "Step (+1)",
                    id="step-button",
                    n_clicks=0,
                    style={"marginLeft": "10px"},
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
    Output("cytoscape-tree", "elements"),
    Output("sim-counter", "children"),
    Input("run-button", "n_clicks"),
    Input("step-button", "n_clicks"),
    Input("show-unvisited", "value"),
    State("model-path", "value"),
    State("num-simulations", "value"),
    State("c-puct", "value"),
    State("seed", "value"),
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
    run_clicks,
    step_clicks,
    show_unvisited_value,
    model_path,
    num_sims,
    c_puct,
    seed,
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
        return dash.no_update, dash.no_update, dash.no_update, [], dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    max_sims = num_sims or 100

    if triggered_id == "run-button":
        # Run full search to configured simulation count
        sims_to_run = max_sims
    else:
        # Step - add one more simulation
        sims_to_run = min((sims_done or 0) + 1, max_sims)

    # Create config with current number of simulations
    config = MCTSConfig()
    config.num_simulations = sims_to_run
    config.c_puct = c_puct if c_puct is not None else float(VIZ_DEFAULTS["c_puct"])
    config.temperature = 0.0
    config.dirichlet_alpha = 0.3
    config.dirichlet_epsilon = 0.25
    agent = MCTSAgent(config)

    # Load model
    if not Path(model_path).exists():
        return (
            None,
            None,
            0,
            [
                {
                    "data": {"id": "error", "label": f"Model not found: {model_path}"},
                    "classes": "decision",
                }
            ],
            "Error: Model not found",
        )

    if not agent.load_model(model_path):
        model_suffix = Path(model_path).suffix.lower()
        if model_suffix == ".pt":
            error_label = (
                "Failed to load model: .pt checkpoint provided. "
                "Use an ONNX file (e.g., training_runs/.../checkpoints/latest.onnx)."
            )
        else:
            error_label = "Failed to load model: expected an ONNX file"
        return (
            None,
            None,
            0,
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
    result = agent.search_with_tree(env, add_noise=False, move_number=0)
    if result is None:
        return (
            None,
            None,
            0,
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
    elements = build_cytoscape_elements(
        tree, max_nodes or 200, show_unvisited, config.c_puct
    )

    # Cache TetrisEnv states for computing resulting boards of virtual nodes
    global _env_cache
    _env_cache.clear()
    for n in tree.nodes:
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
                "board_piece_types": list(n.state.get_board_piece_types()),
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
    }

    return (
        tree_dict,
        {"seed": seed},
        sims_to_run,
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
    node_id_str = str(node_data["id"])

    # Handle virtual (unvisited) chance nodes (from decision node actions)
    if node_id_str.startswith("v_") and not node_id_str.startswith("vp_"):
        return display_virtual_node(node_data, tree_dict, c_puct)

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

    if node["node_type"] == "decision":
        details.extend(
            [
                html.P(f"Move Number: {node['move_number']}"),
                html.P(f"Terminal: {node['is_terminal']}"),
                html.P(f"Valid Actions: {len(node['valid_actions'])}"),
            ]
        )

        if node["action_priors"]:
            details.append(html.Hr())
            details.append(
                html.H4(
                    "Top 5 by Prior (with current PUCT score)",
                    style={"marginBottom": "10px"},
                )
            )

            action_to_prior = dict(zip(node["valid_actions"], node["action_priors"]))
            child_by_action: dict[int, dict] = {}
            for child_id in node["children"]:
                child = tree_dict["nodes"][child_id]
                action_idx = child.get("edge_from_parent")
                if action_idx is not None:
                    child_by_action[action_idx] = child

            sqrt_parent = node["visit_count"] ** 0.5 if node["visit_count"] > 0 else 0.0
            top_prior_rows = []
            for action_idx, prior in action_to_prior.items():
                child = child_by_action.get(action_idx)
                n_child = child["visit_count"] if child is not None else 0
                q_value = child["mean_value"] if child is not None else 0.0
                u_value = c_puct * prior * sqrt_parent / (1 + n_child)
                puct_total = q_value + u_value
                top_prior_rows.append(
                    {
                        "action": action_idx,
                        "prior": prior,
                        "q": q_value,
                        "u": u_value,
                        "puct": puct_total,
                        "visits": n_child,
                    }
                )

            top_prior_rows.sort(key=lambda row: row["prior"], reverse=True)
            top_prior_rows = top_prior_rows[:5]

            top_prior_header = html.Tr(
                [
                    html.Th("Action", style={"padding": "4px", "textAlign": "left"}),
                    html.Th("P", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("Q", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("U", style={"padding": "4px", "textAlign": "right"}),
                    html.Th(
                        "Q+U", style={"padding": "4px", "textAlign": "right"}
                    ),
                    html.Th("N", style={"padding": "4px", "textAlign": "right"}),
                ]
            )
            top_prior_table_rows = [top_prior_header]
            for row in top_prior_rows:
                top_prior_table_rows.append(
                    html.Tr(
                        [
                            html.Td(
                                f"a{row['action']}",
                                style={"padding": "4px"},
                            ),
                            html.Td(
                                f"{row['prior']:.4f}",
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                            html.Td(
                                f"{row['q']:.4f}",
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                            html.Td(
                                f"{row['u']:.4f}",
                                style={
                                    "padding": "4px",
                                    "textAlign": "right",
                                    "color": "#0066cc",
                                },
                            ),
                            html.Td(
                                f"{row['puct']:.4f}",
                                style={
                                    "padding": "4px",
                                    "textAlign": "right",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Td(
                                str(row["visits"]),
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                        ]
                    )
                )

            details.append(
                html.Table(
                    top_prior_table_rows,
                    style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                        "fontSize": "12px",
                    },
                )
            )

        # Compute PUCT breakdown for each child action
        if node["children"] and node["visit_count"] > 0:
            details.append(html.Hr())
            details.append(
                html.H4(
                    "Child Actions (PUCT Breakdown)", style={"marginBottom": "10px"}
                )
            )
            details.append(
                html.P(
                    "PUCT = Q + U, where U = c_puct * P * sqrt(N_parent) / (1 + N_child)",
                    style={"fontSize": "11px", "color": "#666", "marginBottom": "10px"},
                )
            )

            # Build action->prior mapping
            action_to_prior = dict(zip(node["valid_actions"], node["action_priors"]))

            # Gather child info with PUCT terms
            child_info = []
            sqrt_parent = node["visit_count"] ** 0.5

            for child_id in node["children"]:
                child = tree_dict["nodes"][child_id]
                action_idx = child.get("edge_from_parent")
                if action_idx is None:
                    continue

                prior = action_to_prior.get(action_idx, 0.0)
                n_child = child["visit_count"]
                q_value = child["mean_value"]

                # Exploration term: U = c_puct * P * sqrt(N_parent) / (1 + N_child)
                u_value = c_puct * prior * sqrt_parent / (1 + n_child)
                puct_total = q_value + u_value

                child_info.append(
                    {
                        "action": action_idx,
                        "prior": prior,
                        "visits": n_child,
                        "q": q_value,
                        "u": u_value,
                        "puct": puct_total,
                        "attack": child.get("attack", 0),
                    }
                )

            # Sort by PUCT score descending
            child_info.sort(key=lambda x: x["puct"], reverse=True)

            # Display as a table
            table_header = html.Tr(
                [
                    html.Th("Action", style={"padding": "4px", "textAlign": "left"}),
                    html.Th("N", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("P", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("Q", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("U", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("PUCT", style={"padding": "4px", "textAlign": "right"}),
                    html.Th("Atk", style={"padding": "4px", "textAlign": "right"}),
                ]
            )

            table_rows = [table_header]
            for i, info in enumerate(child_info[:15]):  # Show top 15
                is_best = i == 0
                row_style = {"backgroundColor": "#e6ffe6"} if is_best else {}
                table_rows.append(
                    html.Tr(
                        [
                            html.Td(
                                f"a{info['action']}",
                                style={
                                    "padding": "4px",
                                    "fontWeight": "bold" if is_best else "normal",
                                },
                            ),
                            html.Td(
                                str(info["visits"]),
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                            html.Td(
                                f"{info['prior']:.3f}",
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                            html.Td(
                                f"{info['q']:.2f}",
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                            html.Td(
                                f"{info['u']:.2f}",
                                style={
                                    "padding": "4px",
                                    "textAlign": "right",
                                    "color": "#0066cc",
                                },
                            ),
                            html.Td(
                                f"{info['puct']:.2f}",
                                style={
                                    "padding": "4px",
                                    "textAlign": "right",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Td(
                                str(info["attack"]),
                                style={"padding": "4px", "textAlign": "right"},
                            ),
                        ],
                        style=row_style,
                    )
                )

            details.append(
                html.Table(
                    table_rows,
                    style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                        "fontSize": "12px",
                    },
                )
            )

            if len(child_info) > 15:
                details.append(
                    html.P(
                        f"... and {len(child_info) - 15} more children",
                        style={"fontSize": "11px", "color": "#666"},
                    )
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

        # Show parent info if available
        parent_id = node.get("parent_id")
        if parent_id is not None:
            parent = tree_dict["nodes"][parent_id]
            edge = node.get("edge_from_parent")
            if edge is not None and parent["action_priors"]:
                action_to_prior = dict(
                    zip(parent["valid_actions"], parent["action_priors"])
                )
                prior = action_to_prior.get(edge, 0.0)
                details.append(html.Hr())
                details.append(html.H4("Selection Info"))
                details.append(html.P(f"Prior (P): {prior:.4f}"))
                details.append(html.P(f"Parent visits (N_parent): {parent['visit_count']}"))
                details.append(html.P(f"Child visits (N_child): {node['visit_count']}"))
                if parent["visit_count"] > 0:
                    q_value = node["mean_value"]
                    n_parent = parent["visit_count"]
                    n_child = node["visit_count"]
                    sqrt_parent = parent["visit_count"] ** 0.5
                    u_value = c_puct * prior * sqrt_parent / (1 + n_child)
                    puct_total = q_value + u_value
                    details.append(
                        html.P(
                            f"Q: {q_value:.6f}",
                        )
                    )
                    details.append(
                        html.P(
                            f"Exploration (U): {u_value:.3f}",
                            style={"color": "#0066cc"},
                        )
                    )
                    details.append(
                        html.P(
                            (
                                "U = c_puct * P * sqrt(N_parent) / (1 + N_child) = "
                                f"{c_puct:.3f} * {prior:.6f} * sqrt({n_parent}) / (1 + {n_child}) = "
                                f"{u_value:.6f}"
                            ),
                            style={"fontFamily": "monospace", "fontSize": "12px"},
                        )
                    )
                    details.append(
                        html.P(
                            f"PUCT = Q + U = {q_value:.6f} + {u_value:.6f} = {puct_total:.6f}",
                            style={"fontFamily": "monospace", "fontSize": "12px"},
                        )
                    )
                    details.append(
                        html.P(
                            f"Argmax score used at parent: {puct_total:.6f}",
                            style={"fontWeight": "bold"},
                        )
                    )

    # Render board
    board = node["board"]
    board_piece_types = node["board_piece_types"]

    cell_size = 12
    height = len(board)
    width = len(board[0]) if board else 10

    img = Image.new("RGB", (width * cell_size, height * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                color_idx = board_piece_types[y][x]
                if color_idx is not None and color_idx < len(PIECE_COLORS):
                    color = PIECE_COLORS[color_idx]
                else:
                    color = (80, 80, 80)

                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

    # Grid
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
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

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


# Load dagre layout extension
cyto.load_extra_layouts()


def main():
    """Run the MCTS visualizer."""
    print("Starting MCTS Visualizer...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
