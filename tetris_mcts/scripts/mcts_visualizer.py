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
from pathlib import Path

import dash
from dash import html, dcc, callback, Output, Input, State
import dash_cytoscape as cyto
from PIL import Image, ImageDraw

from tetris_core import TetrisEnv, MCTSAgent, MCTSConfig, Piece

# Piece colors (matching tetris_game.py)
PIECE_COLORS = [
    (0, 255, 255),    # I - Cyan
    (255, 255, 0),    # O - Yellow
    (128, 0, 128),    # T - Purple
    (0, 255, 0),      # S - Green
    (255, 0, 0),      # Z - Red
    (0, 0, 255),      # J - Blue
    (255, 165, 0),    # L - Orange
]

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]


def render_board_to_image(env: TetrisEnv, cell_size: int = 8) -> str:
    """Render a TetrisEnv board to a base64 PNG image."""
    board = env.get_board()
    board_colors = env.get_board_colors()
    height = len(board)
    width = len(board[0]) if board else 10

    img = Image.new("RGB", (width * cell_size, height * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                color_idx = board_colors[y][x]
                if color_idx is not None and color_idx < len(PIECE_COLORS):
                    color = PIECE_COLORS[color_idx]
                else:
                    color = (80, 80, 80)

                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

    # Draw grid lines
    for x in range(width + 1):
        draw.line([(x * cell_size, 0), (x * cell_size, height * cell_size)], fill=(40, 40, 40))
    for y in range(height + 1):
        draw.line([(0, y * cell_size), (width * cell_size, y * cell_size)], fill=(40, 40, 40))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def build_cytoscape_elements(tree, max_nodes: int = 500):
    """Convert MCTSTreeExport to Cytoscape elements."""
    elements = []

    # Limit nodes for performance
    nodes_to_show = min(len(tree.nodes), max_nodes)

    # Sort nodes by visit count to show most important ones
    sorted_indices = sorted(
        range(len(tree.nodes)),
        key=lambda i: tree.nodes[i].visit_count,
        reverse=True
    )[:nodes_to_show]
    shown_ids = set(sorted_indices)

    # Always include root
    shown_ids.add(tree.root_id)

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

        elements.append({
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
            },
            "classes": node_class,
        })

        # Edges to children
        for child_id in node.children:
            if child_id in shown_ids:
                child = tree.nodes[child_id]
                edge_label = ""
                if child.edge_from_parent is not None:
                    if is_decision:
                        edge_label = f"a{child.edge_from_parent}"
                    else:
                        edge_label = PIECE_NAMES[child.edge_from_parent] if child.edge_from_parent < 7 else str(child.edge_from_parent)

                elements.append({
                    "data": {
                        "source": str(node.id),
                        "target": str(child_id),
                        "label": edge_label,
                    }
                })

    return elements


# Create Dash app
app = dash.Dash(__name__)

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
        }
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
        }
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
        }
    },
    # Selected node
    {
        "selector": ":selected",
        "style": {
            "border-width": 3,
            "border-color": "#ff0",
        }
    },
]

app.layout = html.Div([
    html.H1("MCTS Tree Visualizer", style={"textAlign": "center"}),

    html.Div([
        # Controls
        html.Div([
            html.Label("Model Path:"),
            dcc.Input(
                id="model-path",
                type="text",
                placeholder="Path to ONNX model",
                value="checkpoints/selfplay.onnx",
                style={"width": "300px", "marginRight": "10px"}
            ),
            html.Label("Simulations:"),
            dcc.Input(
                id="num-simulations",
                type="number",
                value=100,
                min=1,
                max=1000,
                style={"width": "80px", "marginRight": "10px"}
            ),
            html.Label("Seed:"),
            dcc.Input(
                id="seed",
                type="number",
                value=42,
                style={"width": "80px", "marginRight": "10px"}
            ),
            html.Button("Run MCTS", id="run-button", n_clicks=0),
            html.Button("Step (+1 sim)", id="step-button", n_clicks=0, style={"marginLeft": "10px"}),
            html.Span(id="sim-counter", children="Simulations: 0", style={"marginLeft": "20px", "fontWeight": "bold"}),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Max Nodes to Display:"),
            dcc.Slider(
                id="max-nodes-slider",
                min=50,
                max=1000,
                step=50,
                value=200,
                marks={i: str(i) for i in range(50, 1001, 200)},
            ),
        ], style={"width": "400px", "marginBottom": "10px"}),
    ], style={"padding": "10px", "backgroundColor": "#f0f0f0"}),

    html.Div([
        # Tree visualization (left 2/3)
        html.Div([
            cyto.Cytoscape(
                id="cytoscape-tree",
                elements=[],
                style={"width": "100%", "height": "calc(100vh - 180px)", "minHeight": "500px", "border": "1px solid #ccc"},
                layout={
                    "name": "dagre",
                    "rankDir": "TB",
                    "spacingFactor": 1.5,
                },
                stylesheet=stylesheet,
                zoom=1,
                pan={"x": 0, "y": 0},
            ),
        ], style={"flex": "2", "minWidth": "0"}),

        # Node details panel (right 1/3)
        html.Div([
            html.H3("Board State", style={"marginTop": 0}),
            html.Img(id="board-image", style={"border": "1px solid #ccc"}),
            html.Div(id="state-info"),
            html.Hr(),
            html.H3("Node Details"),
            html.Div(id="node-details", children="Click a node to see details"),
        ], style={
            "flex": "1",
            "padding": "10px",
            "backgroundColor": "#f8f8f8",
            "marginLeft": "10px",
            "overflowY": "auto",
            "height": "calc(100vh - 180px)",
            "minHeight": "500px",
        }),
    ], style={"display": "flex", "flexDirection": "row"}),

    # Hidden storage for tree data
    dcc.Store(id="tree-store"),
    dcc.Store(id="env-store"),
    dcc.Store(id="sims-done-store", data=0),  # Track simulations done for stepping
], style={"fontFamily": "Arial, sans-serif", "padding": "20px"})


@callback(
    Output("tree-store", "data"),
    Output("env-store", "data"),
    Output("sims-done-store", "data"),
    Output("cytoscape-tree", "elements"),
    Output("sim-counter", "children"),
    Input("run-button", "n_clicks"),
    Input("step-button", "n_clicks"),
    State("model-path", "value"),
    State("num-simulations", "value"),
    State("seed", "value"),
    State("max-nodes-slider", "value"),
    State("tree-store", "data"),
    State("env-store", "data"),
    State("sims-done-store", "data"),
    prevent_initial_call=True,
)
def run_mcts(run_clicks, step_clicks, model_path, num_sims, seed, max_nodes,
             tree_data, env_data, sims_done):
    """Run MCTS search and update the tree."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, [], dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    max_sims = num_sims or 100

    if triggered_id == "run-button":
        # Fresh start - reset to 1 simulation
        sims_to_run = 1
    else:
        # Step - add one more simulation
        sims_to_run = min((sims_done or 0) + 1, max_sims)

    # Create config with current number of simulations
    config = MCTSConfig()
    config.num_simulations = sims_to_run
    config.c_puct = 1.0
    config.temperature = 0.0
    config.dirichlet_alpha = 0.3
    config.dirichlet_epsilon = 0.25
    agent = MCTSAgent(config)

    # Load model
    if not Path(model_path).exists():
        return None, None, 0, [{
            "data": {"id": "error", "label": f"Model not found: {model_path}"},
            "classes": "decision"
        }], "Error: Model not found"

    if not agent.load_model(model_path):
        return None, None, 0, [{
            "data": {"id": "error", "label": "Failed to load model"},
            "classes": "decision"
        }], "Error: Failed to load model"

    # Create env (same seed for consistency)
    env = TetrisEnv.with_seed(10, 20, seed or 42)

    # Run MCTS with current number of simulations
    result = agent.search_with_tree(env, add_noise=False, move_number=0)
    if result is None:
        return None, None, 0, [{
            "data": {"id": "error", "label": "MCTS failed (game over?)"},
            "classes": "decision"
        }], "Error: MCTS failed"

    mcts_result, tree = result

    # Build elements
    elements = build_cytoscape_elements(tree, max_nodes or 200)

    # Store tree data for click handling
    tree_dict = {
        "nodes": [
            {
                "id": n.id,
                "node_type": n.node_type,
                "visit_count": n.visit_count,
                "mean_value": n.mean_value,
                "value_sum": n.value_sum,
                "attack": n.attack,
                "is_terminal": n.is_terminal,
                "move_number": n.move_number,
                "valid_actions": list(n.valid_actions),
                "action_priors": list(n.action_priors),
                "children": list(n.children),
                "board": list(n.state.get_board()),
                "board_colors": list(n.state.get_board_colors()),
                "current_piece": n.state.get_current_piece().piece_type if n.state.get_current_piece() else None,
                "hold_piece": n.state.get_hold_piece().piece_type if n.state.get_hold_piece() else None,
                "queue": list(n.state.get_queue(5)),
            }
            for n in tree.nodes
        ],
        "root_id": tree.root_id,
        "selected_action": tree.selected_action,
        "num_simulations": tree.num_simulations,
    }

    return tree_dict, {"seed": seed}, sims_to_run, elements, f"Simulations: {sims_to_run}/{max_sims}"


@callback(
    Output("node-details", "children"),
    Output("board-image", "src"),
    Output("state-info", "children"),
    Input("cytoscape-tree", "tapNodeData"),
    State("tree-store", "data"),
)
def display_node_details(node_data, tree_dict):
    """Display details for clicked node."""
    if node_data is None or tree_dict is None:
        return "Click a node to see details", "", ""

    node_id = int(node_data["id"])
    if node_id >= len(tree_dict["nodes"]):
        return "Node not found", "", ""

    node = tree_dict["nodes"][node_id]

    # Format details
    details = [
        html.P(f"Node ID: {node['id']}"),
        html.P(f"Type: {node['node_type']}"),
        html.P(f"Visit Count: {node['visit_count']}"),
        html.P(f"Mean Value: {node['mean_value']:.3f}"),
        html.P(f"Value Sum: {node['value_sum']:.3f}"),
    ]

    if node["node_type"] == "decision":
        details.extend([
            html.P(f"Move Number: {node['move_number']}"),
            html.P(f"Terminal: {node['is_terminal']}"),
            html.P(f"Valid Actions: {len(node['valid_actions'])}"),
        ])
        if node["action_priors"]:
            top_priors = sorted(zip(node["valid_actions"], node["action_priors"]),
                               key=lambda x: x[1], reverse=True)[:5]
            details.append(html.P(f"Top priors: {top_priors}"))
    else:
        details.append(html.P(f"Attack: {node['attack']}"))

    # Render board
    board = node["board"]
    board_colors = node["board_colors"]

    cell_size = 12
    height = len(board)
    width = len(board[0]) if board else 10

    img = Image.new("RGB", (width * cell_size, height * cell_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                color_idx = board_colors[y][x]
                if color_idx is not None and color_idx < len(PIECE_COLORS):
                    color = PIECE_COLORS[color_idx]
                else:
                    color = (80, 80, 80)

                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

    # Grid
    for x in range(width + 1):
        draw.line([(x * cell_size, 0), (x * cell_size, height * cell_size)], fill=(40, 40, 40))
    for y in range(height + 1):
        draw.line([(0, y * cell_size), (width * cell_size, y * cell_size)], fill=(40, 40, 40))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    # State info
    state_info = [
        html.P(f"Current Piece: {PIECE_NAMES[node['current_piece']] if node['current_piece'] is not None else 'None'}"),
        html.P(f"Hold Piece: {PIECE_NAMES[node['hold_piece']] if node['hold_piece'] is not None else 'None'}"),
        html.P(f"Queue: {[PIECE_NAMES[p] for p in node['queue']]}"),
    ]

    return details, f"data:image/png;base64,{img_b64}", state_info


# Load dagre layout extension
cyto.load_extra_layouts()


def main():
    """Run the MCTS visualizer."""
    print("Starting MCTS Visualizer...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
