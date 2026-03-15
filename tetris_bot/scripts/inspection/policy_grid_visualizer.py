from __future__ import annotations

from dataclasses import dataclass

import dash
from dash import Input, Output, State, clientside_callback, dcc, html, callback
import plotly.graph_objects as go
from simple_parsing import parse
import structlog

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_COLORS,
    PIECE_NAMES,
)

logger = structlog.get_logger()

ROTATION_LABELS = ["0", "R", "2", "L"]
GRID_CELLS_PER_MAP = BOARD_HEIGHT * BOARD_WIDTH
AGGREGATE_PIECE_VALUE = "D"
AGGREGATE_PIECE_LABEL = "D (all layers)"

TETROMINO_CELLS: list[list[list[tuple[int, int]]]] = [
    [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    [
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
    ],
    [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
    [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 0), (0, 1), (1, 1), (0, 2)],
    ],
    [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
]


@dataclass(frozen=True)
class ScriptArgs:
    host: str = "127.0.0.1"
    port: int = 8051
    debug: bool = False


@dataclass(frozen=True)
class PlacementInfo:
    piece_type: int
    rotation: int
    grid_x: int
    grid_y: int
    anchor_x: int
    anchor_y: int
    valid: bool
    occupied_cells: tuple[tuple[int, int], ...]
    in_bounds_cells: tuple[tuple[int, int], ...]
    out_of_bounds_cells: tuple[tuple[int, int], ...]
    width: int
    height: int
    min_dx: int
    max_dx: int
    min_dy: int
    max_dy: int


@dataclass(frozen=True)
class AggregateCellInfo:
    rotation: int
    grid_x: int
    grid_y: int
    active_count: int
    inactive_count: int
    active_piece_labels: tuple[str, ...]
    union_cells: tuple[tuple[int, int], ...]


PieceSelection = int | str
REAL_PIECE_VALUES: list[int] = list(range(len(PIECE_NAMES)))
PIECE_SELECTOR_VALUES: list[PieceSelection] = REAL_PIECE_VALUES + [AGGREGATE_PIECE_VALUE]
PIECE_SELECTOR_OPTIONS = [
    {"label": name, "value": index} for index, name in enumerate(PIECE_NAMES)
] + [{"label": AGGREGATE_PIECE_LABEL, "value": AGGREGATE_PIECE_VALUE}]


def _piece_cells(piece_type: int, rotation: int) -> tuple[tuple[int, int], ...]:
    return tuple(TETROMINO_CELLS[piece_type][rotation])


def _piece_bounds(
    piece_type: int, rotation: int
) -> tuple[int, int, int, int, int, int]:
    cells = _piece_cells(piece_type, rotation)
    xs = [dx for dx, _ in cells]
    ys = [dy for _, dy in cells]
    min_dx = min(xs)
    max_dx = max(xs)
    min_dy = min(ys)
    max_dy = max(ys)
    width = max_dx - min_dx + 1
    height = max_dy - min_dy + 1
    return min_dx, max_dx, min_dy, max_dy, width, height


def build_placement_info(
    piece_type: int, rotation: int, grid_x: int, grid_y: int
) -> PlacementInfo:
    min_dx, max_dx, min_dy, max_dy, width, height = _piece_bounds(
        piece_type, rotation
    )
    anchor_x = grid_x - min_dx
    anchor_y = grid_y - min_dy
    occupied_cells = tuple(
        (anchor_x + dx, anchor_y + dy) for dx, dy in _piece_cells(piece_type, rotation)
    )
    in_bounds_cells = tuple(
        (x, y)
        for x, y in occupied_cells
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT
    )
    out_of_bounds_cells = tuple(
        (x, y)
        for x, y in occupied_cells
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT)
    )
    return PlacementInfo(
        piece_type=piece_type,
        rotation=rotation,
        grid_x=grid_x,
        grid_y=grid_y,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        valid=len(out_of_bounds_cells) == 0,
        occupied_cells=occupied_cells,
        in_bounds_cells=in_bounds_cells,
        out_of_bounds_cells=out_of_bounds_cells,
        width=width,
        height=height,
        min_dx=min_dx,
        max_dx=max_dx,
        min_dy=min_dy,
        max_dy=max_dy,
    )


def build_aggregate_cell_info(rotation: int, grid_x: int, grid_y: int) -> AggregateCellInfo:
    active_placements: list[PlacementInfo] = []
    for piece_type in REAL_PIECE_VALUES:
        info = build_placement_info(piece_type, rotation, grid_x, grid_y)
        if info.valid:
            active_placements.append(info)

    union_cells = tuple(
        sorted(
            {
                cell
                for placement in active_placements
                for cell in placement.in_bounds_cells
            }
        )
    )
    active_piece_labels = tuple(
        PIECE_NAMES[placement.piece_type] for placement in active_placements
    )
    active_count = len(active_placements)
    return AggregateCellInfo(
        rotation=rotation,
        grid_x=grid_x,
        grid_y=grid_y,
        active_count=active_count,
        inactive_count=len(REAL_PIECE_VALUES) - active_count,
        active_piece_labels=active_piece_labels,
        union_cells=union_cells,
    )


def count_valid_cells(piece_type: int, rotation: int) -> int:
    valid = 0
    for grid_y in range(BOARD_HEIGHT):
        for grid_x in range(BOARD_WIDTH):
            if build_placement_info(piece_type, rotation, grid_x, grid_y).valid:
                valid += 1
    return valid


def build_summary_counts() -> list[list[dict[str, int]]]:
    summary: list[list[dict[str, int]]] = []
    for piece_type in range(len(PIECE_NAMES)):
        piece_summary: list[dict[str, int]] = []
        for rotation in range(4):
            valid = count_valid_cells(piece_type, rotation)
            piece_summary.append(
                {
                    "valid": valid,
                    "masked": GRID_CELLS_PER_MAP - valid,
                }
            )
        summary.append(piece_summary)
    return summary


SUMMARY_COUNTS = build_summary_counts()
AGGREGATE_ROTATION_COUNTS = [
    {
        "valid": sum(
            1
            for grid_y in range(BOARD_HEIGHT)
            for grid_x in range(BOARD_WIDTH)
            if build_aggregate_cell_info(rotation, grid_x, grid_y).active_count > 0
        )
    }
    for rotation in range(4)
]
for counts in AGGREGATE_ROTATION_COUNTS:
    counts["masked"] = GRID_CELLS_PER_MAP - counts["valid"]

AGGREGATE_GRID_COUNTS_BY_ROTATION = [
    [
        [
            build_aggregate_cell_info(rotation, grid_x, grid_y).active_count
            for grid_x in range(BOARD_WIDTH)
        ]
        for grid_y in range(BOARD_HEIGHT)
    ]
    for rotation in range(4)
]
AGGREGATE_UNION_VALID = sum(
    1
    for grid_y in range(BOARD_HEIGHT)
    for grid_x in range(BOARD_WIDTH)
    if any(
        build_aggregate_cell_info(rotation, grid_x, grid_y).active_count > 0
        for rotation in range(4)
    )
)
AGGREGATE_UNION_MASKED = GRID_CELLS_PER_MAP - AGGREGATE_UNION_VALID
AGGREGATE_ROTATION_TOTAL_VALID = sum(
    counts["valid"] for counts in AGGREGATE_ROTATION_COUNTS
)
AGGREGATE_ROTATION_TOTAL_MASKED = sum(
    counts["masked"] for counts in AGGREGATE_ROTATION_COUNTS
)
TOTAL_REAL_LAYER_VALID = sum(
    counts["valid"] for piece_counts in SUMMARY_COUNTS for counts in piece_counts
)
TOTAL_REAL_LAYER_MASKED = sum(
    counts["masked"] for piece_counts in SUMMARY_COUNTS for counts in piece_counts
)


def _format_cells(cells: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"({x}, {y})" for x, y in cells) if cells else "none"


def _format_layer_labels(labels: tuple[str, ...]) -> str:
    return ", ".join(labels) if labels else "none"


def make_policy_grid_figure(piece_selection: PieceSelection, rotation: int) -> go.Figure:
    if piece_selection == AGGREGATE_PIECE_VALUE:
        customdata: list[list[list[str | int]]] = []
        for grid_y in range(BOARD_HEIGHT):
            custom_row: list[list[str | int]] = []
            for grid_x in range(BOARD_WIDTH):
                aggregate = build_aggregate_cell_info(rotation, grid_x, grid_y)
                custom_row.append(
                    [
                        aggregate.active_count,
                        aggregate.inactive_count,
                        _format_cells(aggregate.union_cells),
                        _format_layer_labels(aggregate.active_piece_labels),
                    ]
                )
            customdata.append(custom_row)

        counts = AGGREGATE_ROTATION_COUNTS[rotation]
        figure = go.Figure(
            data=[
                go.Heatmap(
                    z=AGGREGATE_GRID_COUNTS_BY_ROTATION[rotation],
                    x=list(range(BOARD_WIDTH)),
                    y=list(range(BOARD_HEIGHT)),
                    customdata=customdata,
                    showscale=True,
                    xgap=2,
                    ygap=2,
                    zmin=0,
                    zmax=len(REAL_PIECE_VALUES),
                    colorbar={"title": "Active<br>pieces"},
                    colorscale=[
                        [0.0, "#D7D9CE"],
                        [0.001, "#F2E7C9"],
                        [0.25, "#D9C36A"],
                        [0.6, "#8A9A3A"],
                        [1.0, "#355E3B"],
                    ],
                    hovertemplate=(
                        "Scheme cell: (%{x}, %{y})<br>"
                        "Active pieces: %{customdata[0]}<br>"
                        "Inactive pieces: %{customdata[1]}<br>"
                        "Union occupied cells: %{customdata[2]}<br>"
                        "Active pieces at this rotation: %{customdata[3]}<extra></extra>"
                    ),
                )
            ]
        )
        figure.update_layout(
            title=(
                f"{AGGREGATE_PIECE_LABEL} rotation {ROTATION_LABELS[rotation]}: "
                f"{counts['valid']} valid, {counts['masked']} masked "
                f"| all-rotation union: {AGGREGATE_UNION_VALID} ever-active, {AGGREGATE_UNION_MASKED} never-active"
            ),
            template="plotly_white",
            paper_bgcolor="#F7F3E8",
            plot_bgcolor="#F7F3E8",
            margin=dict(l=32, r=16, t=56, b=32),
            font=dict(family="Menlo, Consolas, monospace", color="#2F241F"),
        )
        figure.update_xaxes(
            title="Scheme x",
            tickmode="array",
            tickvals=list(range(BOARD_WIDTH)),
            side="top",
        )
        figure.update_yaxes(
            title="Scheme y",
            tickmode="array",
            tickvals=list(range(BOARD_HEIGHT)),
            autorange="reversed",
        )
        return figure

    z_values: list[list[int]] = []
    customdata: list[list[list[str | int]]] = []

    for grid_y in range(BOARD_HEIGHT):
        z_row: list[int] = []
        custom_row: list[list[str | int]] = []
        for grid_x in range(BOARD_WIDTH):
            info = build_placement_info(int(piece_selection), rotation, grid_x, grid_y)
            z_row.append(1 if info.valid else 0)
            custom_row.append(
                [
                    info.anchor_x,
                    info.anchor_y,
                    "valid" if info.valid else "masked",
                    _format_cells(info.occupied_cells),
                    _format_cells(info.out_of_bounds_cells),
                ]
            )
        z_values.append(z_row)
        customdata.append(custom_row)

    piece_name = PIECE_NAMES[int(piece_selection)]
    counts = SUMMARY_COUNTS[int(piece_selection)][rotation]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=list(range(BOARD_WIDTH)),
                y=list(range(BOARD_HEIGHT)),
                customdata=customdata,
                showscale=False,
                xgap=2,
                ygap=2,
                colorscale=[
                    [0.0, "#D7D9CE"],
                    [0.499, "#D7D9CE"],
                    [0.5, "#4D7C0F"],
                    [1.0, "#4D7C0F"],
                ],
                hovertemplate=(
                    "Scheme cell: (%{x}, %{y})<br>"
                    "Anchor: (%{customdata[0]}, %{customdata[1]})<br>"
                    "Status: %{customdata[2]}<br>"
                    "Occupied: %{customdata[3]}<br>"
                    "Off-board cells: %{customdata[4]}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title=(
            f"{piece_name} rotation {ROTATION_LABELS[rotation]} "
            f"({counts['valid']} valid, {counts['masked']} masked)"
        ),
        template="plotly_white",
        paper_bgcolor="#F7F3E8",
        plot_bgcolor="#F7F3E8",
        margin=dict(l=32, r=16, t=56, b=32),
        font=dict(family="Menlo, Consolas, monospace", color="#2F241F"),
    )
    figure.update_xaxes(
        title="Scheme x",
        tickmode="array",
        tickvals=list(range(BOARD_WIDTH)),
        side="top",
    )
    figure.update_yaxes(
        title="Scheme y",
        tickmode="array",
        tickvals=list(range(BOARD_HEIGHT)),
        autorange="reversed",
    )
    return figure


def make_preview_board(
    occupied_cells: tuple[tuple[int, int], ...], background_color: str
) -> html.Div:
    occupied = set(occupied_cells)
    grid_children: list[html.Div] = []
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            is_piece = (x, y) in occupied
            background = background_color if is_piece else "#EFE6D2"
            grid_children.append(
                html.Div(
                    style={
                        "width": "24px",
                        "height": "24px",
                        "border": "1px solid #C8B79C",
                        "background": background,
                        "boxSizing": "border-box",
                    }
                )
            )

    return html.Div(
        grid_children,
        style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({BOARD_WIDTH}, 24px)",
            "gridTemplateRows": f"repeat({BOARD_HEIGHT}, 24px)",
            "gap": "1px",
            "padding": "10px",
            "background": "#DCCDB6",
            "border": "1px solid #B89F7A",
            "borderRadius": "12px",
            "width": "fit-content",
        },
    )


def make_summary_table(
    selected_piece: PieceSelection, selected_rotation: int
) -> html.Table:
    header = html.Thead(
        html.Tr(
            [
                html.Th(
                    "Piece",
                    style={
                        "padding": "10px 12px",
                        "textAlign": "left",
                        "borderBottom": "2px solid #B89F7A",
                    },
                )
            ]
            + [
                html.Th(
                    f"Rot {label}",
                    style={
                        "padding": "10px 12px",
                        "textAlign": "left",
                        "borderBottom": "2px solid #B89F7A",
                    },
                )
                for label in ROTATION_LABELS
            ]
            + [
                html.Th(
                    "Total",
                    style={
                        "padding": "10px 12px",
                        "textAlign": "left",
                        "borderBottom": "2px solid #B89F7A",
                    },
                )
            ]
        )
    )

    body_rows: list[html.Tr] = []
    for piece_type, piece_name in enumerate(PIECE_NAMES):
        row_cells = [
            html.Td(
                piece_name,
                style={
                    "padding": "10px 12px",
                    "fontWeight": 700,
                    "borderBottom": "1px solid #D8CBB6",
                },
            )
        ]
        for rotation in range(4):
            counts = SUMMARY_COUNTS[piece_type][rotation]
            is_selected = (
                piece_type == selected_piece and rotation == selected_rotation
            )
            row_cells.append(
                html.Td(
                    [
                        html.Div(
                            f"{counts['masked']} masked",
                            style={"fontWeight": 700, "marginBottom": "2px"},
                        ),
                        html.Div(
                            f"{counts['valid']} valid",
                            style={"fontSize": "12px", "opacity": 0.8},
                        ),
                    ],
                    style={
                        "padding": "10px 12px",
                        "borderBottom": "1px solid #D8CBB6",
                        "background": "#D9E8BF" if is_selected else "transparent",
                    },
                )
            )
        total_valid = sum(counts["valid"] for counts in SUMMARY_COUNTS[piece_type])
        total_masked = sum(counts["masked"] for counts in SUMMARY_COUNTS[piece_type])
        row_cells.append(
            html.Td(
                [
                    html.Div(
                        f"{total_valid} valid total",
                        style={"fontWeight": 700, "marginBottom": "2px"},
                    ),
                    html.Div(
                        f"{total_masked} masked total",
                        style={"fontSize": "12px", "opacity": 0.8},
                    ),
                ],
                style={
                    "padding": "10px 12px",
                    "borderBottom": "1px solid #D8CBB6",
                    "background": "#D9E8BF"
                    if piece_type == selected_piece
                    else "transparent",
                },
            )
        )
        body_rows.append(html.Tr(row_cells))

    aggregate_selected = selected_piece == AGGREGATE_PIECE_VALUE
    aggregate_row = [
        html.Td(
            AGGREGATE_PIECE_LABEL,
            style={
                "padding": "10px 12px",
                "fontWeight": 700,
                "borderBottom": "1px solid #D8CBB6",
            },
        )
    ]
    for rotation in range(4):
        counts = AGGREGATE_ROTATION_COUNTS[rotation]
        aggregate_row.append(
            html.Td(
                [
                    html.Div(
                        f"{counts['masked']} masked",
                        style={"fontWeight": 700, "marginBottom": "2px"},
                    ),
                    html.Div(
                        f"{counts['valid']} valid",
                        style={"fontSize": "12px", "opacity": 0.8},
                    ),
                ],
                style={
                    "padding": "10px 12px",
                    "borderBottom": "1px solid #D8CBB6",
                    "background": "#D9E8BF"
                    if aggregate_selected and rotation == selected_rotation
                    else "transparent",
                },
            )
        )
    aggregate_row.append(
        html.Td(
            [
                html.Div(
                    f"{AGGREGATE_ROTATION_TOTAL_VALID} valid total",
                    style={"fontWeight": 700, "marginBottom": "2px"},
                ),
                html.Div(
                    f"{AGGREGATE_ROTATION_TOTAL_MASKED} masked total",
                    style={"fontSize": "12px", "opacity": 0.8},
                ),
            ],
            style={
                "padding": "10px 12px",
                "borderBottom": "1px solid #D8CBB6",
                "background": "#D9E8BF" if aggregate_selected else "transparent",
            },
        )
    )
    body_rows.append(html.Tr(aggregate_row))

    return html.Table(
        [header, html.Tbody(body_rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontFamily": "Menlo, Consolas, monospace",
            "fontSize": "13px",
        },
    )


def make_hover_details(
    piece_selection: PieceSelection,
    placement_info: PlacementInfo | None = None,
    aggregate_info: AggregateCellInfo | None = None,
) -> html.Div:
    if piece_selection == AGGREGATE_PIECE_VALUE:
        if aggregate_info is None:
            raise ValueError("aggregate_info is required for aggregate selection")
        return html.Div(
            [
                html.H3(
                    f"{AGGREGATE_PIECE_LABEL} rot {ROTATION_LABELS[aggregate_info.rotation]} @ scheme ({aggregate_info.grid_x}, {aggregate_info.grid_y})",
                    style={"margin": "0 0 12px 0", "fontSize": "20px"},
                ),
                html.P(
                    f"Active pieces: {aggregate_info.active_count} of {len(REAL_PIECE_VALUES)}",
                    style={"margin": "0 0 8px 0"},
                ),
                html.P(
                    f"Inactive pieces at this rotation here: {aggregate_info.inactive_count}",
                    style={"margin": "0 0 8px 0"},
                ),
                html.P(
                    f"Union occupied cells: {_format_cells(aggregate_info.union_cells)}",
                    style={"margin": "0 0 8px 0"},
                ),
                html.P(
                    f"Active pieces at this rotation: {_format_layer_labels(aggregate_info.active_piece_labels)}",
                    style={"margin": "0"},
                ),
            ]
        )

    if placement_info is None:
        raise ValueError("placement_info is required for real piece selection")
    piece_name = PIECE_NAMES[placement_info.piece_type]
    return html.Div(
        [
            html.H3(
                f"{piece_name} @ scheme ({placement_info.grid_x}, {placement_info.grid_y})",
                style={"margin": "0 0 12px 0", "fontSize": "20px"},
            ),
            html.P(
                f"Rotation {ROTATION_LABELS[placement_info.rotation]} | anchor=({placement_info.anchor_x}, {placement_info.anchor_y}) | status={'valid' if placement_info.valid else 'masked'}",
                style={"margin": "0 0 8px 0"},
            ),
            html.P(
                f"Bounding box: {placement_info.width}x{placement_info.height} from local x=[{placement_info.min_dx}, {placement_info.max_dx}] y=[{placement_info.min_dy}, {placement_info.max_dy}]",
                style={"margin": "0 0 8px 0"},
            ),
            html.P(
                f"In-bounds occupied cells: {_format_cells(placement_info.in_bounds_cells)}",
                style={"margin": "0 0 8px 0"},
            ),
            html.P(
                f"Off-board occupied cells: {_format_cells(placement_info.out_of_bounds_cells)}",
                style={"margin": "0"},
            ),
        ]
    )
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Policy Grid Visualizer"

app.layout = html.Div(
    [
        html.Div(id="keyboard-target", tabIndex="0", style={"outline": "none"}),
        dcc.Store(id="keyboard-event", data={"key": "", "timestamp": 0}),
        html.Div(
            [
                html.H1(
                    "Policy Grid Visualizer",
                    style={
                        "margin": "0 0 8px 0",
                        "fontFamily": "Georgia, Times New Roman, serif",
                        "fontSize": "36px",
                        "letterSpacing": "-0.02em",
                    },
                ),
                html.P(
                    "Inspect the proposed 20x10x4 placement scheme: each grid cell is a normalized translation slot for the selected piece and rotation.",
                    style={"margin": "0", "maxWidth": "900px", "lineHeight": 1.5},
                ),
            ],
            style={
                "padding": "24px 28px 18px 28px",
                "background": "linear-gradient(135deg, #F1E5C8 0%, #E7D7B3 100%)",
                "border": "1px solid #C8B79C",
                "borderRadius": "20px",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Current Piece",
                            style={"display": "block", "marginBottom": "8px"},
                        ),
                        dcc.Dropdown(
                            id="piece-dropdown",
                            options=PIECE_SELECTOR_OPTIONS,
                            value=2,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "200px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Rotation",
                            style={"display": "block", "marginBottom": "8px"},
                        ),
                        dcc.RadioItems(
                            id="rotation-radio",
                            options=[
                                {"label": label, "value": rotation}
                                for rotation, label in enumerate(ROTATION_LABELS)
                            ],
                            value=0,
                            inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                            labelStyle={"marginRight": "8px"},
                        ),
                    ],
                    style={"minWidth": "240px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Div(
                            "Hold Action",
                            style={"fontWeight": 700, "marginBottom": "6px"},
                        ),
                        html.Div(
                            "Recommended encoding: keep hold as a separate non-spatial logit from pooled fused features, then append it to the flattened placement logits.",
                            style={"lineHeight": 1.45, "fontSize": "13px"},
                        ),
                    ],
                    style={
                        "flex": "2",
                        "padding": "14px 16px",
                        "background": "#EFE6D2",
                        "border": "1px solid #D0BFA2",
                        "borderRadius": "14px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Keys",
                            style={"fontWeight": 700, "marginBottom": "6px"},
                        ),
                        html.Div(
                            "Left/Right: rotation, Up/Down: piece, 1-7 real pieces, 8 selects D aggregate.",
                            style={"lineHeight": 1.45, "fontSize": "13px"},
                        ),
                    ],
                    style={
                        "flex": "1.4",
                        "padding": "14px 16px",
                        "background": "#EFE6D2",
                        "border": "1px solid #D0BFA2",
                        "borderRadius": "14px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "end",
                "flexWrap": "wrap",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id="policy-grid",
                        clear_on_unhover=False,
                        config={"displayModeBar": False},
                    ),
                    style={
                        "flex": "1.2",
                        "minWidth": "460px",
                        "padding": "14px",
                        "background": "#FFF9EE",
                        "border": "1px solid #D0BFA2",
                        "borderRadius": "18px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            id="hover-details",
                            style={
                                "padding": "14px 16px",
                                "background": "#FFF9EE",
                                "border": "1px solid #D0BFA2",
                                "borderRadius": "18px",
                                "marginBottom": "16px",
                            },
                        ),
                        html.Div(
                            id="preview-board",
                            style={
                                "padding": "14px 16px",
                                "background": "#FFF9EE",
                                "border": "1px solid #D0BFA2",
                                "borderRadius": "18px",
                            },
                        ),
                    ],
                    style={"flex": "1", "minWidth": "360px"},
                ),
            ],
            style={"display": "flex", "gap": "18px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                html.H2(
                    "Mask Counts By Piece And Rotation",
                    style={
                        "margin": "0 0 14px 0",
                        "fontFamily": "Georgia, Times New Roman, serif",
                    },
                ),
                html.Div(id="summary-table"),
                html.Div(
                    [
                        html.Div(
                            f"Across all 28 real piece/rotation maps: {TOTAL_REAL_LAYER_MASKED} masked / {TOTAL_REAL_LAYER_VALID} valid",
                            style={"fontWeight": 700},
                        ),
                        html.Div(
                            f"Aggregate D union: {AGGREGATE_UNION_VALID} ever-active cells / {AGGREGATE_UNION_MASKED} never-active cells. This still only reflects board-boundary masking under the normalized 20x10 scheme, not collisions with an actual board state.",
                            style={"marginTop": "6px", "fontSize": "13px"},
                        ),
                    ],
                    style={"marginTop": "14px"},
                ),
            ],
            style={
                "marginTop": "18px",
                "padding": "18px",
                "background": "#FFF9EE",
                "border": "1px solid #D0BFA2",
                "borderRadius": "18px",
            },
        ),
    ],
    style={
        "minHeight": "100vh",
        "padding": "24px",
        "background": "linear-gradient(180deg, #F7F3E8 0%, #EDE2C9 100%)",
        "color": "#2F241F",
        "fontFamily": "Menlo, Consolas, monospace",
    },
)


clientside_callback(
    """
    function(nClicks) {
        if (!window._policyGridKeyboardListenerSet) {
            window._policyGridKeyboardListenerSet = true;
            document.addEventListener('keydown', function(e) {
                var tagName = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : '';
                var isTypingTarget = tagName === 'input' || tagName === 'textarea' || tagName === 'select';
                if (isTypingTarget) {
                    return;
                }

                var allowed = [
                    'ArrowLeft',
                    'ArrowRight',
                    'ArrowUp',
                    'ArrowDown',
                    '1',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7',
                    '8'
                ];
                if (!allowed.includes(e.key)) {
                    return;
                }

                e.preventDefault();
                window._lastPolicyGridKeyEvent = {key: e.key, timestamp: Date.now()};
                document.getElementById('keyboard-target').click();
            });
        }
        return window._lastPolicyGridKeyEvent || {key: '', timestamp: 0};
    }
    """,
    Output("keyboard-event", "data"),
    Input("keyboard-target", "n_clicks"),
)


@callback(
    Output("policy-grid", "figure"),
    Output("summary-table", "children"),
    Input("piece-dropdown", "value"),
    Input("rotation-radio", "value"),
)
def update_grid_and_summary(piece_type: int, rotation: int) -> tuple[go.Figure, html.Table]:
    return make_policy_grid_figure(piece_type, rotation), make_summary_table(
        piece_type, rotation
    )


@callback(
    Output("piece-dropdown", "value"),
    Output("rotation-radio", "value"),
    Input("keyboard-event", "data"),
    State("piece-dropdown", "value"),
    State("rotation-radio", "value"),
    prevent_initial_call=True,
)
def handle_keyboard_event(
    keyboard_event: dict[str, int | str] | None,
    current_piece: PieceSelection,
    current_rotation: int,
) -> tuple[PieceSelection, int]:
    if not keyboard_event:
        return current_piece, current_rotation

    key = keyboard_event.get("key")
    if not isinstance(key, str) or not key:
        return current_piece, current_rotation

    if key == "ArrowLeft":
        return current_piece, (current_rotation - 1) % 4
    if key == "ArrowRight":
        return current_piece, (current_rotation + 1) % 4
    current_index = PIECE_SELECTOR_VALUES.index(current_piece)
    if key == "ArrowUp":
        return PIECE_SELECTOR_VALUES[(current_index - 1) % len(PIECE_SELECTOR_VALUES)], current_rotation
    if key == "ArrowDown":
        return PIECE_SELECTOR_VALUES[(current_index + 1) % len(PIECE_SELECTOR_VALUES)], current_rotation
    if key in {"1", "2", "3", "4", "5", "6", "7"}:
        return int(key) - 1, current_rotation
    if key == "8":
        return AGGREGATE_PIECE_VALUE, current_rotation

    return current_piece, current_rotation


@callback(
    Output("hover-details", "children"),
    Output("preview-board", "children"),
    Input("piece-dropdown", "value"),
    Input("rotation-radio", "value"),
    Input("policy-grid", "hoverData"),
)
def update_hover_preview(
    piece_type: PieceSelection, rotation: int, hover_data: dict | None
) -> tuple[html.Div, html.Div]:
    if hover_data and hover_data.get("points"):
        point = hover_data["points"][0]
        grid_x = int(point["x"])
        grid_y = int(point["y"])
    else:
        grid_x = 0
        grid_y = 0

    if piece_type == AGGREGATE_PIECE_VALUE:
        aggregate_info = build_aggregate_cell_info(rotation, grid_x, grid_y)
        details = make_hover_details(piece_type, aggregate_info=aggregate_info)
        preview_body = make_preview_board(
            aggregate_info.union_cells,
            background_color="#9C6644",
        )
        preview_title = "Board Preview (union across active pieces at selected rotation)"
    else:
        placement_info = build_placement_info(int(piece_type), rotation, grid_x, grid_y)
        details = make_hover_details(piece_type, placement_info=placement_info)
        piece_color = PIECE_COLORS[int(piece_type)]
        preview_body = make_preview_board(
            placement_info.in_bounds_cells,
            background_color=f"rgb({piece_color[0]}, {piece_color[1]}, {piece_color[2]})",
        )
        preview_title = "Board Preview"

    preview = html.Div(
        [
            html.H3(
                preview_title,
                style={"margin": "0 0 12px 0", "fontSize": "20px"},
            ),
            preview_body,
        ]
    )
    return details, preview


def main(args: ScriptArgs) -> None:
    logger.info(
        "Starting policy grid visualizer",
        host=args.host,
        port=args.port,
        url=f"http://{args.host}:{args.port}",
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main(parse(ScriptArgs))
