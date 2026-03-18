from __future__ import annotations

from pathlib import Path

import matplotlib
import structlog
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

matplotlib.use("Agg")
from matplotlib import pyplot as plt

logger = structlog.get_logger()

TETRIS_BOARD_ROWS = [
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "....LLL...",
    "LL...LLLLL",
    "LLL.LLLLLL",
]


class DiagramPalette(BaseModel):
    board_fill: str = "#f3f6fb"
    board_edge: str = "#2b4c7e"
    board_grid: str = "#b8c6db"
    filled_cell: str = "#1f2430"
    conv_fill: str = "#cfe4ff"
    conv_edge: str = "#3d74b6"
    aux_fill: str = "#ffe6cc"
    aux_edge: str = "#d47b2c"
    fusion_fill: str = "#dff3e3"
    fusion_edge: str = "#3f8f58"
    head_fill: str = "#eadff7"
    head_edge: str = "#7a53a8"
    text_main: str = "#1d2430"
    text_muted: str = "#4f5d73"
    arrow: str = "#506179"
    note_fill: str = "#f7f8fa"
    note_edge: str = "#98a2b3"


class BlockSpec(BaseModel):
    x: float
    y: float
    width: float
    height: float
    depth: int
    dx: float = 0.18
    dy: float = 0.12
    face_color: str
    edge_color: str
    title: str
    subtitle: str


class BoxSpec(BaseModel):
    x: float
    y: float
    width: float
    height: float
    face_color: str
    edge_color: str
    title: str
    subtitle: str
    title_size: float = 10.5
    subtitle_size: float = 8.5
    radius: float = 0.08


class ListBoxSpec(BaseModel):
    x: float
    y: float
    width: float
    height: float
    face_color: str
    edge_color: str
    title: str
    lines: list[str]
    title_size: float = 11.0
    line_size: float = 7.8
    radius: float = 0.08


@dataclass
class DiagramArgs:
    output_pdf_path: Path = PROJECT_ROOT / "paper" / "plots" / "network_architecture.pdf"
    output_png_path: Path = PROJECT_ROOT / "paper" / "plots" / "network_architecture.png"
    width_inches: float = 8.6
    height_inches: float = 12.2
    dpi: int = 220


def create_figure(*, width_inches: float, height_inches: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(-1.15, 15.1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def draw_board_input(ax: Axes, palette: DiagramPalette) -> None:
    outer = FancyBboxPatch(
        (3.7, 10.55),
        1.42,
        2.35,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=palette.board_edge,
        facecolor=palette.board_fill,
        zorder=2,
    )
    ax.add_patch(outer)

    cell_w = 1.42 / 10.0
    cell_h = 2.35 / 20.0
    filled_cells = {
        row_index * 10 + col_index
        for row_index, row in enumerate(TETRIS_BOARD_ROWS)
        for col_index, cell in enumerate(row)
        if cell != "."
    }
    for row in range(20):
        for col in range(10):
            y = 10.55 + (19 - row) * cell_h
            x = 3.7 + col * cell_w
            flat_index = row * 10 + col
            cell_color = palette.filled_cell if flat_index in filled_cells else "white"
            cell = Rectangle(
                (x, y),
                cell_w,
                cell_h,
                linewidth=0.35,
                edgecolor=palette.board_grid,
                facecolor=cell_color,
                zorder=3,
            )
            ax.add_patch(cell)

    ax.text(
        4.41,
        13.12,
        "Input Tetris Board",
        ha="center",
        va="bottom",
        fontsize=10.9,
        fontweight="bold",
        color=palette.text_main,
        zorder=20,
    )


def draw_volume_block(ax: Axes, spec: BlockSpec, palette: DiagramPalette) -> None:
    for layer_index in range(spec.depth - 1, -1, -1):
        x = spec.x + layer_index * spec.dx
        y = spec.y + layer_index * spec.dy
        front = Rectangle(
            (x, y),
            spec.width,
            spec.height,
            linewidth=1.4,
            edgecolor=spec.edge_color,
            facecolor=spec.face_color,
            alpha=0.98,
            zorder=5 + layer_index,
        )
        top = Polygon(
            [
                [x, y + spec.height],
                [x + spec.dx, y + spec.height + spec.dy],
                [x + spec.width + spec.dx, y + spec.height + spec.dy],
                [x + spec.width, y + spec.height],
            ],
            closed=True,
            linewidth=1.1,
            edgecolor=spec.edge_color,
            facecolor=spec.face_color,
            alpha=0.82,
            zorder=4 + layer_index,
        )
        side = Polygon(
            [
                [x + spec.width, y],
                [x + spec.width + spec.dx, y + spec.dy],
                [x + spec.width + spec.dx, y + spec.height + spec.dy],
                [x + spec.width, y + spec.height],
            ],
            closed=True,
            linewidth=1.1,
            edgecolor=spec.edge_color,
            facecolor=spec.face_color,
            alpha=0.88,
            zorder=4 + layer_index,
        )
        ax.add_patch(front)
        ax.add_patch(top)
        ax.add_patch(side)

    center_x = spec.x + spec.width / 2.0 + (spec.depth - 1) * spec.dx * 0.5
    ax.text(
        center_x,
        spec.y + spec.height + spec.depth * spec.dy + 0.08,
        spec.title,
        ha="center",
        va="bottom",
        fontsize=10.4,
        fontweight="bold",
        color=palette.text_main,
        zorder=20,
    )
    ax.text(
        center_x,
        spec.y - 0.18,
        spec.subtitle,
        ha="center",
        va="top",
        fontsize=7.4,
        linespacing=1.1,
        color=palette.text_muted,
        zorder=20,
    )


def draw_box(ax: Axes, spec: BoxSpec, palette: DiagramPalette) -> None:
    box = FancyBboxPatch(
        (spec.x, spec.y),
        spec.width,
        spec.height,
        boxstyle=f"round,pad=0.02,rounding_size={spec.radius}",
        linewidth=1.5,
        edgecolor=spec.edge_color,
        facecolor=spec.face_color,
        zorder=10,
    )
    ax.add_patch(box)
    ax.text(
        spec.x + spec.width / 2.0,
        spec.y + spec.height - 0.16,
        spec.title,
        ha="center",
        va="top",
        fontsize=spec.title_size,
        fontweight="bold",
        color=palette.text_main,
        zorder=20,
    )
    ax.text(
        spec.x + spec.width / 2.0,
        spec.y + spec.height - 0.48,
        spec.subtitle,
        ha="center",
        va="top",
        fontsize=spec.subtitle_size,
        color=palette.text_muted,
        linespacing=1.16,
        zorder=20,
    )


def draw_list_box(ax: Axes, spec: ListBoxSpec, palette: DiagramPalette) -> None:
    box = FancyBboxPatch(
        (spec.x, spec.y),
        spec.width,
        spec.height,
        boxstyle=f"round,pad=0.02,rounding_size={spec.radius}",
        linewidth=1.5,
        edgecolor=spec.edge_color,
        facecolor=spec.face_color,
        zorder=10,
    )
    ax.add_patch(box)
    ax.text(
        spec.x + 0.16,
        spec.y + spec.height - 0.18,
        spec.title,
        ha="left",
        va="top",
        fontsize=spec.title_size,
        fontweight="bold",
        color=palette.text_main,
        zorder=20,
    )
    ax.text(
        spec.x + 0.16,
        spec.y + spec.height - 0.48,
        "\n".join(spec.lines),
        ha="left",
        va="top",
        fontsize=spec.line_size,
        color=palette.text_muted,
        linespacing=1.28,
        zorder=20,
    )


def draw_arrow(
    ax: Axes,
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    palette: DiagramPalette,
    label: str | None = None,
    label_x: float | None = None,
    label_y: float | None = None,
) -> None:
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.7,
        color=palette.arrow,
        shrinkA=0,
        shrinkB=0,
        zorder=9,
    )
    ax.add_patch(arrow)
    if label is not None and label_x is not None and label_y is not None:
        ax.text(
            label_x,
            label_y,
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            color=palette.text_muted,
            zorder=20,
        )


def draw_architecture(ax: Axes) -> None:
    palette = DiagramPalette()

    ax.text(
        0.45,
        14.75,
        "Gated Fusion Network",
        fontsize=15.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=palette.text_main,
        zorder=20,
    )

    draw_list_box(
        ax,
        ListBoxSpec(
            x=0.45,
            y=8.0,
            width=2.75,
            height=2.75,
            face_color=palette.aux_fill,
            edge_color=palette.aux_edge,
            title="Piece/Game Features (61)",
            lines=[
                "Current piece (7 one-hot)",
                "Hold piece (8 incl. empty)",
                "Hold available (1)",
                "Next queue (35 = 5 x 7)",
                "Placement count (1)",
                "Combo (1)",
                "Back-to-back (1)",
                "Hidden-piece distribution (7)",
            ],
            title_size=10.7,
            line_size=7.55,
        ),
        palette,
    )

    draw_board_input(ax, palette)

    draw_list_box(
        ax,
        ListBoxSpec(
            x=5.55,
            y=2.15,
            width=3.0,
            height=2.55,
            face_color=palette.aux_fill,
            edge_color=palette.aux_edge,
            title="Custom Board Stats (19)",
            lines=[
                "Column heights (10)",
                "Max column height (1)",
                "Bottom-row fill counts (4)",
                "Total blocks (1)",
                "Bumpiness (1)",
                "Holes (1)",
                "Overhang fields (1)",
            ],
            title_size=10.7,
            line_size=7.55,
        ),
        palette,
    )

    draw_volume_block(
        ax,
        BlockSpec(
            x=3.83,
            y=8.85,
            width=0.94,
            height=0.78,
            depth=4,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Conv Stem",
            subtitle="16 x 20 x 10",
        ),
        palette,
    )
    draw_volume_block(
        ax,
        BlockSpec(
            x=3.55,
            y=7.1,
            width=1.34,
            height=0.92,
            depth=4,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Residual Trunk",
            subtitle="3x blocks, 16 x 20 x 10",
        ),
        palette,
    )
    draw_volume_block(
        ax,
        BlockSpec(
            x=3.72,
            y=5.45,
            width=1.08,
            height=0.62,
            depth=3,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Stride-2 Reduce",
            subtitle="32 x 10 x 5",
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=3.92,
            y=4.0,
            width=0.82,
            height=0.64,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Flatten",
            subtitle="1600",
            title_size=10.2,
            subtitle_size=8.4,
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=0.55,
            y=1.35,
            width=2.18,
            height=1.5,
            face_color=palette.aux_fill,
            edge_color=palette.aux_edge,
            title="Aux Encoder + Gates",
            subtitle="aux_fc: 61 -> 64\nLayerNorm + SiLU -> aux_h\n gate_fc: 64 -> 128\naux_proj: 64 -> 128",
            title_size=10.1,
            subtitle_size=7.55,
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=3.6,
            y=2.75,
            width=1.48,
            height=0.7,
            face_color=palette.note_fill,
            edge_color=palette.note_edge,
            title="Concat",
            subtitle="flatten + 19 board stats",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=3.02,
            y=1.55,
            width=2.56,
            height=0.92,
            face_color="#d7efff",
            edge_color=palette.conv_edge,
            title="board_proj",
            subtitle="Linear(1600 + 19 -> 128)\ncached board_h embedding",
            title_size=10.2,
            subtitle_size=7.75,
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=2.95,
            y=0.25,
            width=2.7,
            height=1.02,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="Gated Fusion",
            subtitle="fused = board_h * (1 + gate)\n        + aux_proj(aux_h)\n1 residual fusion MLP -> 128-d features",
            title_size=10.2,
            subtitle_size=7.55,
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=2.05,
            y=-0.9,
            width=1.9,
            height=0.78,
            face_color=palette.head_fill,
            edge_color=palette.head_edge,
            title="Policy Head",
            subtitle="128 -> 256 -> 735 logits",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )
    draw_box(
        ax,
        BoxSpec(
            x=4.65,
            y=-0.9,
            width=1.9,
            height=0.78,
            face_color=palette.head_fill,
            edge_color=palette.head_edge,
            title="Value Head",
            subtitle="128 -> 64 -> 1 scalar",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )

    draw_arrow(ax, start_x=4.41, start_y=10.48, end_x=4.41, end_y=9.62, palette=palette)
    draw_arrow(ax, start_x=4.41, start_y=8.72, end_x=4.41, end_y=8.08, palette=palette)
    draw_arrow(ax, start_x=4.41, start_y=7.02, end_x=4.28, end_y=6.16, palette=palette)
    draw_arrow(ax, start_x=4.28, start_y=5.38, end_x=4.28, end_y=4.72, palette=palette)
    draw_arrow(ax, start_x=4.28, start_y=3.98, end_x=4.28, end_y=3.48, palette=palette)
    draw_arrow(ax, start_x=4.28, start_y=2.72, end_x=4.28, end_y=2.48, palette=palette)
    draw_arrow(ax, start_x=4.28, start_y=1.52, end_x=4.28, end_y=1.3, palette=palette)
    draw_arrow(ax, start_x=4.3, start_y=0.24, end_x=3.0, end_y=0.02, palette=palette)
    draw_arrow(ax, start_x=4.3, start_y=0.24, end_x=5.6, end_y=0.02, palette=palette)

    draw_arrow(
        ax,
        start_x=1.82,
        start_y=8.0,
        end_x=1.82,
        end_y=2.85,
        palette=palette,
    )
    draw_arrow(ax, start_x=7.25, start_y=2.15, end_x=5.08, end_y=3.08, palette=palette)
    draw_arrow(ax, start_x=2.73, start_y=1.95, end_x=2.95, end_y=0.92, palette=palette)

    ax.text(
        5.35,
        1.02,
        "board_h",
        fontsize=8.3,
        color=palette.text_muted,
        ha="left",
        va="center",
        zorder=20,
    )


def write_outputs(
    output_pdf_path: Path,
    output_png_path: Path,
    *,
    width_inches: float,
    height_inches: float,
    dpi: int,
) -> None:
    fig, ax = create_figure(width_inches=width_inches, height_inches=height_inches)
    draw_architecture(ax)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf_path, bbox_inches="tight")
    fig.savefig(output_png_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info(
        "Wrote network architecture diagram",
        pdf_path=output_pdf_path,
        png_path=output_png_path,
    )


def main() -> None:
    args = parse(DiagramArgs)
    write_outputs(
        args.output_pdf_path,
        args.output_png_path,
        width_inches=args.width_inches,
        height_inches=args.height_inches,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
