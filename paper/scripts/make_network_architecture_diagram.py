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
    output_pdf_path: Path = (
        PROJECT_ROOT / "paper" / "plots" / "network_architecture.pdf"
    )
    output_png_path: Path = (
        PROJECT_ROOT / "paper" / "plots" / "network_architecture.png"
    )
    width_inches: float = 8.6
    height_inches: float = 15.0
    dpi: int = 220


def create_figure(*, width_inches: float, height_inches: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(-2.5, 21.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def draw_board_input(ax: Axes, palette: DiagramPalette) -> None:
    board_x = 3.7
    board_y = 16.80
    board_w = 1.42
    board_h = 2.35

    outer = FancyBboxPatch(
        (board_x, board_y),
        board_w,
        board_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=palette.board_edge,
        facecolor=palette.board_fill,
        zorder=2,
    )
    ax.add_patch(outer)

    cell_w = board_w / 10.0
    cell_h = board_h / 20.0
    filled_cells = {
        row_index * 10 + col_index
        for row_index, row in enumerate(TETRIS_BOARD_ROWS)
        for col_index, cell in enumerate(row)
        if cell != "."
    }
    for row in range(20):
        for col in range(10):
            y = board_y + (19 - row) * cell_h
            x = board_x + col * cell_w
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
        board_x + board_w / 2.0,
        board_y + board_h + 0.22,
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

    # Spine center x
    cx = 4.30

    # --- Title ---
    ax.text(
        0.45,
        21.1,
        "Gated Fusion Network",
        fontsize=15.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=palette.text_main,
        zorder=20,
    )

    # ===== Board input (y=16.80..19.15, title ~19.4) =====
    draw_board_input(ax, palette)

    # ===== Conv Stem volume block =====
    # Block body: y=14.90..15.68, top layer top ~16.22, title ~16.30
    # subtitle ~14.72
    draw_volume_block(
        ax,
        BlockSpec(
            x=3.83,
            y=14.90,
            width=0.94,
            height=0.78,
            depth=4,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Conv Stem",
            subtitle="16 × 20 × 10",
        ),
        palette,
    )

    # ===== Residual Trunk volume block =====
    # Block body: y=12.00..12.92, top ~13.40, title ~13.48
    # subtitle ~11.82
    draw_volume_block(
        ax,
        BlockSpec(
            x=3.55,
            y=12.00,
            width=1.34,
            height=0.92,
            depth=4,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Residual Trunk",
            subtitle="3× blocks, 16 × 20 × 10",
        ),
        palette,
    )

    # ===== Stride-2 Reduce volume block =====
    # Block body: y=9.30..9.92, top ~10.28, title ~10.36
    # subtitle ~9.12
    draw_volume_block(
        ax,
        BlockSpec(
            x=3.72,
            y=9.30,
            width=1.08,
            height=0.62,
            depth=3,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Stride-2 Reduce",
            subtitle="32 × 10 × 5",
        ),
        palette,
    )

    # ===== Flatten box =====
    # y=7.60..8.30
    draw_box(
        ax,
        BoxSpec(
            x=3.82,
            y=7.60,
            width=1.0,
            height=0.70,
            face_color=palette.conv_fill,
            edge_color=palette.conv_edge,
            title="Flatten",
            subtitle="1600",
            title_size=10.2,
            subtitle_size=8.4,
        ),
        palette,
    )

    # ===== Concat box =====
    # y=5.95..6.75
    draw_box(
        ax,
        BoxSpec(
            x=3.40,
            y=5.95,
            width=1.80,
            height=0.80,
            face_color=palette.note_fill,
            edge_color=palette.note_edge,
            title="Concat",
            subtitle="flatten (1600) ++ board_stats_h (32)",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )

    # ===== Board Projection box =====
    # y=4.00..5.20
    draw_box(
        ax,
        BoxSpec(
            x=2.90,
            y=4.00,
            width=2.80,
            height=1.20,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="Board Projection",
            subtitle="Linear(1632 → 256 → 256)\ncached board embedding",
            title_size=10.2,
            subtitle_size=7.75,
        ),
        palette,
    )

    # ===== Concat box (board_h + aux_h) =====
    draw_box(
        ax,
        BoxSpec(
            x=3.10,
            y=3.10,
            width=2.40,
            height=0.70,
            face_color=palette.note_fill,
            edge_color=palette.note_edge,
            title="Concat",
            subtitle="board_h (256) ++ aux_h (64)",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )

    # ===== Fusion MLP box (green) =====
    draw_box(
        ax,
        BoxSpec(
            x=2.70,
            y=1.60,
            width=3.20,
            height=1.15,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="MLP",
            subtitle="Linear(320 → 256) + LayerNorm + SiLU\nResidual: 256 → 256 → 256 + skip",
            title_size=10.2,
            subtitle_size=7.55,
        ),
        palette,
    )

    # ===== Policy Head (centered under left half of fusion) =====
    head_gap = 0.25
    head_w = 2.0
    head_h = 0.82
    head_y = -0.45
    policy_x = cx - head_gap / 2 - head_w
    value_x = cx + head_gap / 2

    draw_box(
        ax,
        BoxSpec(
            x=policy_x,
            y=head_y,
            width=head_w,
            height=head_h,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="Policy Head",
            subtitle="256 → 512 → 735 logits",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )

    # ===== Value Head (centered under right half of fusion) =====
    draw_box(
        ax,
        BoxSpec(
            x=value_x,
            y=head_y,
            width=head_w,
            height=head_h,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="Value Head",
            subtitle="256 → 128 → 1 scalar",
            title_size=10.0,
            subtitle_size=7.7,
        ),
        palette,
    )

    # ===== Side boxes =====

    # --- Piece/Game Features (left side, spanning conv stem to stride-2 area) ---
    draw_list_box(
        ax,
        ListBoxSpec(
            x=0.10,
            y=12.30,
            width=3.10,
            height=3.30,
            face_color=palette.aux_fill,
            edge_color=palette.aux_edge,
            title="Auxiliary Features (61)",
            lines=[
                "Current piece (7 one-hot)",
                "Hold piece (8 incl. empty)",
                "Hold available (1)",
                "Next queue (35 = 5 × 7)",
                "Placement count (1)",
                "Combo (1)",
                "Back-to-back (1)",
                "Hidden-piece distribution (7)",
            ],
            title_size=10.5,
            line_size=7.7,
        ),
        palette,
    )

    # --- Handcrafted Board Features (right side, near Flatten/Concat) ---
    draw_list_box(
        ax,
        ListBoxSpec(
            x=5.80,
            y=7.20,
            width=3.20,
            height=2.75,
            face_color=palette.aux_fill,
            edge_color=palette.aux_edge,
            title="Handcrafted Board Features (19)",
            lines=[
                "Column heights (10)",
                "Max column height (1)",
                "Bottom-row fill counts (4)",
                "Total blocks (1)",
                "Bumpiness (1)",
                "Holes (1)",
                "Overhang fields (1)",
            ],
            title_size=10.0,
            line_size=7.7,
        ),
        palette,
    )

    # --- Board Stats Encoder (right side, between features and concat) ---
    draw_box(
        ax,
        BoxSpec(
            x=6.05,
            y=6.10,
            width=2.70,
            height=0.90,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="Board Stats Encoder",
            subtitle="Linear(19 → 32) + LayerNorm + SiLU",
            title_size=10.0,
            subtitle_size=7.55,
        ),
        palette,
    )

    # --- Aux Encoder (left side, below features) ---
    draw_box(
        ax,
        BoxSpec(
            x=0.10,
            y=3.10,
            width=2.50,
            height=1.35,
            face_color=palette.fusion_fill,
            edge_color=palette.fusion_edge,
            title="Aux Encoder",
            subtitle="Linear(61 → 64) + LayerNorm + SiLU",
            title_size=10.1,
            subtitle_size=7.55,
        ),
        palette,
    )

    # ===== Arrows: central spine =====
    # Volume block geometry reference:
    #   Conv Stem:  front y=14.90, topmost top-polygon=16.22, title baseline~16.32
    #     subtitle top ~14.72
    #   Res Trunk:  front y=12.00, topmost top-polygon=13.40, title baseline~13.50
    #     subtitle top ~11.82
    #   Stride-2:   front y=9.30, topmost top-polygon=9.98, title baseline~10.08
    #     subtitle top ~9.12

    # Board bottom (16.80) -> Conv Stem topmost polygon (16.22)
    draw_arrow(ax, start_x=cx, start_y=16.75, end_x=cx, end_y=16.26, palette=palette)
    # Conv Stem front bottom (14.90) -> Residual Trunk topmost polygon (13.40)
    # Short arrows: stop above title, start below subtitle
    draw_arrow(ax, start_x=cx, start_y=14.62, end_x=cx, end_y=13.58, palette=palette)
    # Residual Trunk front bottom (12.00) -> Stride-2 topmost polygon (9.98)
    draw_arrow(ax, start_x=cx, start_y=11.72, end_x=cx, end_y=10.16, palette=palette)
    # Stride-2 front bottom (9.30) -> Flatten top (8.30)
    draw_arrow(ax, start_x=cx, start_y=9.02, end_x=cx, end_y=8.33, palette=palette)
    # Flatten bottom (7.60) -> Concat top (6.75)
    draw_arrow(ax, start_x=cx, start_y=7.57, end_x=cx, end_y=6.78, palette=palette)
    # Concat bottom (5.95) -> Board Projection top (5.20)
    draw_arrow(ax, start_x=cx, start_y=5.92, end_x=cx, end_y=5.23, palette=palette)
    # Board Projection bottom (4.00) -> Concat top (3.80)
    draw_arrow(ax, start_x=cx, start_y=3.97, end_x=cx, end_y=3.83, palette=palette)
    # Concat bottom (3.10) -> Fusion MLP top (2.75)
    draw_arrow(ax, start_x=cx, start_y=3.07, end_x=cx, end_y=2.78, palette=palette)

    # Fusion MLP bottom (1.60) -> Policy Head top
    policy_cx = policy_x + head_w / 2
    value_cx = value_x + head_w / 2
    draw_arrow(
        ax,
        start_x=cx - 0.35,
        start_y=1.57,
        end_x=policy_cx,
        end_y=head_y + head_h + 0.03,
        palette=palette,
    )
    # Fusion MLP bottom (1.60) -> Value Head top
    draw_arrow(
        ax,
        start_x=cx + 0.35,
        start_y=1.57,
        end_x=value_cx,
        end_y=head_y + head_h + 0.03,
        palette=palette,
    )

    # ===== Side arrows =====
    # Auxiliary Features -> Aux Encoder
    draw_arrow(
        ax,
        start_x=1.65,
        start_y=12.30,
        end_x=1.65,
        end_y=4.48,
        palette=palette,
    )
    # Handcrafted Board Features -> Board Stats Encoder
    draw_arrow(ax, start_x=7.4, start_y=7.20, end_x=7.4, end_y=7.03, palette=palette)
    # Board Stats Encoder -> Concat
    draw_arrow(ax, start_x=7.4, start_y=6.10, end_x=5.20, end_y=6.40, palette=palette)
    # Aux Encoder -> Concat
    draw_arrow(ax, start_x=2.60, start_y=3.10, end_x=3.10, end_y=3.45, palette=palette)


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
