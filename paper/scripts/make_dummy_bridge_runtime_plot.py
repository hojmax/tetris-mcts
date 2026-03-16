from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

logger = structlog.get_logger()


@dataclass(frozen=True)
class PlotPoint:
    simulations: int
    runtime_ms: float
    bridge: float


@dataclass(frozen=True)
class Curve:
    name: str
    color: str
    points: list[PlotPoint]


@dataclass
class PlotArgs:
    output_path: Path = PROJECT_ROOT / "paper" / "plots" / "bridge_vs_runtime_dummy.svg"
    width: int = 1280
    height: int = 820


def build_dummy_curves() -> list[Curve]:
    """Dummy fronts for layout prototyping only."""
    return [
        Curve(
            name="Baseline",
            color="#4C78A8",
            points=[
                PlotPoint(simulations=64, runtime_ms=5_500, bridge=0.32),
                PlotPoint(simulations=128, runtime_ms=10_000, bridge=0.40),
                PlotPoint(simulations=256, runtime_ms=19_000, bridge=0.49),
                PlotPoint(simulations=512, runtime_ms=36_000, bridge=0.56),
                PlotPoint(simulations=1_024, runtime_ms=68_000, bridge=0.63),
            ],
        ),
        Curve(
            name="Bridge-Tuned",
            color="#F58518",
            points=[
                PlotPoint(simulations=64, runtime_ms=4_800, bridge=0.37),
                PlotPoint(simulations=128, runtime_ms=8_800, bridge=0.45),
                PlotPoint(simulations=256, runtime_ms=16_500, bridge=0.53),
                PlotPoint(simulations=512, runtime_ms=31_000, bridge=0.61),
                PlotPoint(simulations=1_024, runtime_ms=59_000, bridge=0.68),
            ],
        ),
        Curve(
            name="Search-Heavy",
            color="#54A24B",
            points=[
                PlotPoint(simulations=64, runtime_ms=7_200, bridge=0.42),
                PlotPoint(simulations=128, runtime_ms=13_500, bridge=0.50),
                PlotPoint(simulations=256, runtime_ms=25_000, bridge=0.59),
                PlotPoint(simulations=512, runtime_ms=47_000, bridge=0.67),
                PlotPoint(simulations=1_024, runtime_ms=84_000, bridge=0.75),
            ],
        ),
    ]


def nice_step(span: float, target_ticks: int) -> float:
    if span <= 0:
        return 1.0
    raw_step = span / max(target_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiplier in (1, 2, 2.5, 5, 10):
        step = magnitude * multiplier
        if step >= raw_step:
            return step
    return magnitude * 10


def build_ticks(value_min: float, value_max: float, target_ticks: int) -> list[float]:
    step = nice_step(value_max - value_min, target_ticks)
    tick_min = math.floor(value_min / step) * step
    tick_max = math.ceil(value_max / step) * step
    ticks: list[float] = []
    value = tick_min
    while value <= tick_max + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def format_runtime_ms(value: float) -> str:
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def format_bridge(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def render_marker(
    shape: str,
    x: float,
    y: float,
    size: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    half = size / 2
    if shape == "circle":
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{half:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" />'
        )
    if shape == "square":
        return (
            f'<rect x="{x - half:.1f}" y="{y - half:.1f}" width="{size:.1f}" '
            f'height="{size:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.1f}" />'
        )
    if shape == "diamond":
        points = [
            (x, y - half),
            (x + half, y),
            (x, y + half),
            (x - half, y),
        ]
        return polygon(points, fill=fill, stroke=stroke, stroke_width=stroke_width)
    if shape == "triangle":
        points = [
            (x, y - half),
            (x + half * 0.95, y + half * 0.85),
            (x - half * 0.95, y + half * 0.85),
        ]
        return polygon(points, fill=fill, stroke=stroke, stroke_width=stroke_width)
    if shape == "hexagon":
        points = [
            (x - half * 0.9, y),
            (x - half * 0.45, y - half * 0.78),
            (x + half * 0.45, y - half * 0.78),
            (x + half * 0.9, y),
            (x + half * 0.45, y + half * 0.78),
            (x - half * 0.45, y + half * 0.78),
        ]
        return polygon(points, fill=fill, stroke=stroke, stroke_width=stroke_width)
    raise ValueError(f"Unsupported marker shape: {shape}")


def polygon(
    points: list[tuple[float, float]],
    *,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polygon points="{point_text}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.1f}" />'
    )


def render_svg(curves: list[Curve], width: int, height: int) -> str:
    runtime_values = [point.runtime_ms for curve in curves for point in curve.points]
    bridge_values = [point.bridge for curve in curves for point in curve.points]

    runtime_min = min(runtime_values) * 0.88
    runtime_max = max(runtime_values) * 1.08
    bridge_min = max(0.0, min(bridge_values) - 0.05)
    bridge_max = min(1.0, max(bridge_values) + 0.05)

    x_ticks = build_ticks(runtime_min, runtime_max, target_ticks=6)
    y_ticks = build_ticks(bridge_min, bridge_max, target_ticks=6)

    plot_left = 130
    plot_top = 110
    plot_width = 820
    plot_height = 560
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    legend_left = 995
    legend_top = 170

    def scale_x(runtime_ms: float) -> float:
        ratio = (runtime_max - runtime_ms) / (runtime_max - runtime_min)
        return plot_left + ratio * plot_width

    def scale_y(bridge: float) -> float:
        ratio = (bridge - bridge_min) / (bridge_max - bridge_min)
        return plot_bottom - ratio * plot_height

    simulations = sorted(
        {point.simulations for curve in curves for point in curve.points}
    )
    marker_shapes = {
        simulations[0]: "circle",
        simulations[1]: "square",
        simulations[2]: "diamond",
        simulations[3]: "triangle",
        simulations[4]: "hexagon",
    }

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<defs>",
        '<filter id="card-shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="6" stdDeviation="12" flood-color="#0F172A" flood-opacity="0.12" />',
        "</filter>",
        "</defs>",
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #16212B; }",
        ".title { font-size: 28px; font-weight: 700; }",
        ".subtitle { font-size: 15px; fill: #506070; }",
        ".axis-label { font-size: 16px; font-weight: 600; }",
        ".tick { font-size: 13px; fill: #5C6C7A; }",
        ".legend-title { font-size: 15px; font-weight: 700; }",
        ".legend-text { font-size: 14px; }",
        ".note { font-size: 13px; fill: #607080; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#F7F8FA" />',
        (
            f'<rect x="{plot_left - 40}" y="{plot_top - 55}" width="1110" height="640" '
            'rx="22" fill="#FFFFFF" filter="url(#card-shadow)" />'
        ),
        '<text class="title" x="130" y="58">Bridge vs. Whole-Game Runtime</text>',
        (
            '<text class="subtitle" x="130" y="84">'
            'Dummy data for paper layout only. Runtime axis is reversed so faster models sit to the right.'
            "</text>"
        ),
    ]

    for tick in x_ticks:
        x = scale_x(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" '
            'stroke="#E4E8ED" stroke-width="1" />'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 28}" text-anchor="middle">'
            f"{escape(format_runtime_ms(tick))}</text>"
        )

    for tick in y_ticks:
        y = scale_y(tick)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            'stroke="#E4E8ED" stroke-width="1" />'
        )
        parts.append(
            f'<text class="tick" x="{plot_left - 18}" y="{y + 4:.1f}" text-anchor="end">'
            f"{escape(format_bridge(tick))}</text>"
        )

    parts.extend(
        [
            (
                f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" '
                'stroke="#24323F" stroke-width="1.5" />'
            ),
            (
                f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" '
                'stroke="#24323F" stroke-width="1.5" />'
            ),
            (
                f'<text class="axis-label" x="{plot_left + plot_width / 2:.1f}" '
                f'y="{plot_bottom + 62}" text-anchor="middle">'
                "Whole-game runtime (ms, lower is better)</text>"
            ),
            (
                f'<text class="axis-label" x="46" y="{plot_top + plot_height / 2:.1f}" '
                f'transform="rotate(-90 46 {plot_top + plot_height / 2:.1f})" text-anchor="middle">'
                "Bridge</text>"
            ),
            (
                f'<text class="note" x="{plot_right}" y="{plot_bottom + 86}" text-anchor="end">'
                "Same marker shape = same simulation budget across all fronts"
                "</text>"
            ),
        ]
    )

    for curve in curves:
        points = [(scale_x(point.runtime_ms), scale_y(point.bridge)) for point in curve.points]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{curve.color}" '
            'stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />'
        )
        for point, (x, y) in zip(curve.points, points, strict=True):
            parts.append(
                render_marker(
                    marker_shapes[point.simulations],
                    x,
                    y,
                    15,
                    fill=curve.color,
                    stroke="#FFFFFF",
                    stroke_width=2.5,
                )
            )

    parts.extend(
        [
            f'<text class="legend-title" x="{legend_left}" y="{legend_top}">Front</text>',
            f'<text class="legend-title" x="{legend_left}" y="{legend_top + 146}">Simulations</text>',
        ]
    )

    for index, curve in enumerate(curves):
        y = legend_top + 28 + index * 28
        parts.append(
            f'<line x1="{legend_left}" y1="{y}" x2="{legend_left + 30}" y2="{y}" '
            f'stroke="{curve.color}" stroke-width="4" stroke-linecap="round" />'
        )
        parts.append(
            f'<text class="legend-text" x="{legend_left + 44}" y="{y + 5}">{escape(curve.name)}</text>'
        )

    for index, simulation in enumerate(simulations):
        y = legend_top + 172 + index * 34
        parts.append(
            render_marker(
                marker_shapes[simulation],
                legend_left + 10,
                y,
                14,
                fill="#CBD5E1",
                stroke="#334155",
                stroke_width=1.8,
            )
        )
        parts.append(
            f'<text class="legend-text" x="{legend_left + 34}" y="{y + 5}">{simulation:,} sims</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def write_plot(output_path: Path, *, width: int, height: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg = render_svg(build_dummy_curves(), width=width, height=height)
    output_path.write_text(svg, encoding="utf-8")
    logger.info("Wrote dummy bridge/runtime plot", path=output_path)


def main() -> None:
    args = parse(PlotArgs)
    write_plot(args.output_path, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
