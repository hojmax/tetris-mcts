from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import structlog
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

matplotlib.use("Agg")

logger = structlog.get_logger()
MARKER_SEQUENCE = ("o", "s", "D", "^", "h")


@dataclass(frozen=True)
class PlotPoint:
    simulations: int
    runtime_ms: float
    avg_attack: float


@dataclass(frozen=True)
class Curve:
    name: str
    color: str
    points: list[PlotPoint]


@dataclass
class PlotArgs:
    output_path: Path = PROJECT_ROOT / "paper" / "plots" / "avg_attack_vs_runtime.pdf"
    width_inches: float = 9.8
    height_inches: float = 5.8


def build_curves() -> list[Curve]:
    return [
        Curve(
            name="Baseline",
            color="#4C78A8",
            points=[
                PlotPoint(simulations=64, runtime_ms=5_500, avg_attack=5.2),
                PlotPoint(simulations=128, runtime_ms=10_000, avg_attack=6.1),
                PlotPoint(simulations=256, runtime_ms=19_000, avg_attack=6.9),
                PlotPoint(simulations=512, runtime_ms=36_000, avg_attack=7.5),
                PlotPoint(simulations=1_024, runtime_ms=68_000, avg_attack=7.9),
            ],
        ),
        Curve(
            name="Variant A",
            color="#F58518",
            points=[
                PlotPoint(simulations=64, runtime_ms=4_800, avg_attack=5.8),
                PlotPoint(simulations=128, runtime_ms=8_800, avg_attack=6.6),
                PlotPoint(simulations=256, runtime_ms=16_500, avg_attack=7.3),
                PlotPoint(simulations=512, runtime_ms=31_000, avg_attack=8.0),
                PlotPoint(simulations=1_024, runtime_ms=59_000, avg_attack=8.4),
            ],
        ),
        Curve(
            name="Variant B",
            color="#54A24B",
            points=[
                PlotPoint(simulations=64, runtime_ms=7_200, avg_attack=6.3),
                PlotPoint(simulations=128, runtime_ms=13_500, avg_attack=7.1),
                PlotPoint(simulations=256, runtime_ms=25_000, avg_attack=7.9),
                PlotPoint(simulations=512, runtime_ms=47_000, avg_attack=8.7),
                PlotPoint(simulations=1_024, runtime_ms=84_000, avg_attack=9.4),
            ],
        ),
    ]


def format_runtime_ms(value: float, _position: float) -> str:
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def write_plot(output_path: Path, *, width_inches: float, height_inches: float) -> None:
    curves = build_curves()
    simulations = sorted(
        {point.simulations for curve in curves for point in curve.points}
    )
    if len(simulations) > len(MARKER_SEQUENCE):
        raise ValueError(
            f"Only {len(MARKER_SEQUENCE)} marker shapes are defined, "
            f"but received {len(simulations)} simulation budgets."
        )
    marker_map = {
        simulation: MARKER_SEQUENCE[index]
        for index, simulation in enumerate(simulations)
    }

    fig, ax = plt.subplots(figsize=(width_inches, height_inches), constrained_layout=True)

    for curve_index, curve in enumerate(curves):
        runtimes = [point.runtime_ms for point in curve.points]
        avg_attacks = [point.avg_attack for point in curve.points]
        color = f"C{curve_index}"
        ax.plot(
            runtimes,
            avg_attacks,
            color=color,
            linewidth=1.8,
            label=curve.name,
        )
        for point in curve.points:
            ax.scatter(
                point.runtime_ms,
                point.avg_attack,
                s=55,
                marker=marker_map[point.simulations],
                color=color,
            )

    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(format_runtime_ms))
    ax.set_xlabel("Whole-game runtime (ms, lower is better)")
    ax.set_ylabel("Avg. attack")

    front_handles = [
        Line2D(
            [0],
            [0],
            color=f"C{curve_index}",
            linewidth=1.8,
            label=curve.name,
        )
        for curve_index, curve in enumerate(curves)
    ]
    simulation_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker_map[simulation],
            color="black",
            markersize=7,
            label=f"{simulation:,} sims",
        )
        for simulation in simulations
    ]

    front_legend = ax.legend(
        handles=front_handles,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    ax.add_artist(front_legend)
    ax.legend(
        handles=simulation_handles,
        title="Simulations",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote avg attack/runtime plot", path=output_path)


def main() -> None:
    args = parse(PlotArgs)
    write_plot(
        args.output_path,
        width_inches=args.width_inches,
        height_inches=args.height_inches,
    )


if __name__ == "__main__":
    main()
