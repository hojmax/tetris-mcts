from __future__ import annotations

import json
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
MARKER_SEQUENCE = ("o", "s", "D", "^", "v", "P", "X", "*", "<", ">")


@dataclass(frozen=True)
class PlotPoint:
    simulations: int
    runtime_ms: float
    avg_attack: float


@dataclass(frozen=True)
class Curve:
    name: str
    color: str | None
    points: list[PlotPoint]
    sort_order: int


@dataclass
class PlotArgs:
    results_root: Path = PROJECT_ROOT / "paper" / "results" / "avg_attack_vs_runtime"
    output_path: Path = PROJECT_ROOT / "paper" / "plots" / "avg_attack_vs_runtime.pdf"
    width_inches: float = 9.8
    height_inches: float = 5.8


def load_curves(results_root: Path) -> list[Curve]:
    summary_paths = sorted(results_root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(
            f"No runtime/attack result summaries found under {results_root}. "
            "Run paper/scripts/benchmark_avg_attack_vs_runtime.py first."
        )

    curves: list[Curve] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text())
        points_payload = payload["points"]
        if not isinstance(points_payload, list) or not points_payload:
            raise ValueError(f"summary file has no points: {summary_path}")
        points = [
            PlotPoint(
                simulations=int(point["simulations"]),
                runtime_ms=float(point["avg_runtime_ms"]),
                avg_attack=float(point["avg_attack"]),
            )
            for point in sorted(
                points_payload, key=lambda point: int(point["simulations"])
            )
        ]
        plot_config = payload.get("plot", {})
        if not isinstance(plot_config, dict):
            raise ValueError(f"plot config must be an object: {summary_path}")
        curves.append(
            Curve(
                name=str(payload["label"]),
                color=(
                    str(plot_config["color"])
                    if plot_config.get("color") is not None
                    else None
                ),
                points=points,
                sort_order=int(plot_config.get("sort_order", 0)),
            )
        )

    ordered_curves = sorted(curves, key=lambda curve: (curve.sort_order, curve.name))
    logger.info(
        "Loaded runtime/attack plot data",
        results_root=str(results_root),
        curves=[curve.name for curve in ordered_curves],
    )
    return ordered_curves


def format_runtime_ms(value: float, _position: float) -> str:
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def write_plot(
    results_root: Path,
    output_path: Path,
    *,
    width_inches: float,
    height_inches: float,
) -> None:
    curves = load_curves(results_root)
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

    fig, ax = plt.subplots(
        figsize=(width_inches, height_inches), constrained_layout=True
    )

    for curve_index, curve in enumerate(curves):
        runtimes = [point.runtime_ms for point in curve.points]
        avg_attacks = [point.avg_attack for point in curve.points]
        color = curve.color or f"C{curve_index}"
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
            color=curve.color or f"C{curve_index}",
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
    logger.info(
        "Wrote avg attack/runtime plot",
        path=output_path,
        results_root=results_root,
    )


def main() -> None:
    args = parse(PlotArgs)
    write_plot(
        args.results_root,
        args.output_path,
        width_inches=args.width_inches,
        height_inches=args.height_inches,
    )


if __name__ == "__main__":
    main()
