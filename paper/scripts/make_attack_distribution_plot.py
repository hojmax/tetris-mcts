from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import structlog
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

matplotlib.use("Agg")

logger = structlog.get_logger()


@dataclass(frozen=True)
class Distribution:
    name: str
    color: str | None
    attacks: list[int]
    sort_order: int


@dataclass
class PlotArgs:
    results_root: Path = PROJECT_ROOT / "paper" / "results" / "attack_distribution"
    output_path: Path = PROJECT_ROOT / "paper" / "plots" / "attack_distribution.pdf"
    width_inches: float = 9.0
    height_inches: float = 5.4
    bin_width: int = 1
    dpi: int = 300


def load_distributions(results_root: Path) -> list[Distribution]:
    summary_paths = sorted(results_root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(
            f"No attack distribution summaries found under {results_root}."
        )

    distributions: list[Distribution] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text())
        attacks_payload = payload["attacks"]
        if not isinstance(attacks_payload, list) or not attacks_payload:
            raise ValueError(f"summary file has no attacks: {summary_path}")
        plot_config = payload.get("plot", {})
        if not isinstance(plot_config, dict):
            raise ValueError(f"plot config must be an object: {summary_path}")
        distributions.append(
            Distribution(
                name=str(payload["label"]),
                color=(
                    str(plot_config["color"])
                    if plot_config.get("color") is not None
                    else None
                ),
                attacks=[int(a) for a in attacks_payload],
                sort_order=int(plot_config.get("sort_order", 0)),
            )
        )

    ordered = sorted(distributions, key=lambda d: (d.sort_order, d.name))
    logger.info(
        "Loaded attack distributions",
        results_root=str(results_root),
        distributions=[d.name for d in ordered],
    )
    return ordered


def write_plot(
    results_root: Path,
    output_path: Path,
    *,
    width_inches: float,
    height_inches: float,
    bin_width: int,
    dpi: int,
) -> None:
    distributions = load_distributions(results_root)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#333333",
            "axes.labelsize": 11,
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(
        figsize=(width_inches, height_inches), constrained_layout=True
    )

    overall_min = min(min(d.attacks) for d in distributions)
    overall_max = max(max(d.attacks) for d in distributions)
    bins = np.arange(overall_min, overall_max + bin_width + 1, bin_width) - 0.5

    legend_handles: list[tuple] = []
    annotations: list[tuple[float, float, str, str]] = []

    for index, distribution in enumerate(distributions):
        color = distribution.color or f"C{index}"
        attacks = np.asarray(distribution.attacks)
        mean_attack = float(attacks.mean())
        min_attack = int(attacks.min())
        max_attack = int(attacks.max())
        n_games = len(attacks)

        counts, _, _ = ax.hist(
            attacks,
            bins=bins,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85 if len(distributions) == 1 else 0.65,
        )
        ax.axvline(
            mean_attack,
            color=color,
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            alpha=0.95,
        )

        bin_centers = (bins[:-1] + bins[1:]) / 2
        for value, label in (
            (min_attack, f"Min {min_attack}"),
            (max_attack, f"Max {max_attack}"),
        ):
            bar_idx = int(np.argmin(np.abs(bin_centers - value)))
            bar_height = counts[bar_idx]
            annotations.append((value, bar_height, label, color))

        legend_handles.append(
            (
                Patch(
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.85 if len(distributions) == 1 else 0.65,
                ),
                f"{distribution.name}  (n = {n_games}, Min = {min_attack}, Max = {max_attack})",
            )
        )
        legend_handles.append(
            (
                Line2D([0], [0], color=color, linestyle=(0, (5, 3)), linewidth=1.8),
                f"Mean = {mean_attack:.2f}",
            )
        )

    # Pad y-axis so Min/Max annotations don't bump into the title/top spine
    current_top = ax.get_ylim()[1]
    ax.set_ylim(top=current_top * 1.18)

    for x, y, label, color in annotations:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    total_n = sum(len(d.attacks) for d in distributions)
    title = (
        f"Histogram of Total Episode Attack ({total_n} games)"
        if len(distributions) == 1
        else "Histogram of Total Episode Attack"
    )
    ax.set_xlabel("Total episode attack")
    ax.set_ylabel("Number of games")
    ax.set_title(title, loc="center", fontsize=14, fontweight="semibold", pad=14)
    ax.grid(True, axis="y", alpha=0.35, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    ax.tick_params(length=0)

    ax.legend(
        [h for h, _ in legend_handles],
        [t for _, t in legend_handles],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#e0e0e0",
        fontsize=9,
        handlelength=1.6,
        labelspacing=0.6,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info(
        "Wrote attack distribution plot",
        path=str(output_path),
        results_root=str(results_root),
    )


def main() -> None:
    args = parse(PlotArgs)
    write_plot(
        args.results_root,
        args.output_path,
        width_inches=args.width_inches,
        height_inches=args.height_inches,
        bin_width=args.bin_width,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
