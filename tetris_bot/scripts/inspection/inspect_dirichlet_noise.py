from __future__ import annotations

import sys

import numpy as np
from pydantic.dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from simple_parsing import parse

from tetris_bot.constants import NUM_ACTIONS

console = Console()


@dataclass
class ScriptArgs:
    dirichlet_alpha: float = 0.01  # Dirichlet concentration alpha for each action
    dirichlet_epsilon: float = 0.25  # Root noise mixing weight epsilon
    num_actions: int = NUM_ACTIONS  # Action space size
    num_draws: int = 15  # Number of Dirichlet samples to print
    seed: int = 1  # RNG seed for reproducible draws

    def __post_init__(self) -> None:
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError(
                f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}"
            )
        if self.num_actions <= 1:
            raise ValueError(f"num_actions must be > 1, got {self.num_actions}")
        if self.num_draws <= 0:
            raise ValueError(f"num_draws must be > 0, got {self.num_draws}")


def format_full_vector(vector: np.ndarray) -> str:
    return np.array2string(
        vector,
        separator=", ",
        max_line_width=200,
        threshold=sys.maxsize,
        precision=10,
        floatmode="maxprec",
    )


def main(args: ScriptArgs) -> None:
    rng = np.random.default_rng(args.seed)
    alpha_vector = np.full(args.num_actions, args.dirichlet_alpha, dtype=np.float64)
    uniform_prior = np.full(args.num_actions, 1.0 / args.num_actions, dtype=np.float64)
    dirichlet_stats: list[tuple[float, float, float]] = []
    mixed_stats: list[tuple[float, float, float]] = []

    print(
        "Dirichlet root-noise inspection "
        f"(alpha={args.dirichlet_alpha}, epsilon={args.dirichlet_epsilon}, "
        f"num_actions={args.num_actions}, num_draws={args.num_draws}, seed={args.seed})"
    )

    for draw_idx in range(args.num_draws):
        dirichlet_draw = rng.dirichlet(alpha_vector)
        mixed_prior = (
            1.0 - args.dirichlet_epsilon
        ) * uniform_prior + args.dirichlet_epsilon * dirichlet_draw

        print(f"\n--- Draw {draw_idx + 1}/{args.num_draws} ---")
        print(
            "Dirichlet stats: "
            f"sum={dirichlet_draw.sum():.10f}, "
            f"min={dirichlet_draw.min():.10f}, "
            f"max={dirichlet_draw.max():.10f}"
        )
        dirichlet_stats.append(
            (
                float(np.mean(dirichlet_draw)),
                float(np.min(dirichlet_draw)),
                float(np.max(dirichlet_draw)),
            )
        )
        print("dirichlet_draw:")
        print(format_full_vector(dirichlet_draw))

        print(
            "Mixed-prior stats: "
            f"sum={mixed_prior.sum():.10f}, "
            f"min={mixed_prior.min():.10f}, "
            f"max={mixed_prior.max():.10f}"
        )
        mixed_stats.append(
            (
                float(np.mean(mixed_prior)),
                float(np.min(mixed_prior)),
                float(np.max(mixed_prior)),
            )
        )
        print("mixed_prior = (1 - epsilon) * uniform + epsilon * dirichlet_draw:")
        print(format_full_vector(mixed_prior))

    dirichlet_stats_array = np.array(dirichlet_stats)
    mixed_stats_array = np.array(mixed_stats)
    table = Table(title="Dirichlet Noise Summary Stats")
    table.add_column("Draw", justify="right")
    table.add_column("Distribution")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for draw_idx in range(args.num_draws):
        dir_mean, dir_min, dir_max = dirichlet_stats[draw_idx]
        mix_mean, mix_min, mix_max = mixed_stats[draw_idx]
        draw_label = f"{draw_idx + 1}/{args.num_draws}"
        table.add_row(
            draw_label,
            "dirichlet_draw",
            f"{dir_mean:.10f}",
            f"{dir_min:.10f}",
            f"{dir_max:.10f}",
        )
        table.add_row(
            draw_label,
            "mixed_prior",
            f"{mix_mean:.10f}",
            f"{mix_min:.10f}",
            f"{mix_max:.10f}",
        )

    table.add_section()
    table.add_row(
        "avg",
        "dirichlet_draw",
        f"{np.mean(dirichlet_stats_array[:, 0]):.10f}",
        f"{np.mean(dirichlet_stats_array[:, 1]):.10f}",
        f"{np.mean(dirichlet_stats_array[:, 2]):.10f}",
    )
    table.add_row(
        "avg",
        "mixed_prior",
        f"{np.mean(mixed_stats_array[:, 0]):.10f}",
        f"{np.mean(mixed_stats_array[:, 1]):.10f}",
        f"{np.mean(mixed_stats_array[:, 2]):.10f}",
    )
    console.print()
    console.print(table)


if __name__ == "__main__":
    main(parse(ScriptArgs))
