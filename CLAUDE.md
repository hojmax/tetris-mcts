# Project Context for Agents

## Keep This File Current

Treat this document as a living guide.

- Fix outdated or incorrect info when you notice it.
- Add important constraints, workflow gotchas, or debugging tips as they are discovered.
- Prefer small, continuous updates over occasional big rewrites.

## Non-Negotiable Policies

### Feature Worktrees

- Small edits (tiny bug fixes, docs tweaks, narrowly scoped changes) can be done directly on `main`.
- Feature work or risky changes must use a dedicated worktree under:
  `/Users/axelhojmark/Desktop/tetris-mcts-worktrees/`
- Do not implement large feature work directly in:
  `/Users/axelhojmark/Desktop/tetris-mcts`
- Create isolated feature work with:
  `git worktree add -b <branch-name> /Users/axelhojmark/Desktop/tetris-mcts-worktrees/<worktree-name> <base-ref>`
- After merge, immediately remove the feature worktree and delete the branch.

### Parallel Game Generation/Evaluation

- Always parallelize game generation/evaluation in normal runs.
- Do not use single-worker loops when a parallel path exists.
- Rust evaluation entry points (`evaluate_model`, `evaluate_model_without_nn`) must use `num_workers > 1`.
- Python scripts that generate games should expose and use worker-count args, defaulting to multi-process execution.

## Project Overview

`tetris-mcts` is an AlphaZero-style Tetris system:

- Rust engine for game logic and MCTS
- PyO3 bindings for Python integration
- PyTorch model training in Python
- ONNX export for Rust inference
- Pygame + Dash tools for visualization

## Quick Commands

```bash
make install    # uv sync bootstrap (installs uv if missing)
make build      # release Rust extension (slow, optimized)
make build-dev  # debug Rust extension (fast iteration)
make play       # interactive game
make viz        # MCTS tree visualizer
make test       # Rust + Python tests
make check      # ruff + pyright + rust fixes/formatting
make train      # training entry point (requires tmux)
make profile    # performance profile runner
```

Useful extras:

```bash
make sweep-lr-model
make eval-nn-value-weight
make sweep-mcts-config
make compare-offline-network-scaling
```

## High-Level Repository Layout

```text
tetris_core/src/
├── game/       # deterministic Tetris rules and environment
├── search/     # MCTS implementation
├── inference/  # ONNX inference backends
├── runtime/    # self-play and evaluator runtime
├── replay/     # replay and NPZ persistence
└── lib.rs      # PyO3 module exports

tetris_bot/
├── ml/         # model, trainer, loss, replay buffer, checkpoints
├── scripts/    # training, ablations, inspection tools
├── visualization.py
└── constants.py
```

## Architecture and Ownership Boundaries

Data flow:

```text
Python training -> ONNX export -> Rust inference during self-play/eval
```

Ownership split:

- Rust (`tetris_core/`) owns game logic, move generation, scoring, MCTS, runtime state, inference.
- Python (`tetris_bot/`) owns training, analysis scripts, and visualization/UI.
- UI styling/color choices are Python-owned; Rust should expose state only.

## Core Concepts

### Action Space

- Fixed 735 actions: 734 placements + hold.
- `ActionSpace` maps action index <-> placement params.
- Placement caching is heavily used; hold availability is part of cache keys.

### MCTS with Chance Nodes

- `DecisionNode`: choose action.
- `ChanceNode`: random next piece from 7-bag process.
- Chance outcomes use visible queue-tail identity for consistent subtree reuse.
- Hold-swap deterministic transitions use `NO_CHANCE_OUTCOME` sentinel.

### Network Shape and Features

- Input: 280 features (`200 board + 80 aux`).
- Aux split:
  - 61 piece/game features for uncached heads path.
  - 19 board-derived stats folded into cached board embedding.
- Outputs: policy over 735 actions + scalar value.
- Architecture options:
  - `gated_fusion` (default): conv trunk + cached board embedding + aux-conditioned fusion.
  - `simple_aux_mlp`: aux-only MLP over all 80 aux features, then linear policy/value heads (board input ignored for predictions).

### Scoring Reminder

- Attack baseline: single 0, double 1, triple 2, tetris 4.
- T-spins, combo, back-to-back, perfect clear apply bonuses.

## Important Defaults (TrainingConfig)

- MCTS: `num_simulations=2000`, `c_puct=1.5`, `temperature=0.8`, `reuse_tree=true`, `max_placements=50`.
- Training: `batch_size=1024`, `learning_rate=5e-4` with decay to `1e-4`, `weight_decay=1e-4`.
- Workers/buffer: `num_workers=7`, replay ring buffer size `2_000_000`.
- Bootstrap: starts no-network, typically `bootstrap_num_simulations=4000` until first promotion.
- Candidate gate: deterministic fixed-seed eval window (default 50 games, no Dirichlet noise, `visit_sampling_epsilon=0`, fixed MCTS seed), promote only if candidate beats the stored incumbent evaluation average.
- NN value scaling: `nn_value_weight=0.01` default with promotion-based ramp and cap.
- Q normalization: tanh mode on by default (`q_scale=8.0`).

Memory note:

- CPU RAM usually becomes the bottleneck before GPU VRAM.
- Biggest levers: `buffer_size`, `num_workers`, `bootstrap_num_simulations`, replay mirror/staging settings.
- Rust per-worker caches (`PLACEMENT_CACHE_MAX_ENTRIES`, `BOARD_ANALYSIS_CACHE_MAX_ENTRIES`) scale with worker count.

## Common Workflows

### Change Game Logic

1. Edit Rust code in `tetris_core/src/game/env/` (or related modules).
2. Update Python-facing bindings in `tetris_core/src/game/env/pymethods.rs` if needed.
3. Rebuild (`make build-dev` for iteration).
4. Validate with `make test` and optionally `make play`.

### Change Network/Features

1. Edit `tetris_bot/ml/network.py`.
2. Update Rust inference encoding in `tetris_core/src/inference/mod.rs` if feature schema changes.
3. Retrain/export ONNX artifacts.

### Train / Resume

```bash
python tetris_bot/scripts/train.py --total_steps 100000
python tetris_bot/scripts/train.py --resume_dir training_runs/vN
python tetris_bot/scripts/train.py --architecture simple_aux_mlp --fc_hidden 64
```

Training runs live under `training_runs/vN/` with checkpoints and ONNX exports.

### Profile

```bash
make profile SIMS=200
make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"
make profile-samply SIMS=50
```

Set `MODEL_PROFILE=<existing-onnx-path>` when the Makefile default model path is missing.

## Testing

```bash
make test
cd tetris_core && PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python" cargo test
```

In feature worktrees, use the primary checkout virtualenv path for `PYO3_PYTHON`.

## Data/Artifacts Notes

- `training_data.npz` is for resume/inspection and includes policy/value targets plus per-state diagnostics.
- Resumed runs create a new `vN` directory and initialize from the source run checkpoint.
- Promotion state (incumbent model + `nn_value_weight`) is treated as an atomic runtime state.

## Coding Rules (Condensed)

- Fix problems you touch; do not leave known errors behind.
- Prefer one clean implementation path; avoid backward-compatibility fallbacks for internal formats.
- Remove dead code when replacing functionality.
- Keep imports at top level (except explicit optional/cycle cases).
- Keep functions focused and short; extract reused logic.
- Use modern typing syntax (`A | B`, `list[T]`) and `from __future__ import annotations` when useful.
- Use `pathlib.Path` and `PROJECT_ROOT` (`tetris_bot.constants`) for project-relative paths.
- Use `simple_parsing` dataclass args in scripts; put defaults directly in dataclass fields.
- Use `structlog` for logging; do not use `print()` as logging.
- Prefer early returns and readable positive conditions.
- Comment sparingly; explain why, not what.
- Fail fast on impossible internal states; only be defensive for truly external/user-provided data.

## When in Doubt

- Keep Rust focused on engine/search/runtime performance.
- Keep Python focused on training/analysis/visualization.
- Parallelize game workloads.
- Update this file whenever reality changes.
