# Project Context for Agents

## Keep This File Current

Treat this document as a living guide.

- Fix outdated or incorrect info when you notice it.
- Add important constraints, workflow gotchas, or debugging tips as they are discovered.
- Prefer small, continuous updates over occasional big rewrites.

## Memory Workflow

- At the start of every session, read `memories/MEMORY.md` before doing substantive work. If it is missing, create it first.
- Treat `memories/MEMORY.md` as the short-term operational memory log for repo-specific learnings and user feedback.
- Write to `memories/MEMORY.md` liberally whenever you learn something useful: corrected command usage, debugging realizations, unintuitive behavior, hard-to-find facts, user preferences, workflow gotchas, or other details likely to save time later.
- Prefer concise entries with the final takeaway, not a long play-by-play.
- When `memories/MEMORY.md` reaches 200 lines, compact it. Group related stable notes into themed Markdown files under `memories/`.
- When compacting, append to an existing themed memory file if one already fits the topic; otherwise create a new file with a descriptive name.
- After compacting, keep `memories/MEMORY.md` as the active short-term log with a reduced set of current notes and pointers to the long-term themed files.

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
- Rust parallel evaluation uses dynamic seed claiming to reduce stragglers in fixed-seed sweeps; benchmark sweeps should prefer larger `num_games` than `num_workers` to reduce seed-length variance noise.
- GameGenerator workers commit completed games in small batches (`GAME_COMMIT_BATCH_SIZE` in `runtime/game_generator/runtime.rs`) to reduce lock contention at high worker counts.
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
make install    # uv sync + Rust bootstrap + PATH persist + debug extension build
make build      # release Rust extension (slow, optimized; native CPU + thin LTO by default)
make build-ort  # release build with ONNX Runtime support (`nn-ort` feature)
make build-dev  # debug Rust extension (fast iteration)
make play       # interactive game
make viz        # MCTS tree visualizer
make viz-policy-grid # policy-grid visualizer for structured placement-head inspection
make test       # Rust + Python tests
make check      # ruff + pyright + rust fixes/formatting
make train      # training entry point (requires tmux; auto-loads machine optimize cache)
make warm-start # create a new warm-started run dir from another run's training_data.npz
make compare-offline-spatial-policy # compare current gated fusion vs 4x20x10 spatial policy decoder offline
make profile    # performance profile runner
make optimize   # auto-tune build/backend/workers for this machine (with cache)
```

`make install` performs a best-effort Linux system dependency bootstrap for ORT builds (`pkg-config` + OpenSSL headers) and maturin builds (`patchelf`); disable with `AUTO_INSTALL_SYSTEM_DEPS=0`.
`make viz` accepts pass-through args via `VIZ_ARGS`, e.g. `make viz VIZ_ARGS="--state_preset tetris_bot/scripts/inspection/viz_state_presets/training_data1_game721_move32.json"`; generate presets from NPZ with `python tetris_bot/scripts/inspection/extract_viz_state_preset.py`. By default it loads `checkpoints/incumbent.onnx` from the selected run.
`make viz-policy-grid` launches the browser visualizer for the normalized `20x10x4` placement scheme; pass flags such as `--port` via `POLICY_GRID_ARGS`.
Candidate-gate evaluation now also saves the worst full-game tree playback to `training_runs/vN/analysis/eval_trees/`; reopen it with `make viz VIZ_ARGS="--saved_playback training_runs/vN/analysis/eval_trees/latest_worst_candidate_eval_tree.json"`.
If a saved full-game playback is too large to load comfortably, extract one exact step into a smaller playback with `./.venv/bin/python tetris_bot/scripts/inspection/extract_saved_playback_step.py --saved_playback <full-playback.json> --step_index <n>` and open that output with `make viz VIZ_ARGS="--saved_playback <step-playback.json>"`.
For inspection/export defaults, prefer the effective incumbent search state from the saved checkpoint (`latest.pt`) over bare `config.json`; older runs can have promoted `nn_value_weight` / penalty settings that differ from the static config file.
The visualizer UI includes a `Play Full Game` control that rolls out the rest of the game from the current state using tree reuse, stitches the per-move search trees together with `reuse` edges, and highlights the chosen action/chance path.

Useful extras:

```bash
make sweep-lr-model
make eval-nn-value-weight
make sweep-mcts-config
make compare-offline-network-scaling
make compare-offline-spatial-policy
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
- Candidate gate: deterministic fixed-seed eval window (default 20 games, no Dirichlet noise, `visit_sampling_epsilon=0`, fixed MCTS seed), promote only if candidate beats the stored incumbent evaluation average. Fixed-seed eval trajectories are not added to the replay buffer.
- NN value scaling: `nn_value_weight=0.01` default with promotion-based ramp and cap.
- Q normalization: min-max mode on by default (`use_tanh_q_normalization=false`, so `q_scale=None` in rollout config).

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
python tetris_bot/scripts/train.py --resume_wandb entity/project/run_id
python tetris_bot/scripts/inspection/download_wandb_training_data.py --reference entity/project/run_id --run_dir training_runs/vN --overwrite true
python tetris_bot/scripts/train.py --architecture simple_aux_mlp --fc_hidden 64
python tetris_bot/scripts/warm_start.py --source_run_dir training_runs/v3
```

Training runs live under `training_runs/vN/` with checkpoints and ONNX exports.
`--resume_wandb` accepts either a run reference (`entity/project/run_id`, defaults to the run's `tetris-model-<run_id>:latest` artifact) or a direct artifact reference (`entity/project/tetris-model-<run_id>:alias`).
`warm_start.py` creates the next sibling `training_runs/vN` by default, builds the network from the source `config.json`, trains offline on the source `training_data.npz`, copies that NPZ into the new run, saves `latest`/`incumbent`/`parallel` model bundles plus `checkpoint_0.pt`, forces the saved incumbent startup state to `nn_value_weight=1.0` with zero search penalties, writes `analysis/warm_start_summary.json` with the offline-loss history and fixed-seed eval result, and logs the offline train/eval run to the `tetris-mcts-offline` W&B project by default. Offline warm-start training now runs in `epochs_per_round` chunks (default `2.0`), evaluates after each chunk, stops after `early_stopping_patience=20` consecutive rounds without improving the fixed selection metric `val_policy_loss + val_value_loss / 4`, keeps the best checkpointed weights, saves `checkpoints/warm_start_offline_latest.pt` for true offline continuation, and defaults final eval workers from the machine optimize cache (`TETRIS_OPT_NUM_WORKERS`) when available. Use `--output_run_dir` to override the destination and `--eval_num_simulations` / `--eval_max_placements` for quicker smoke checks without changing the saved run config. Set `--resume_from_source_offline_state true` to continue from another warm-start run's saved offline weights and optimizer state; older warm-start runs created before this checkpoint existed cannot use that flag.
During shutdown, the first `Ctrl+C` requests graceful trainer/generator stop; additional `Ctrl+C` presses are deferred until the trainer finishes generator shutdown, the final replay snapshot/checkpoint save, and the WandB artifact upload.
Checkpoint resume now persists incumbent search penalties (`incumbent_death_penalty`, `incumbent_overhang_penalty_weight`) alongside `incumbent_nn_value_weight`; older checkpoints without those fields infer zero penalties when `nn_value_weight` is already at cap.

### Profile

```bash
make profile SIMS=200
make profile SIMS=1000 WORKERS_PROFILE=6
make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"
make profile-samply SIMS=50
make optimize
```

Set `MODEL_PROFILE=<existing-onnx-path>` when the Makefile default model path is missing.
Set `TETRIS_NN_BACKEND=ort` to profile ONNX Runtime instead of default `tract`.
`make optimize` is fast by default; for an exhaustive sweep: `make optimize OPT_BACKEND_STRATEGY=exhaustive OPT_WORKER_SEARCH=grid OPT_GAMES=40 OPT_SIMS=500`.
Results written to `benchmarks/profiles/optimize_cache/<machine_fingerprint>.env`; `make train` computes the current machine fingerprint and loads the matching env (runs optimize first if missing).
Do NOT use `maturin develop` — use `maturin build --out tetris_core/dist` then `$(PYTHON) -m pip install` (respects `.venv` `sys.prefix` correctly).

## Testing

```bash
make test
cd tetris_core && PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python" cargo test
```

In feature worktrees, use the primary checkout virtualenv path for `PYO3_PYTHON`.
If you see `ModuleNotFoundError: No module named 'tetris_core.tetris_core'`, the native extension is missing for the active interpreter; rebuild with `make build-dev` (or `make build`) in that same `.venv`.

## Data/Artifacts Notes

- `training_data.npz` is for resume/inspection and includes policy/value targets plus per-state diagnostics.
- Large replay snapshots can exceed 4 GiB in `policy_targets.npy`; NPZ writing must keep ZIP64 enabled and snapshot writes should be atomic (`.tmp` then replace).
- Final WandB model artifact upload includes `training_data.npz` whenever the file exists; no extra Python-side NPZ integrity gate is applied at upload time.
- Resumed runs create a new `vN` directory and initialize from the source run checkpoint.
- Promotion state (incumbent model + `nn_value_weight`) is treated as an atomic runtime state.

## Coding Rules (Condensed)

- Fix problems you touch; do not leave known errors behind.
- Prefer one clean implementation path; avoid backward-compatibility fallbacks for internal formats.
- Remove dead code when replacing functionality.
- Keep imports at top level (except explicit optional/cycle cases).
- Keep functions focused and short; extract reused logic.
- Keep the Rust PyO3 API minimal; do not add dict/string/repr convenience wrappers unless they have a concrete in-repo caller.
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
- Read `memories/MEMORY.md` first and record durable learnings there.
- Update this file whenever reality changes.
