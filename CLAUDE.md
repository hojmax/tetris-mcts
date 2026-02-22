# Project Context for Agents

## Keep This File Current (All Agents)

Treat this document as a living resource, not static documentation.

- If you spot inaccurate, outdated, or misleading information while working, update it immediately in the same change.
- If you discover new critical context (architecture constraints, workflows, gotchas, debugging tips, command conventions), add it here.
- Make small, iterative improvements continuously so future agents inherit accurate context.
- When handing off work, call out any updates you made here and suggest follow-up updates if context is still incomplete.

## Feature Worktree Policy (All Agents)

- Small changes can be done directly in the primary checkout on `main` (for example: tiny bug fixes, docs tweaks, or narrowly scoped edits that do not need isolation).
- For feature work or larger/riskier code changes, create a dedicated git worktree under:
  `/Users/axelhojmark/Desktop/tetris-mcts-worktrees/`
- Do not implement feature changes directly in the primary checkout at:
  `/Users/axelhojmark/Desktop/tetris-mcts`
- Use a feature branch with `git worktree add -b <branch-name> /Users/axelhojmark/Desktop/tetris-mcts-worktrees/<worktree-name> <base-ref>`
- Keep each feature isolated to its own worktree; merge to `main` only after validation.
- After merging a feature branch into `main`, always clean up the feature worktree and branch immediately (for example: `git -C /Users/axelhojmark/Desktop/tetris-mcts worktree remove /Users/axelhojmark/Desktop/tetris-mcts-worktrees/<worktree-name>` and `git -C /Users/axelhojmark/Desktop/tetris-mcts branch -d <branch-name>`).

## Game Parallelism Policy (All Agents)

- **ALWAYS PARALLELIZE GENERATING AND EVALUATING GAMES (MANDATORY, NO EXCEPTIONS IN NORMAL RUNS).**
- Do not run single-worker game generation/evaluation loops when a parallel path exists.
- For Rust evaluation entry points (`evaluate_model`, `evaluate_model_without_nn`), set `num_workers > 1` (typically near available CPU cores).
- For Python inspection/benchmark scripts that generate games, expose and use a worker-count argument so multi-process execution is the default behavior.

## Project Overview

This is **tetris-mcts**, an AlphaZero-style reinforcement learning system for Tetris that combines:

- A high-performance **Rust game engine** with PyO3 Python bindings
- **Monte Carlo Tree Search (MCTS)** with neural network guidance
- **PyTorch CNN** for policy and value prediction
- **Pygame visualization** for interactive play

## Quick Commands

```bash
make build      # Compile Rust with maturin (release mode, slow but optimized)
make build-dev  # Fast debug build (~10x faster, for development only)
make play       # Run interactive Tetris game
make viz        # Run MCTS tree visualizer (Dash app at localhost:8050)
make test       # Run Rust tests + Python tests
make check      # Run ruff + pyright linting
make sweep-lr-model  # Run W&B sweep for learning rate + model size
make eval-nn-value-weight  # Evaluate fixed network at multiple nn_value_weight values
make compare-offline-network-scaling  # Compare default vs scaled network variants offline
make sweep-mcts-config  # Sweep an MCTS config param (q_scale, c_puct, etc.)
make rebuild    # Force clean rebuild (slow, only when needed)
```

**Development tip:** Use `make build-dev` for fast iteration. Only use `make build` or `make rebuild` for benchmarking or production.

**Speed up Rust compilation** (optional):

```bash
# Install sccache (caches compiled crates)
cargo install sccache

# Enable in shell profile (~/.zshrc or ~/.bashrc)
export RUSTC_WRAPPER=sccache

# Check cache stats
sccache --show-stats
```

This caches compiled dependencies across projects, making rebuilds much faster.

### Training

```bash
# Start new training run (creates training_runs/v0/)
python tetris_bot/scripts/train.py --total_steps 100000

# With custom hyperparameters
python tetris_bot/scripts/train.py \
    --total_steps 500000 \
    --num_simulations 800 \
    --learning_rate 0.0005

# Resume from checkpoint (creates a new versioned run initialized from v0/latest.pt)
python tetris_bot/scripts/train.py --resume_dir training_runs/v0
```

### Offline Architecture Comparison

```bash
# Compare legacy concat+FC baseline vs current gated-fusion model on a fixed NPZ snapshot.
# Logs all per-step losses and final comparison to WandB.
# Gated model is auto-matched to baseline by parameter count and forward FLOPs
# within configurable relative tolerances (cache-weighted with hit-rate default 0.96).
# Matching search may choose 0+ fusion residual blocks depending on fairness constraints.
# Default uses all examples in the NPZ (`max_examples=0`) and 400 training steps.
# Throughput and system metrics are logged (batches/sec, examples/sec, eval throughput,
# step time, grad norm, and CUDA memory where available).
python scripts/ablations/compare_offline_architectures.py \
    --data_path training_runs/v32/training_data.npz

# Optional: preload selected examples to GPU once to reduce per-batch transfer overhead
python scripts/ablations/compare_offline_architectures.py \
    --data_path training_runs/v32/training_data.npz \
    --preload_to_gpu true
```

### Offline Network Scaling Comparison

```bash
# Compare three gated-fusion variants on a fixed NPZ snapshot:
# 1) default model
# 2) 2x board trunk (cached conv path)
# 3) 2x post-fusion hidden size
# Logs losses, throughput, parameter counts, and cache-weighted FLOPs to WandB.
python scripts/ablations/compare_offline_network_scaling.py \
    --data_path training_runs/v32/training_data.npz

# Optional: tune multipliers (defaults are both 2)
python scripts/ablations/compare_offline_network_scaling.py \
    --data_path training_runs/v32/training_data.npz \
    --board_trunk_multiplier 2 \
    --post_fusion_multiplier 2
```

### Offline Feature Ablation (State Features)

```bash
# Compare gated-fusion variants with extra board-state diagnostics:
# no-extra, all-extra, and all-minus-one ablations.
# Logs per-step metrics for each variant, per-step comparison curves under
# `comparison/curves/*`, a main line-series chart under
# `comparison/charts/eval_val_total_loss_main`, plus Plotly overlay charts.
# Requires NPZ snapshots that include:
# column_heights, max_column_heights, min_column_heights,
# row_fill_counts, total_blocks, bumpiness (and optionally move_numbers).
python tetris_bot/scripts/ablations/compare_offline_feature_ablation.py \
    --data_path training_runs/v32/training_data.npz

# Optional: include move_numbers as an additional ablation group
python scripts/ablations/compare_offline_feature_ablation.py \
    --data_path training_runs/v32/training_data.npz \
    --include_move_number_feature true
```

### MCTS Config Sweep

```bash
# Sweep q_scale (tanh divisor) over multiple values using a trained model.
# Runs MCTS evaluation games in Rust for each value and outputs JSON + PNG plot.
python scripts/ablations/sweep_mcts_config.py \
    --run_dir training_runs/v32 \
    --sweep_param q_scale \
    --sweep_values '[2, 4, 8, 16, 32]'

# Sweep any other float MCTSConfig param (e.g. c_puct, nn_value_weight)
python scripts/ablations/sweep_mcts_config.py \
    --run_dir training_runs/v32 \
    --sweep_param c_puct \
    --sweep_values '[0.5, 1.0, 1.5, 2.0, 3.0]' \
    --num_games 100

# Override base MCTS config values that aren't being swept
python scripts/ablations/sweep_mcts_config.py \
    --run_dir training_runs/v32 \
    --sweep_param q_scale \
    --sweep_values '[4, 8, 16]' \
    --nn_value_weight 0.05 \
    --num_simulations 500
```

### Row Fill Zero-Rate Analysis

```bash
# Report per-row zero rates for row_fill_counts in a replay snapshot.
python tetris_bot/scripts/inspection/row_fill_zero_rates.py \
    --data_path training_runs/vN/training_data.npz

# Optional: adjust tolerance used for zero checks.
python tetris_bot/scripts/inspection/row_fill_zero_rates.py \
    --data_path training_runs/vN/training_data.npz \
    --epsilon 1e-8
```

### Bootstrap Tree Reuse Analysis

```bash
# Measure tree reuse metrics (including tree_reuse_carry_fraction) in bootstrap/no-NN mode.
# Defaults: 10 games, 4000 simulations, 50 max placements, add_noise=true,
# and auto-parallel workers (all available CPU cores, capped by num_games).
# tree_reuse_hits/misses exclude terminal and max-placement transitions
# where no next search step exists.
# Also reports traversal-outcome fractions across all simulations:
# traversal_expansion_fraction, traversal_terminal_fraction, traversal_horizon_fraction.
python tetris_bot/scripts/inspection/measure_bootstrap_tree_reuse.py

# Optional: override run size/worker count/output path.
python tetris_bot/scripts/inspection/measure_bootstrap_tree_reuse.py \
    --num_games 20 \
    --simulations 4000 \
    --max_placements 50 \
    --num_workers 8 \
    --output_json benchmarks/bootstrap_tree_reuse_run2.json
```


### Performance Profiling

**Timing Benchmarks** (saves results to JSONL):

```bash
make profile              # Uses Makefile defaults (MODEL_PROFILE=training_runs/v6/checkpoints/latest.onnx, SIMS=1000)
make profile SIMS=50      # Faster profiling with fewer simulations
make profile SIMS=200     # More accurate with more simulations
make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"  # No-network bootstrap mode
make profile MODEL_PROFILE=<path-to-existing-onnx>  # Override model path explicitly
```

If the default `MODEL_PROFILE` path does not exist in your local checkout, pass an explicit
existing ONNX path (for example `training_runs/v32/checkpoints/latest.onnx`).

For optimization validation on a busy desktop machine, run baseline and candidate back-to-back
with identical flags and a fixed `--mcts_seed` (for example `123`) to reduce run-to-run variance.

Results saved to `benchmarks/profile_results.jsonl` with timing data for comparison across runs.
If profiling ends with zero completed games (for example model load/inference failure), `profile_games.py` now logs a warning and reports zero throughput instead of crashing.

**Inference backend A/B workflow** (tract vs ONNX Runtime CPU):

```bash
# Build baseline tract backend
python -m maturin develop --release --manifest-path tetris_core/Cargo.toml

# Build tract with tract-linalg multithread-mm enabled
python -m maturin develop --release --manifest-path tetris_core/Cargo.toml \
  --features "extension-module,nn-tract-multithread-mm"

# Build with optional ONNX Runtime backend support
python -m maturin develop --release --manifest-path tetris_core/Cargo.toml \
  --features "extension-module,nn-ort"

# Run identical profiles and switch backend via env var
TETRIS_NN_BACKEND=tract python tetris_bot/scripts/inspection/profile_games.py \
  --model_path training_runs/v45/checkpoints/latest.onnx --num_games 5 --simulations 200 --seed_start 42 --mcts_seed 123
TETRIS_NN_BACKEND=ort python tetris_bot/scripts/inspection/profile_games.py \
  --model_path training_runs/v45/checkpoints/latest.onnx --num_games 5 --simulations 200 --seed_start 42 --mcts_seed 123
```

`TETRIS_NN_BACKEND=ort` requires a build with Cargo feature `nn-ort`; otherwise model load fails with a clear error.

**Interactive Profiling** (requires [samply](https://github.com/mstange/samply)):

```bash
# Install samply (one-time)
cargo install samply

# Profile and view flamegraph in browser
make profile-samply SIMS=50

# Or run directly
samply record python scripts/inspection/profile_games.py --num_games 3 --simulations 50
```

Opens interactive flamegraph viewer showing ALL function calls automatically. Best for finding bottlenecks during development.

**Function-level profiling workflow** (CPU sampled, not per-call stopwatch timing):

```bash
# 1) MCTS-only hotspot profile (no NN inference)
samply record python tetris_bot/scripts/inspection/profile_games.py \
  --use_dummy_network true \
  --num_games 3 \
  --simulations 300 \
  --mcts_seed 123

# 2) Full MCTS + ONNX profile (includes NN inference)
samply record python tetris_bot/scripts/inspection/profile_games.py \
  --model_path training_runs/v32/checkpoints/latest.onnx \
  --num_games 3 \
  --simulations 300 \
  --mcts_seed 123
```

Use the flamegraph search to inspect specific functions/namespaces (for example `tetris_core::moves::find_all_placements`, `tetris_core::nn::TetrisNN`, `tract_onnx::`, `tract_linalg::`, `ort::`).
If your summary only groups `tetris_core::*`, NN compute can appear missing because much of ONNX runtime time is attributed to backend symbols (`tract_*` or `ort::*`) rather than `tetris_core::*`.

**macOS native profiling** (Instruments):

```bash
instruments -t "Time Profiler" python scripts/inspection/profile_games.py --num_games 3
```

## Architecture

```
Python (PyTorch training)
    ↓
Rust Core (tetris_core/)     ← Fast game logic + MCTS
    ↓
ONNX Export                  ← Model exported for Rust inference
```

**Key design**: Game logic and MCTS run entirely in Rust for speed. Neural network training and all visualization/presentation concerns run in Python. Models are exported as ONNX for Rust inference.

**Ownership boundary (strict):**

- Rust (`tetris_core/`) = environment logic, move generation, scoring, MCTS, inference/runtime state.
- Python (`tetris_bot/`) = training, evaluation UX, and visualization/rendering.
- Color palettes and UI styling are Python-owned. Rust should expose piece identity/state (for example `piece_type`), not display colors.

## Project Structure

```
tetris_core/src/             # Rust game engine
├── lib.rs                   # PyO3 module exports
├── constants.rs             # Board size (10x20), piece indices
├── piece.rs                 # Tetromino shapes and rotations
├── kicks.rs                 # SRS wall kick data
├── scoring.rs               # Attack calculation (lines, T-spins, combos)
├── moves.rs                 # Move generation/pathfinding
├── env/                     # TetrisEnv game state
│   ├── state.rs             # Main TetrisEnv struct
│   ├── board.rs             # Board collision detection
│   ├── movement.rs          # Piece movement (left/right/down)
│   ├── placement.rs         # Piece locking
│   ├── piece_management.rs  # Queue, hold, 7-bag spawning
│   ├── clearing.rs          # Line clear mechanics
│   ├── lock_delay.rs        # Lock delay timer
│   ├── global_cache.rs      # Thread-local placement & board analysis caches
│   └── pymethods.rs         # Python API exports
├── mcts/                    # Monte Carlo Tree Search
│   ├── agent.rs             # MCTSAgent PyO3 interface
│   ├── search.rs            # Core MCTS algorithm (simulate, expand, backup)
│   ├── export.rs            # Tree visualization export
│   ├── nodes.rs             # DecisionNode & ChanceNode
│   ├── config.rs            # MCTSConfig hyperparameters
│   ├── action_space.rs      # 735-action mapping (734 placements + hold)
│   ├── results.rs           # TrainingExample, GameResult, GameStats
│   └── utils.rs             # PUCT scoring, Dirichlet noise
├── nn.rs                    # ONNX inference (tract default, optional ONNX Runtime backend)
└── generator/               # Background game generation
    ├── game_generator/
    │   ├── mod.rs
    │   ├── py_api.rs
    │   ├── runtime.rs
    │   ├── shared.rs
    │   └── tests.rs
    ├── evaluation.rs
    ├── npz.rs
    └── types.rs             # GameReplay, ReplayMove types

tetris_bot/                 # Python package
├── constants.py             # PROJECT_ROOT, board/action constants
├── visualization.py         # Board rendering + replay visualization
├── run_setup.py             # Run directory management
├── ml/
│   ├── config.py            # TrainingConfig and all hyperparameter dataclasses
│   ├── network.py           # TetrisNet (PyTorch CNN)
│   ├── trainer.py           # Trainer class, training loop
│   ├── loss.py              # Loss functions and metrics
│   ├── weights.py           # Checkpoint/ONNX export
│   ├── aux_features.py      # Auxiliary feature encoding
│   ├── replay_buffer.py     # Replay buffer management and sampling
│   ├── game_metrics.py      # Per-game metric tracking and aggregation
│   └── artifacts.py         # WandB artifact management
└── scripts/
    ├── train.py                    # Main training entry point
    ├── tetris_game.py              # Interactive Pygame game
    ├── ablations/                 # Network ablation/architecture experiments
    │   ├── average_value_target.py
    │   ├── benchmark_batch_chance.py
    │   ├── benchmark_batch_size.py
    │   ├── benchmark_network_size.py
    │   ├── benchmark_tree_reuse_dummy.py
    │   ├── check_nn_value.py
    │   ├── compare_offline_architectures.py
    │   ├── compare_offline_conv_depth.py
    │   ├── compare_offline_feature_ablation.py
    │   ├── compare_offline_network_scaling.py
    │   ├── evaluate_nn_value_weight_sweep.py
    │   ├── sweep_mcts_config.py
    │   └── wandb_sweep_lr_model_size.py
    └── inspection/                 # Data/MCTS inspection and debugging tools
        ├── analyze_training_data.py
        ├── audit_mcts_tree.py
        ├── buffer_viewer.py
        ├── count_reachable_states.py
        ├── inspect_dirichlet_noise.py
        ├── inspect_onnx_model.py
        ├── inspect_training_data.py
        ├── measure_bootstrap_tree_reuse.py
        ├── mcts_visualizer.py
        ├── profile_games.py
        ├── render_buffer_frames.py
        ├── replay_viewer.py
        ├── row_fill_zero_rates.py
        └── value_predictor.py
```

## Key Concepts

### Action Space (735 actions)

All valid (x, y, rotation) placements are enumerated. The `ActionSpace` struct maps between action indices and placements.

Placement metadata includes `last_move_was_rotation` and `last_kick_index` for T-spin scoring when executing action indices directly. If multiple shortest input paths reach the same placement, move generation prefers paths whose last move is **not** a rotation to avoid accidental T-spin attribution from arbitrary path tie breaks.
Per-state placement caching now also stores a precomputed sorted list of placement action indices, so valid-action lookup no longer rescans all 735 slots on cache hits. Global placement-cache keys include hold availability to avoid stale hold action masks across otherwise identical board/current-piece states. For MCTS/runtime paths, cache construction now uses a lightweight placement-params generator (`find_all_placement_params`) that skips move-sequence reconstruction; full `Placement.moves` sequences are generated on demand for Python-facing placement inspection (`get_possible_placements`). Move-generation hard-drop (`Board::get_drop_y`) now uses precomputed per-column occupancy bitmasks (with exact-step fallback) to avoid repeated collision probes in BFS.

### MCTS with Chance Nodes

Unlike standard AlphaZero, Tetris has stochastic piece spawning:

- **DecisionNode**: Player chooses action (move/rotate/drop)
- **ChanceNode**: Random piece spawns from 7-bag

### Neural Network (TetrisNet)

- **Input**: 280 features (200 board cells + 80 auxiliary). Aux is split into 61 piece/game features (current piece, hold, queue, placement count, combo, back-to-back, hidden-piece distribution) sent to the uncached heads model, and 19 board-derived stats (column heights, max column height, bottom-4 row fill counts, total blocks, bumpiness, holes, overhang fields) folded into the cached board embedding. Training data packs all 80 features together; the model splits internally.
- **Architecture**: Conv2d(1→16) + ResBlock(16) + stride-2 Conv(16→32) + board projection (conv features + board stats → cached embedding) + aux-conditioned gated fusion (61-dim piece/game features) + optional fusion residual blocks + policy/value heads
- **Output**: Policy probabilities over 735 actions (734 placements + hold), value (trained on raw cumulative-attack `value_targets`)

### 7-Bag Randomizer

Pieces spawn in random order, 7 at a time (no repeats within a bag). The queue shows next 5 pieces.

### Scoring System

- Single: 0 attack, Double: 1, Triple: 2, Tetris: 4
- T-spin bonuses, combo multipliers, back-to-back bonus, perfect clear (10)

## Important Types

### Rust (exported to Python)

- `TetrisEnv` - Game state (board, piece, queue, hold, attack, lines, game_over)
- `Piece` - Tetromino (piece_type, x, y, rotation)
- `MCTSAgent` - MCTS search coordinator
- `MCTSConfig` - Search hyperparameters (num_simulations, c_puct, temperature, etc.)
- `EvalResult` - Aggregated fixed-seed evaluation metrics (`avg_attack`, `game_results`, and `avg_tree_nodes` for mean tree size in nodes)
- `TrainingExample` - State + MCTS policy target + value target (`value`, raw cumulative attack) + saved board diagnostics (`column_heights`, `max_column_height`, `row_fill_counts` for bottom 4 rows, `total_blocks`, `bumpiness`, `holes`, `overhang_fields`, where `holes`/`overhang_fields` are normalized from the current state board)
- `GameGenerator` - Background self-play worker

### Python

- `TetrisNet` - PyTorch neural network
- `Trainer` - Training loop manager
- `WeightManager` - Checkpoint and ONNX export

## Code Patterns

### Default Hyperparameters

From `config.py` TrainingConfig defaults:

- **MCTS**: 2000 simulations, c_puct=1.5, temperature=0.8, reuse_tree=true, max_placements=50, death_penalty=5.0, overhang_penalty_weight=5.0
- **Training**: batch_size=1024, lr=0.0005, linear schedule to 0.0001 over 200k steps (then constant), weight_decay=1e-4, use_torch_compile=true
- **Value Loss**: `use_huber_value_loss=false` by default (MSE for value head; set true for Huber)
- **Architecture**: Conv(1→16) + 1 ResBlock(16) + stride-2 Conv(16→32), gated-fusion hidden size 48, 735 policy outputs, 1 value output
- **Buffer**: 2M examples (ring buffer), 7 parallel workers, staged sampling with `prefetch_batches=1` (one Rust sample call stages `batch_size * prefetch_batches` examples), and staged queue target `staged_batch_cache_batches=1` (train-sized batches kept resident on host/device queue before being consumed); `pin_memory_batches=true` enables pinned-host transfer on CUDA. Full replay mirroring is enabled by default on accelerator training (`mirror_replay_on_accelerator=true`): snapshot replay to device once, then incrementally append replay deltas every `replay_mirror_refresh_seconds` in chunks of `replay_mirror_delta_chunk_examples`.
- **Memory gotcha (Linux OOM killer)**: host RAM can OOM before GPU VRAM is full (for example `nvtop` looks fine) because self-play state lives in CPU memory. The biggest CPU-RAM levers are `buffer_size`, `num_workers`, `bootstrap_num_simulations`, and replay staging/mirror chunk sizes. `pin_memory_batches` usually contributes less than those, but can still add transfer-buffer overhead.
- **Cache-cap gotcha**: Rust per-worker global caches can dominate RAM. `PLACEMENT_CACHE_MAX_ENTRIES` and `BOARD_ANALYSIS_CACHE_MAX_ENTRIES` are applied per thread-local worker cache (`tetris_core/src/env/global_cache.rs`), so raising them dramatically scales memory with `num_workers`.
- **Exploration**: Dirichlet alpha=0.02, epsilon=0.25, visit-sampling epsilon=0.0
- **NN Value Scaling**: `nn_value_weight=0.01` by default. Promotion ramp is event-driven on accepted candidates with multiplicative targets and additive updates: `delta = min(current * (nn_value_weight_promotion_multiplier - 1.0), nn_value_weight_promotion_max_delta)` then `next = min(nn_value_weight_cap, current + delta)`. Defaults: multiplier `1.4` (adds 40%), max delta `0.10`, cap `1.0`.
- **Q Normalization**: `use_tanh_q_normalization=true` by default; when true, NN-guided MCTS uses `tanh(Q / q_scale)` (default `q_scale=8.0`) for the Q term. When false, uses sibling min-max Q normalization even in NN mode. Bootstrap (no-NN) mode always uses min-max regardless of this setting.
- **Wall-Clock Intervals**: training cadence is time-based (not step-based): `log_interval_seconds=10`, `model_sync_interval_seconds=300`, `checkpoint_interval_seconds=10800`; replay snapshots use `save_interval_seconds=3600` (`0` disables periodic snapshot saves).
- **Training-loop logging defaults**: full scalar train-step metrics are collected every `train_step_metrics_interval=16` steps, extra diagnostics (`compute_metrics` forward pass for policy entropy/accuracy) are enabled by default with `compute_extra_train_metrics_on_log=true` (overhead tracked as `timing/extra_metrics_ms`), and individual per-game rows are logged by default (`log_individual_games_to_wandb=true`). Aggregated per-tick replay summaries under `replay/completed_games_*` on `trainer_step` are always logged.
- **Model Promotion Gate**: candidate window=50 games on fixed seeds `0..N`, evaluator noise enabled by default; candidate evaluation carries an explicit `candidate_nn_value_weight`, and promotion atomically updates `(incumbent model, incumbent nn_value_weight)` together. Candidate eval events include per-game results and best/worst game replays for trajectory GIF rendering.
- **Bootstrap Mode**: starts without NN, uses 4000 simulations until first promoted model

Override via CLI: `--num_simulations 800 --learning_rate 0.0005`

Temperature behavior:

- `temperature` shapes the MCTS visit-count policy target used for training.
- In self-play (including candidate evaluation), action execution samples from the visit policy with probability `visit_sampling_epsilon` and otherwise uses argmax.

### Loss Function

```python
policy_loss = -sum(target_policy * log(masked_policy))  # Cross-entropy
value_loss = MSE(predicted_value, target_value)  # default
# or Huber when use_huber_value_loss = true
total_loss = policy_loss + value_loss
```

### Move Masking

Invalid actions get logits set to -inf before softmax, ensuring 0 probability.
Dirichlet root noise is mixed over `DecisionNode.action_priors` (valid actions only), not all 735 actions.

### Self-Play Data Generation

Training uses parallel Rust game generation via `GameGenerator`:

1. Multiple worker threads (default: 7) run MCTS games in parallel
2. One dedicated evaluator worker tests queued candidate ONNX models over a fixed game window (default 50 games on seeds `0..N`)
3. Candidate evaluation games use fixed seeds (default `0..model_promotion_eval_games`) for consistent benchmarking; each eval also records per-game results and best/worst game replays
4. Candidates are compared against the previous promotion winner's eval avg attack (from the same fixed seeds); if better, evaluator commits candidate games then promotes the model globally
5. Promotion is atomic for `(model, nn_value_weight)`: evaluator runs candidate games with the queued candidate weight, and if promoted, workers switch to that exact weight with the model
6. If candidate is worse, evaluator discards candidate games and keeps incumbent
7. If multiple candidates queue while evaluator is busy, only the newest pending candidate is kept
8. Before first promotion (default), workers run no-network MCTS (uniform policy prior + zero value) with separate simulation count
9. Training examples from accepted games are stored in a shared in-memory ring buffer
10. Python sampling has two modes:
   - Default staged mode: `generator.sample_batch(batch_size * prefetch_batches, max_placements)`, then move staged tensors once to the training device, split into train-sized batches, and keep a queued cache up to `staged_batch_cache_batches` before consuming.
   - Full mirror mode (CUDA/MPS only, `mirror_replay_on_accelerator=true`): `generator.replay_buffer_snapshot(max_placements)` initializes a full device mirror, then `generator.replay_buffer_delta(from_index, max_examples, max_placements)` incrementally appends new examples and drops evicted prefix rows to match FIFO windowing. Rust now snapshots replay rows and logical index bounds atomically from a shared replay state (`SharedBufferState`), so Python deltas stay aligned with FIFO index space under concurrent generation.
   Both modes use `(boards, aux, policy_targets, value_targets, overhang_fields, action_masks)` tensors; periodic NPZ saves remain resume-only.
11. `training_data.npz` snapshots include `value_targets` (per-state cumulative raw attack), `game_numbers` (1-indexed WandB game ids), `game_total_attacks` (raw per-game attack), raw integer counters (`move_numbers` as uint32, `placement_counts` as uint32, `combos` as uint32), and saved board diagnostics (`column_heights`, `max_column_height`, `row_fill_counts` for bottom 4 rows, `total_blocks`, `bumpiness`, `holes`, `overhang_fields`, with `holes`/`overhang_fields` normalized from each example's current board) for exact replay/WandB alignment plus future feature experiments

## Testing

```bash
make test  # Run Rust tests + Python tests (recommended)
cd tetris_core && PYO3_PYTHON=../.venv/bin/python cargo test  # Rust-only equivalent
# In feature worktrees, use the primary checkout virtualenv instead (example):
# cd tetris_core && PYO3_PYTHON=/Users/axelhojmark/Desktop/tetris-mcts/.venv/bin/python cargo test
```

Tests are in:

- `tetris_core/src/piece.rs` - Piece creation and rotation states
- `tetris_core/src/env/tests.rs` - Game logic, line clearing, scoring

## Common Workflows

### Adding new game logic

1. Modify Rust code in `tetris_core/src/env/`
2. Export state/logic to Python in `pymethods.rs` (without presentation concerns)
3. Run `make build` to recompile
4. Test with `make test` and `make play`

### Modifying the neural network

1. Edit `tetris_bot/ml/network.py`
2. Update input encoding in `tetris_core/src/nn.rs` if features change
3. Re-export ONNX after training

Current behavior: split-model Rust inference caches board embeddings as `board_proj(conv(board) ++ board_stats)` where `board_stats` is the 19-dim board-derived statistics (column heights, bumpiness, holes, etc.). On cache hits, both conv and board_stats computation are skipped; only the 61-dim piece/game features are encoded for the heads model. Runtime backend defaults to `tract` and can be switched at runtime with `TETRIS_NN_BACKEND=tract|ort` when the extension is built with Cargo feature `nn-ort`. `fc.bin` stores `board_proj` weights/bias (now shape `(hidden, conv_out + 19)`), and Rust validates that `fc.bin` columns equal conv output width + `BOARD_STATS_FEATURES`. Row-fill diagnostics always use the last `ROW_FILL_FEATURE_ROWS` rows from the provided row-fill slice (tail-based, not absolute-board-index based), and normalization expects at least that many rows. MCTS leaf expansion keeps NN priors sparse (aligned to valid actions) instead of materializing dense 735-action vectors for chance-node caching. Self-play workers also maintain thread-local global caches for move generation and board diagnostics: placements are cached by packed board + current piece state, and `(overhang_fields, holes)` are cached by packed board.

### Training a model

1. Run `python scripts/train.py --training.total-steps N`
2. Creates versioned directory: `training_runs/v0/`, `v1/`, etc.
3. Checkpoints saved to `training_runs/vN/checkpoints/checkpoint_*.pt` with `latest.pt` symlink
4. ONNX exported in `training_runs/vN/checkpoints/` (`latest.onnx`, `parallel.onnx`, and incumbent snapshot `incumbent.onnx` + split artifacts when NN incumbent is active)
5. Training data backed up to `training_runs/vN/training_data.npz` (periodic saves)
6. Resume with `--resume-dir training_runs/vN`

`inspect_training_data.py` supports:

- `--highest_attack_only true` to auto-select the highest-attack game in the snapshot
- `--wandb_game_number <N>` to select by WandB `game_number` when NPZ metadata is present
- If NPZ metadata is missing (older snapshots), `--wandb_game_number` falls back to local index `N-1` with a warning

`analyze_training_data.py` now targets the modern replay schema and fails fast when required keys are missing. Required keys include `placement_counts`, `combos`, `next_hidden_piece_probs`, `column_heights`, `max_column_heights`, `row_fill_counts`, `total_blocks`, `bumpiness`, `holes`, `overhang_fields`, `game_numbers`, and `game_total_attacks`.

## Training Directory Structure

```
training_runs/
├── v0/                          # First training run
│   ├── config.json              # Saved hyperparameters
│   ├── training_data.npz        # Periodic backup (resume capability)
│   └── checkpoints/
│       ├── checkpoint_1000.pt
│       ├── latest.pt
│       ├── latest.onnx
│       ├── parallel.onnx
│       └── incumbent.onnx
├── v1/                          # Second run
└── v2/                          # And so on...
```

- Automatic version incrementing
- Each run isolated with its own checkpoints and config
- Config saved as JSON for reproducibility
- `--resume-dir` creates a new versioned run (for example `v18`) and initializes it from `source_run/checkpoints/latest.pt` while copying `source_run/training_data.npz` when present
- Resumed runs also copy `source_run/checkpoints/incumbent.onnx` (+ split artifacts) when present, so self-play can restart from the saved incumbent artifact instead of the trainer's latest checkpoint export
- Resumed runs restore self-play startup mode from checkpoint field `incumbent_uses_network` (captured on periodic/final saves), so resume starts with NN only if the previous incumbent had been promoted
- Resumed runs restore `incumbent_eval_avg_attack` from checkpoint, so evaluator gating continues against the same promotion baseline instead of resetting to auto-promote mode
- Older checkpoints that predate `incumbent_uses_network` default to starting with NN and emit a warning

## WandB Metrics

Step-alignment rule for resumed runs:

- Any metric namespace/key that should continue on checkpoint `step` must be explicitly mapped with `wandb.define_metric(..., step_metric="trainer_step")`. If a key is not mapped, WandB uses internal `_step`, which resets in new resumed runs.
- Current trainer mappings include `train/*`, `batch/*`, `eval/*`, `timing/*`, `replay/*`, `throughput/*`, `incumbent/*`, `model_gate/*`. All training-specific scalars (policy entropy, value error, accuracies) are namespaced under `train/` so the glob covers them.
- Per-game metrics are mapped to `game_number` via `wandb.define_metric("game/*", step_metric="game_number")` and individual game rows are emitted by default (`log_individual_games_to_wandb=true`). Aggregated per-tick replay summaries under `replay/completed_games_*` on `trainer_step` are always logged.

### Training Metrics

- `loss`, `policy_loss`, `value_loss` - Loss components
- `learning_rate` - Current LR (with scheduling)
- `train/policy_entropy` - Policy distribution entropy
- `buffer_size` - Current examples in memory
- `throughput/games_per_second` and `throughput/steps_per_second` are windowed rates computed from counter deltas divided by elapsed wall-clock seconds since the previous training log tick.
- Candidate evaluator games are only added to `replay/games_generated` if the candidate is promoted. During rejection-heavy periods, evaluator work still consumes compute but does not increment this counter, which can make generation throughput look lower.
- Aggregated replay completion metrics are logged as `replay/completed_games_*` each training log tick (count, first/last game number in the drained window, averages/maxes for attack/lines/moves, and averaged attack-per-move/hold-rate).

### Per-Game Metrics (step_metric="game_number")

- `game/attack` - Total attack in game
- `game/lines` - Total lines cleared
- `game/singles`, `doubles`, `triples`, `tetrises` - Line clear counts
- `game/tspin_*` - T-spin statistics
- `game/max_combo` - Longest combo achieved
- `game/back_to_back` - Back-to-back count

### Evaluation Metrics (from candidate evaluations on fixed seeds)

Eval metrics are derived from candidate model evaluations (not a separate evaluation pass). Each candidate eval plays `model_promotion_eval_games` games on fixed seeds `0..N` and reports per-game results.

- `eval/num_games`, `eval/avg_attack`, `eval/max_attack` - Attack statistics
- `eval/avg_lines`, `eval/max_lines` - Line clear statistics
- `eval/avg_moves` - Average placements per game
- `eval/attack_per_piece`, `eval/lines_per_piece` - Efficiency metrics
- `eval/nn_value_weight` - NN value weight used for the evaluation
- `eval/best_trajectory` GIF renders the highest-attack candidate game replay
- `eval/worst_trajectory` GIF renders the lowest-attack candidate game replay

### WandB Artifacts

- On training shutdown/interruption, trainer uploads a final `model` artifact containing: `checkpoint_<step>.pt`, `latest_metadata.json`, `latest.onnx`, `latest.conv.onnx`, `latest.heads.onnx`, and `latest.fc.bin` (aliases: `latest`, `final`, `step-<N>`).

## Coding Rules

### Code Organization

- **Fix all errors you encounter**: When you see a bug, lint error, type error, or warning — even if it's pre-existing and unrelated to your current task — fix it immediately. Never skip over broken code with a "not my problem" attitude.
- **No fallbacks or backwards compatibility**: When changing file formats, APIs, or data structures, update all code to use the new approach. Don't add fallback code to support old formats.
- **Clean up old code**: When replacing functionality, delete the old implementation entirely. No legacy code paths, deprecated functions, or "just in case" fallbacks.
- **Delete unused code**: If something is no longer used, remove it completely. Don't comment it out, don't add `# removed` markers, don't keep it around.
- **Single approach**: Pick one way to do something and use it consistently. Don't support multiple approaches simultaneously.
- **Top-level imports**: Place all imports at the top of files, never inside functions. Exception: optional dependencies or circular import avoidance.
- **Keep functions short**: Functions should do ONE thing well. If over ~30-50 lines or multiple responsibilities, split it up.
- **Keep it DRY**: Extract common patterns into reusable functions. If copying code, you're doing it wrong.

### Script Arguments

Use `simple_parsing` with dataclasses for CLI scripts:

```python
from dataclasses import dataclass
from pathlib import Path
from simple_parsing import parse
from tetris_bot.constants import PROJECT_ROOT

@dataclass
class ScriptArgs:
    """Script description."""
    data_path: Path  # Required arg with comment as help text
    index: int = -1  # Optional arg with default
    output: Path = PROJECT_ROOT / "outputs" / "results.jsonl"  # Use PROJECT_ROOT for relative paths

def main(args: ScriptArgs) -> None:
    ...

if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
```

**Important**: Put default values directly in the dataclass fields, not as global constants:

```python
# ✅ GOOD: Default values in dataclass
@dataclass
class ScriptArgs:
    model_path: Path = PROJECT_ROOT / "benchmarks" / "models" / "model.onnx"
    batch_size: int = 32

# ❌ BAD: Global constants for defaults
DEFAULT_MODEL = PROJECT_ROOT / "benchmarks" / "models" / "model.onnx"
DEFAULT_BATCH_SIZE = 32

@dataclass
class ScriptArgs:
    model_path: Path = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
```

### Code Simplification

- **Early returns**: Use guard clauses to reduce nesting depth.
- **Positive conditions**: Prefer `if is_valid` over `if not is_invalid`.
- **Extract complex conditionals**: Name complex boolean expressions for clarity.
- **Use `any()`/`all()`**: Instead of loop-with-flag patterns.

```python
# ❌ BAD: Deeply nested
def get_discount(user: User) -> float:
    if user.is_premium:
        if user.years_active > 5:
            return 0.25
        return 0.10
    return 0.0

# ✅ GOOD: Early returns
def get_discount(user: User) -> float:
    if not user.is_premium:
        return 0.0
    if user.years_active <= 5:
        return 0.10
    return 0.25
```

### Naming Conventions

- **Booleans as questions**: `is_active`, `has_data`, `was_successful` - not `active`, `data`, `success`.

### Type Annotations

Always use modern Python syntax:

- ✅ Use: `A | B`, `A | None`, `list[A]`
- ❌ Avoid: `Union[A, B]`, `Optional[A]`, `List[A]`

Use `from __future__ import annotations` for forward references instead of quoted strings.

### Comments & Documentation

- Only add comments for non-intuitive things that cannot be read from code
- Comment "why" not "what"
- **Avoid docstrings** - use clear, descriptive function names instead. Most docstrings become outdated maintenance burdens.
- **Never create documentation files** (.md) unless explicitly requested

### Logging

Use `structlog` for all logging. Never use `print()` for logging.

```python
import structlog
logger = structlog.get_logger()

logger.info("Processing started", user_id=123)
logger.warning("Retry attempt", attempt=2)
logger.error("Failed to process", error=str(e))
```

### File & Path Handling

- Always use `pathlib.Path` for file paths
- **Use relative paths with `PROJECT_ROOT`**: Import `PROJECT_ROOT` from `tetris.config` for project-relative paths instead of hardcoding absolute paths
- For script-relative paths within the same directory, use `Path(__file__).parent`

```python
from tetris_bot.constants import PROJECT_ROOT

# ✅ GOOD: Relative to project root with explicit separators
model_path = PROJECT_ROOT / "benchmarks" / "models" / "model.onnx"
config_path = PROJECT_ROOT / "configs" / "training.yaml"

# ❌ BAD: Hardcoded absolute paths
model_path = Path("/Users/someone/project/benchmarks/models/model.onnx")

# ❌ BAD: Redefining PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Import instead!
```

### Error Handling - Fail Fast

Let it fail; avoid defensive programming when you control the code/data.

Never use silent fallbacks for invalid internal states. If code reaches an impossible or invalid branch, raise an error immediately instead of returning placeholder values.

```python
# ❌ BAD: Silent fallback hides bugs
company = MAPPING.get(key, "default")

# ✅ GOOD: Fail fast
company = MAPPING[key]  # KeyError if not found

# ❌ BAD: Defensive check on controlled data
if hasattr(obj, "field") and obj.field:
    use(obj.field)

# ✅ GOOD: Trust your data structures
use(obj.field)
```

Only be defensive for user inputs, external responses, and dynamic runtime data.
