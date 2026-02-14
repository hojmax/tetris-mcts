# Project Context for Agents

## Keep This File Current (All Agents)

Treat this document as a living resource, not static documentation.

- If you spot inaccurate, outdated, or misleading information while working, update it immediately in the same change.
- If you discover new critical context (architecture constraints, workflows, gotchas, debugging tips, command conventions), add it here.
- Make small, iterative improvements continuously so future agents inherit accurate context.
- When handing off work, call out any updates you made here and suggest follow-up updates if context is still incomplete.

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
make test       # Run Rust tests (cargo test)
make check      # Run ruff + pyright linting
make sweep-lr-model  # Run W&B sweep for learning rate + model size
make eval-nn-value-weight  # Evaluate fixed network at multiple nn_value_weight values
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
python tetris_mcts/train.py --training.total-steps 100000

# With custom hyperparameters
python tetris_mcts/train.py \
    --training.total-steps 500000 \
    --training.num-simulations 800 \
    --training.learning-rate 0.0005

# Resume from checkpoint (creates a new versioned run initialized from v0/latest.pt)
python tetris_mcts/train.py --resume-dir training_runs/v0
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
python tetris_mcts/scripts/compare_offline_architectures.py \
    --data_path training_runs/v32/training_data.npz

# Optional: preload selected examples to GPU once to reduce per-batch transfer overhead
python tetris_mcts/scripts/compare_offline_architectures.py \
    --data_path training_runs/v32/training_data.npz \
    --preload_to_gpu true
```

### Offline Feature Ablation (State Features)

```bash
# Compare gated-fusion variants with extra board-state diagnostics:
# no-extra, all-extra, and all-minus-one ablations.
# Logs per-step metrics for each variant plus overlay charts to WandB.
# Requires NPZ snapshots that include:
# column_heights, max_column_heights, min_column_heights,
# row_fill_counts, total_blocks, bumpiness (and optionally move_numbers).
python tetris_mcts/scripts/compare_offline_feature_ablation.py \
    --data_path training_runs/v32/training_data.npz

# Optional: include move_numbers as an additional ablation group
python tetris_mcts/scripts/compare_offline_feature_ablation.py \
    --data_path training_runs/v32/training_data.npz \
    --include_move_number_feature true
```

### Replay Buffer Combo Patch

```bash
# Patch older replay buffers where combos are not normalized.
# Rules used by the script:
# - If max(combos) > 1.0 => not normalized
# - If combos contain only 0/1 and no intermediate values => not normalized
# Then it rewrites the NPZ in-place with combos = combos / 12.
python tetris_mcts/scripts/normalize_combo_feature_npz.py \
    --data_path training_runs/v17/training_data.npz

# Dry-run check (no file rewrite)
python tetris_mcts/scripts/normalize_combo_feature_npz.py \
    --data_path training_runs/v17/training_data.npz \
    --dry_run true
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

Results saved to `benchmarks/profile_results.jsonl` with timing data for comparison across runs.

**Interactive Profiling** (requires [samply](https://github.com/mstange/samply)):

```bash
# Install samply (one-time)
cargo install samply

# Profile and view flamegraph in browser
make profile-samply SIMS=50

# Or run directly
samply record python tetris_mcts/scripts/profile_games.py --num_games 3 --simulations 50
```

Opens interactive flamegraph viewer showing ALL function calls automatically. Best for finding bottlenecks during development.

**macOS native profiling** (Instruments):

```bash
instruments -t "Time Profiler" python tetris_mcts/scripts/profile_games.py --num_games 3
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
- Python (`tetris_mcts/`) = training, evaluation UX, and visualization/rendering.
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
├── nn.rs                    # ONNX inference via tract-onnx
└── generator/               # Background game generation
    ├── game_generator.rs
    ├── evaluation.rs
    └── npz.rs

tetris_mcts/                 # Python package
├── config.py                # TrainingConfig dataclass (all hyperparameters)
├── ml/
│   ├── network.py           # TetrisNet (PyTorch CNN)
│   ├── training.py          # Trainer class, training loop
│   ├── loss.py              # Loss functions and metrics
│   ├── evaluation.py        # Model evaluation on fixed seeds
│   ├── weights.py           # Checkpoint/ONNX export
│   └── visualization.py     # Board rendering for eval trajectories
└── scripts/
    ├── tetris_game.py              # Interactive Pygame game
    ├── mcts_visualizer.py          # Dash tree visualization (localhost:8050)
    ├── replay_viewer.py            # View saved game replays
    ├── buffer_viewer.py            # Inspect GameGenerator's in-memory buffer
    ├── inspect_training_data.py    # View contents of NPZ files
    ├── analyze_training_data.py    # Compute statistics over training data
    ├── compare_offline_architectures.py  # Baseline vs gated offline architecture benchmark
    ├── compare_offline_feature_ablation.py  # Offline sweep over state-feature ablations
    ├── normalize_combo_feature_npz.py  # In-place combo normalization patch for old NPZ buffers
    ├── count_reachable_states.py  # Enumerate 734 valid placements
    └── profile_games.py            # Performance profiling of game generation
```

## Key Concepts

### Action Space (735 actions)

All valid (x, y, rotation) placements are enumerated. The `ActionSpace` struct maps between action indices and placements.

Placement metadata includes `last_move_was_rotation` and `last_kick_index` for T-spin scoring when executing action indices directly. If multiple shortest input paths reach the same placement, move generation prefers paths whose last move is **not** a rotation to avoid accidental T-spin attribution from arbitrary path tie breaks.

### MCTS with Chance Nodes

Unlike standard AlphaZero, Tetris has stochastic piece spawning:

- **DecisionNode**: Player chooses action (move/rotate/drop)
- **ChanceNode**: Random piece spawns from 7-bag

### Neural Network (TetrisNet)

- **Input**: 261 features (200 board cells + 61 auxiliary: current piece, hold, queue, move number, combo, back-to-back, hidden-piece distribution)
- **Architecture**: Conv2d(1→4→8) + board projection + aux-conditioned gated fusion + optional fusion residual blocks + policy/value heads
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
- `TrainingExample` - State + MCTS policy target + value target (`value`, raw cumulative attack) + saved board diagnostics (`column_heights`, `max_column_height`, `min_column_height`, `row_fill_counts`, `total_blocks`, `bumpiness`, `holes`, `overhang_fields`, where `holes`/`overhang_fields` are from the current state board)
- `GameGenerator` - Background self-play worker

### Python

- `TetrisNet` - PyTorch neural network
- `Trainer` - Training loop manager
- `WeightManager` - Checkpoint and ONNX export
- `Evaluator` - Model evaluation on fixed seeds

## Code Patterns

### Default Hyperparameters

From `config.py` TrainingConfig defaults:

- **MCTS**: 1000 simulations, c_puct=1.5, temperature=0.8
- **Training**: batch_size=1024, lr=0.0005, linear schedule to 0.0001 over 200k steps (then constant), weight_decay=1e-4
- **Architecture**: Conv(1→4→8), gated-fusion hidden size 128, 735 policy outputs, 1 value output
- **Buffer**: 1M examples (ring buffer), 7 parallel workers, staged sampling with `prefetch_batches=8` (one Rust sample call stages `batch_size * prefetch_batches` examples), and staged queue target `staged_batch_cache_batches=16` (train-sized batches kept resident on host/device queue before being consumed); `pin_memory_batches=true` enables pinned-host transfer on CUDA. Full replay mirroring is enabled by default on accelerator training (`mirror_replay_on_accelerator=true`): snapshot replay to device once, then incrementally append replay deltas every `replay_mirror_refresh_seconds` in chunks of `replay_mirror_delta_chunk_examples`.
- **Exploration**: Dirichlet alpha=0.02, epsilon=0.25, visit-sampling epsilon=0.0
- **NN Value Scaling**: `nn_value_weight=0.025` by default.
- **Wall-Clock Intervals**: training cadence is time-based (not step-based): `log_interval_seconds=10`, `model_sync_interval_seconds=300`, `eval_interval_seconds=1800`, `checkpoint_interval_seconds=10800`; replay snapshots use `save_interval_seconds=10800` (`0` disables periodic snapshot saves).
- **Model Promotion Gate**: candidate window=50 games, evaluator noise enabled by default
- **Bootstrap Mode**: starts without NN, uses 4000 simulations until first promoted model

Override via CLI: `--training.num-simulations 800 --training.learning-rate 0.0005`

Temperature behavior:

- `temperature` shapes the MCTS visit-count policy target used for training.
- In training self-play, action execution samples from the visit policy with probability `visit_sampling_epsilon` and otherwise uses argmax.
- In evaluation, action execution is deterministic argmax.

### Loss Function

```python
policy_loss = -sum(target_policy * log(masked_policy))  # Cross-entropy
value_loss = MSE(predicted_value, target_value)
total_loss = policy_loss + value_loss
```

### Move Masking

Invalid actions get logits set to -inf before softmax, ensuring 0 probability.
Dirichlet root noise is mixed over `DecisionNode.action_priors` (valid actions only), not all 735 actions.

### Self-Play Data Generation

Training uses parallel Rust game generation via `GameGenerator`:

1. Multiple worker threads (default: 7) run MCTS games in parallel
2. One dedicated evaluator worker tests queued candidate ONNX models over a fixed game window (default 50 games)
3. Candidate evaluation games use fresh random environment seeds (not a fixed seed list)
4. Candidates are compared against incumbent lifetime average attack; if better, evaluator commits candidate games then promotes the model globally
5. If candidate is worse, evaluator discards candidate games and keeps incumbent
6. If multiple candidates queue while evaluator is busy, only the newest pending candidate is kept
7. Before first promotion (default), workers run no-network MCTS (uniform policy prior + zero value) with separate simulation count
8. Training examples from accepted games are stored in a shared in-memory ring buffer
9. Python sampling has two modes:
   - Default staged mode: `generator.sample_batch(batch_size * prefetch_batches, max_placements)`, then move staged tensors once to the training device, split into train-sized batches, and keep a queued cache up to `staged_batch_cache_batches` before consuming.
   - Full mirror mode (CUDA/MPS only, `mirror_replay_on_accelerator=true`): `generator.replay_buffer_snapshot(max_placements)` initializes a full device mirror, then `generator.replay_buffer_delta(from_index, max_examples, max_placements)` incrementally appends new examples and drops evicted prefix rows to match FIFO windowing. Rust now snapshots replay rows and logical index bounds atomically from a shared replay state (`SharedBufferState`), so Python deltas stay aligned with FIFO index space under concurrent generation.
   Both modes use `(boards, aux, policy_targets, value_targets, overhang_fields, action_masks)` tensors; periodic NPZ saves remain resume-only.
10. `training_data.npz` snapshots include `value_targets` (per-state cumulative raw attack), `game_numbers` (1-indexed WandB game ids), `game_total_attacks` (raw per-game attack), normalized aux scalars (`move_numbers`, `placement_counts`, `combos` where combo is clamped at 12 then divided by 12), and saved board diagnostics (`column_heights`, `max_column_height`, `min_column_height`, `row_fill_counts`, `total_blocks`, `bumpiness`, `holes`, `overhang_fields`, with `holes`/`overhang_fields` computed from each example's current board) for exact replay/WandB alignment plus future feature experiments

## Testing

```bash
make test  # Run Rust tests + Python tests (recommended)
cd tetris_core && PYO3_PYTHON=../.venv/bin/python cargo test  # Rust-only equivalent
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

1. Edit `tetris_mcts/ml/network.py`
2. Update input encoding in `tetris_core/src/nn.rs` if features change
3. Re-export ONNX after training

Current behavior: split-model Rust inference caches board embeddings as `board_proj(conv(board))`. `fc.bin` stores `board_proj` weights/bias only, and Rust validates that `conv.onnx` output width matches `fc.bin` columns, so changing `conv_filters[-1]` still does not require a Rust constant edit. Self-play workers also maintain thread-local global caches for move generation and board diagnostics: placements are cached by packed board + current piece state, and `(overhang_fields, holes)` are cached by packed board.

### Training a model

1. Run `python tetris_mcts/train.py --training.total-steps N`
2. Creates versioned directory: `training_runs/v0/`, `v1/`, etc.
3. Checkpoints saved to `training_runs/vN/checkpoints/checkpoint_*.pt` with `latest.pt` symlink
4. ONNX exported in `training_runs/vN/checkpoints/` (`latest.onnx` and `parallel.onnx`)
5. Training data backed up to `training_runs/vN/training_data.npz` (periodic saves)
6. Resume with `--resume-dir training_runs/vN`

`inspect_training_data.py` supports:

- `--highest_attack_only true` to auto-select the highest-attack game in the snapshot
- `--wandb_game_number <N>` to select by WandB `game_number` when NPZ metadata is present
- If NPZ metadata is missing (older snapshots), `--wandb_game_number` falls back to local index `N-1` with a warning

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
│       └── parallel.onnx
├── v1/                          # Second run
└── v2/                          # And so on...
```

- Automatic version incrementing
- Each run isolated with its own checkpoints and config
- Config saved as JSON for reproducibility
- `--resume-dir` creates a new versioned run (for example `v18`) and initializes it from `source_run/checkpoints/latest.pt` while copying `source_run/training_data.npz` when present
- Resumed runs restore self-play startup mode from checkpoint field `incumbent_uses_network` (captured on periodic/final saves), so resume starts with NN only if the previous incumbent had been promoted
- Older checkpoints that predate `incumbent_uses_network` default to starting with NN and emit a warning

## WandB Metrics

Step-alignment rule for resumed runs:

- Any metric namespace/key that should continue on checkpoint `step` must be explicitly mapped with `wandb.define_metric(..., step_metric="trainer_step")`. If a key is not mapped, WandB uses internal `_step`, which resets in new resumed runs.
- Current trainer mappings include `train/*`, `batch/*`, `eval/*`, `timing/*`, `replay/*`, `throughput/*`, `incumbent/*`, `model_gate/*`, and scalar keys `policy_entropy`, `value_error`, `top1_accuracy`, `top3_accuracy`.
- Per-game metrics remain mapped to `game_number` via `wandb.define_metric("game/*", step_metric="game_number")`.

### Training Metrics

- `loss`, `policy_loss`, `value_loss` - Loss components
- `learning_rate` - Current LR (with scheduling)
- `policy_entropy` - Policy distribution entropy
- `buffer_size` - Current examples in memory
- `throughput/games_per_second` and `throughput/steps_per_second` are windowed rates computed from counter deltas divided by elapsed wall-clock seconds since the previous training log tick.
- Candidate evaluator games are only added to `replay/games_generated` if the candidate is promoted. During rejection-heavy periods, evaluator work still consumes compute but does not increment this counter, which can make generation throughput look lower.

### Per-Game Metrics (step_metric="game_number")

- `game/attack` - Total attack in game
- `game/lines` - Total lines cleared
- `game/singles`, `doubles`, `triples`, `tetrises` - Line clear counts
- `game/tspin_*` - T-spin statistics
- `game/max_combo` - Longest combo achieved
- `game/back_to_back` - Back-to-back count

### Evaluation Metrics (fixed seeds, up to `max_placements`)

- `eval/avg_attack` - Average attack over eval games
- `eval/max_attack` - Best single game
- `eval/attack_per_piece` - Efficiency metric
- Breakdown by clear types and T-spins
- `eval/avg_moves` reflects played placements (hold actions excluded) up to `max_placements` (default 50)
- `eval/trajectory` GIF renders the full first replay episode (all recorded actions plus a final post-action board frame), so length is no longer capped by a visualization frame limit

### WandB Artifacts

- On training shutdown/interruption, trainer uploads a final `model` artifact containing: `checkpoint_<step>.pt`, `latest_metadata.json`, `latest.onnx`, `latest.conv.onnx`, `latest.heads.onnx`, and `latest.fc.bin` (aliases: `latest`, `final`, `step-<N>`).

## Coding Rules

### Code Organization

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
from tetris_mcts.config import PROJECT_ROOT

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
- **Use relative paths with `PROJECT_ROOT`**: Import `PROJECT_ROOT` from `tetris_mcts.config` for project-relative paths instead of hardcoding absolute paths
- For script-relative paths within the same directory, use `Path(__file__).parent`

```python
from tetris_mcts.config import PROJECT_ROOT

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
