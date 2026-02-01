# CLAUDE.md - Project Context for Claude Code

## Project Overview

This is **tetris-mcts**, an AlphaZero-style reinforcement learning system for Tetris that combines:

- A high-performance **Rust game engine** with PyO3 Python bindings
- **Monte Carlo Tree Search (MCTS)** with neural network guidance
- **PyTorch CNN** for policy and value prediction
- **Pygame visualization** for interactive play

## Quick Commands

```bash
make build      # Compile Rust with maturin
make play       # Run interactive Tetris game
make viz        # Run MCTS tree visualizer (Dash app at localhost:8050)
make test       # Run Rust tests (cargo test)
make check      # Run ruff + pyright linting
make rebuild    # Force rebuild
```

### Training

```bash
python tetris_mcts/scripts/train.py --total-steps 100000 --simulations 100
```

## Architecture

```
Python (PyTorch training)
    ↓
Rust Core (tetris_core/)     ← Fast game logic + MCTS
    ↓
ONNX Export                  ← Model exported for Rust inference
```

**Key design**: Game logic and MCTS run entirely in Rust for speed. Neural network training happens in Python. Models are exported as ONNX for Rust inference.

## Project Structure

```
tetris_core/src/             # Rust game engine
├── lib.rs                   # PyO3 module exports
├── constants.rs             # Board size (10x20), piece indices
├── piece.rs                 # Tetromino shapes, rotations, colors
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
│   ├── action_space.rs      # 734-action mapping
│   ├── results.rs           # TrainingExample, GameResult, GameStats
│   └── utils.rs             # PUCT scoring, Dirichlet noise
├── nn.rs                    # ONNX inference via tract-onnx
└── generator/               # Background game generation
    ├── game_generator.rs
    ├── evaluation.rs
    └── npz.rs

tetris_mcts/                 # Python package
├── ml/
│   ├── network.py           # TetrisNet (PyTorch CNN)
│   ├── training.py          # Trainer class, training loop
│   ├── loss.py              # Loss functions and metrics
│   ├── evaluation.py        # Model evaluation on fixed seeds
│   ├── data.py              # TetrisDataset, NPZ save/load
│   ├── weights.py           # Checkpoint/ONNX export
│   └── visualization.py     # Board rendering for eval trajectories
└── scripts/
    ├── tetris_game.py       # Interactive Pygame game
    ├── train.py             # Training entry point
    ├── mcts_visualizer.py   # Dash tree visualization
    └── count_reachable_states.py
```

## Key Concepts

### Action Space (734 actions)

All valid (x, y, rotation) placements are enumerated. The `ActionSpace` struct maps between action indices and placements.

### MCTS with Chance Nodes

Unlike standard AlphaZero, Tetris has stochastic piece spawning:

- **DecisionNode**: Player chooses action (move/rotate/drop)
- **ChanceNode**: Random piece spawns from 7-bag

### Neural Network (TetrisNet)

- **Input**: 252 features (200 board cells + 52 auxiliary: current piece, hold, queue, move number)
- **Architecture**: Conv2d(1→4→8) + FC(1652→128) + policy head (734) + value head (1)
- **Output**: Policy probabilities over 734 actions, value (predicted cumulative attack)

### 7-Bag Randomizer

Pieces spawn in random order, 7 at a time (no repeats within a bag). The queue shows next 5 pieces.

### Scoring System

- Single: 0 attack, Double: 1, Triple: 2, Tetris: 4
- T-spin bonuses, combo multipliers, back-to-back bonus, perfect clear (10)

## Important Types

### Rust (exported to Python)

- `TetrisEnv` - Game state (board, piece, queue, hold, score, game_over)
- `Piece` - Tetromino (piece_type, x, y, rotation)
- `MCTSAgent` - MCTS search coordinator
- `MCTSConfig` - Search hyperparameters (num_simulations, c_puct, temperature, etc.)
- `TrainingExample` - State + MCTS policy target + value target
- `GameGenerator` - Background self-play worker

### Python

- `TetrisNet` - PyTorch neural network
- `Trainer` - Training loop manager
- `WeightManager` - Checkpoint and ONNX export
- `Evaluator` - Model evaluation on fixed seeds

## Code Patterns

### Loss Function

```python
policy_loss = -sum(target_policy * log(masked_policy))  # Cross-entropy
value_loss = MSE(predicted_value, target_attack)
total_loss = policy_loss + value_loss
```

### Move Masking

Invalid actions get logits set to -inf before softmax, ensuring 0 probability.

### Self-Play Data Generation

Training uses parallel Rust game generation:

1. Rust `GameGenerator` runs in background thread
2. MCTS agent plays games using network priors
3. Training examples stored in shared in-memory buffer
4. Python samples directly via `generator.sample_batch()` - no disk I/O
5. Periodic disk saves (NPZ) for resume capability only
6. Model hot-swapped when Python exports new ONNX

## Testing

```bash
make test           # Run Rust tests
cargo test -p tetris_core  # Equivalent
```

Tests are in:

- `tetris_core/src/piece.rs` - Piece creation, colors, rotation states
- `tetris_core/src/env/tests.rs` - Game logic, line clearing, scoring

## Dependencies

### Rust (tetris_core/Cargo.toml)

- `pyo3` (0.20) - Python bindings
- `numpy` (0.20) - NumPy array interop
- `tract-onnx` (0.21) - ONNX inference
- `rand`, `rand_distr` - RNG, Dirichlet sampling
- `npyz` - NPZ file format

### Python (pyproject.toml)

- `torch` - Neural network training
- `pygame` - Interactive game
- `wandb` - Experiment tracking
- `dash`, `dash-cytoscape` - MCTS tree visualization
- `maturin` - Build Rust extension

## Common Workflows

### Adding new game logic

1. Modify Rust code in `tetris_core/src/env/`
2. Export to Python in `pymethods.rs`
3. Run `make build` to recompile
4. Test with `make test` and `make play`

### Modifying the neural network

1. Edit `tetris_mcts/ml/network.py`
2. Update input encoding in `tetris_core/src/nn.rs` if features change
3. Re-export ONNX after training

### Training a model

1. `python tetris_mcts/scripts/train.py --total-steps N`
2. Checkpoints saved to `outputs/checkpoints/`
3. ONNX exported as `parallel.onnx` for Rust inference
4. Game data periodically saved to `outputs/data/games/training_data.npz` for resume

## Coding Rules

- **No fallbacks or backwards compatibility**: When changing file formats, APIs, or data structures, update all code to use the new approach. Don't add fallback code to support old formats.
- **Clean up old code**: When replacing functionality, delete the old implementation entirely. No legacy code paths, deprecated functions, or "just in case" fallbacks.
- **Delete unused code**: If something is no longer used, remove it completely. Don't comment it out, don't add `# removed` markers, don't keep it around.
- **Single approach**: Pick one way to do something and use it consistently. Don't support multiple approaches simultaneously.
