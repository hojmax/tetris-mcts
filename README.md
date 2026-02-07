# Tetris MCTS

A Tetris implementation with a high-performance Rust backend and Python tooling for MCTS training, evaluation, profiling, and visualization.

## Architecture

- **Rust Core** (`tetris_core/`): game logic, move generation, scoring, MCTS, ONNX inference, replay buffer generation
- **Python Layer** (`tetris_mcts/`): training loop, model definition, visualization scripts, and CLI utilities

## Requirements

- Python `>=3.12`
- Rust (via [rustup](https://rustup.rs/))
- `uv`

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
make build
```

## Common Commands

```bash
make play        # Interactive Tetris game
make viz         # MCTS tree visualizer (Dash)
make train       # Training entrypoint (forward args via ARGS=...)
make profile     # Benchmark game generation speed
make test        # Rust tests
make check       # Ruff + pyright + cargo fmt/fix
```

## Interactive Play

```bash
source .venv/bin/activate
python tetris_mcts/scripts/tetris_game.py
```

## Controls

| Key   | Action                   |
| ----- | ------------------------ |
| ← →   | Move left/right          |
| ↓     | Soft drop                |
| ↑ / X | Rotate clockwise         |
| Z     | Rotate counter-clockwise |
| Space | Hard drop                |
| C     | Hold                     |
| P     | Pause                    |
| R     | Restart                  |
| ESC   | Quit                     |

## Programmatic API

```python
from tetris_core import TetrisEnv

env = TetrisEnv(width=10, height=20)

# Actions: 0=noop, 1=left, 2=right, 3=down, 4=rotate_cw, 5=rotate_ccw, 6=hard_drop, 7=hold
reward, game_over = env.step(action)

board = env.get_board()
colors = env.get_board_colors()
current_piece = env.get_current_piece()
next_piece = env.get_next_piece()
ghost_piece = env.get_ghost_piece()

# Environment state
env.attack
env.lines_cleared
env.combo
env.back_to_back
env.game_over
```

## Attack Scoring (Current Rules)

| Clear Type          | Base Attack |
| ------------------- | ----------- |
| Single              | 0           |
| Double              | 1           |
| Triple              | 2           |
| Tetris              | 4           |
| T-Spin Mini Single  | 0           |
| T-Spin Single       | 2           |
| T-Spin Double       | 4           |
| T-Spin Triple       | 6           |

Additional bonuses:
- Combo bonus from combo table (`scoring.rs`)
- Back-to-back bonus: `+1`
- Perfect clear bonus: `+10`

## Notes on MCTS Temperature

`temperature` is used to shape the MCTS visit-count policy **training targets**.
Executed moves are selected as the best move (argmax / most-visited child) in both training and evaluation.

## Project Structure

```text
tetris-mcts/
├── tetris_mcts/
│   ├── train.py
│   ├── config.py
│   ├── ml/
│   └── scripts/
├── tetris_core/
│   ├── Cargo.toml
│   └── src/
├── benchmarks/
├── training_runs/
└── README.md
```

## License

MIT
