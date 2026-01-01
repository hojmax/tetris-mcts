# Tetris MCTS

A Tetris implementation with a high-performance Rust backend and Python pygame visualization. Designed for both interactive play and AI/MCTS experimentation.

## Architecture

- **Rust Core** (`tetris_core/`): Game logic implemented in Rust with PyO3 bindings for Python interop
- **Python Frontend** (`tetris_game.py`): Pygame-based visualization and user input handling

## Installation

### Prerequisites

- Python 3.8+
- Rust (install via [rustup](https://rustup.rs/))
- uv or pip

### Setup

```bash
# Clone the repository
cd tetris-mcts

# Create virtual environment (if not exists)
uv venv
source .venv/bin/activate

# Install Python dependencies
uv pip install pygame maturin

# Build and install the Rust extension
cd tetris_core
maturin develop --release
cd ..
```

## Usage

### Interactive Play

```bash
source .venv/bin/activate
python tetris_game.py
```

### Controls

| Key   | Action                   |
| ----- | ------------------------ |
| ← →   | Move left/right          |
| ↓     | Soft drop                |
| ↑ / X | Rotate clockwise         |
| Z     | Rotate counter-clockwise |
| Space | Hard drop                |
| P     | Pause                    |
| R     | Restart                  |
| ESC   | Quit                     |

### Programmatic API

The Rust backend exposes a clean API for AI/MCTS implementations:

```python
from tetris_core import TetrisEnv, Piece

# Create environment
env = TetrisEnv(width=10, height=20)

# Actions: 0=noop, 1=left, 2=right, 3=down, 4=rotate_cw, 5=rotate_ccw, 6=hard_drop
reward, game_over = env.step(action)

# Clone state for MCTS simulation
env_copy = env.clone_state()

# Access game state
board = env.get_board()           # 2D list of cell values
colors = env.get_board_colors()   # 2D list of piece type indices
piece = env.get_current_piece()   # Current falling piece
next_piece = env.get_next_piece() # Next piece preview
ghost = env.get_ghost_piece()     # Ghost piece (drop preview)

# Piece properties
piece.x, piece.y                  # Position
piece.piece_type                  # 0-6 (I, O, T, S, Z, J, L)
piece.rotation                    # 0-3
piece.get_cells()                 # List of (x, y) coordinates
piece.get_color()                 # RGB tuple

# Environment properties
env.score                         # Current score
env.lines_cleared                 # Total lines cleared
env.level                         # Current level
env.game_over                     # Game over flag

# Direct movement methods
env.move_left()                   # Returns True if successful
env.move_right()
env.move_down()
env.rotate_cw()
env.rotate_ccw()
env.hard_drop()                   # Returns drop distance
env.tick()                        # Gravity tick (same as move_down)
env.reset()                       # Reset to initial state
```

## Scoring

| Lines Cleared | Points      |
| ------------- | ----------- |
| 1 (Single)    | 100 × level |
| 2 (Double)    | 300 × level |
| 3 (Triple)    | 500 × level |
| 4 (Tetris)    | 800 × level |

Hard drops award 2 points per cell dropped.

## Project Structure

```
tetris-mcts/
├── tetris_game.py          # Pygame visualization
├── requirements.txt        # Python dependencies
├── README.md
└── tetris_core/            # Rust library
    ├── Cargo.toml
    ├── pyproject.toml
    └── src/
        └── lib.rs          # Game logic + PyO3 bindings
```

## License

MIT
