from pathlib import Path

# Project root (tetris-mcts/)
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_RUNS_DIR = PROJECT_ROOT / "training_runs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# Core game constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
NUM_PIECE_TYPES = 7
QUEUE_SIZE = 5
NUM_ACTIONS = 735

# Artifact names
CHECKPOINT_DIRNAME = "checkpoints"
MODEL_CANDIDATES_DIRNAME = "model_candidates"
CONFIG_FILENAME = "config.json"
PARALLEL_ONNX_FILENAME = "parallel.onnx"
INCUMBENT_ONNX_FILENAME = "incumbent.onnx"
TRAINING_DATA_FILENAME = "training_data.npz"
LATEST_ONNX_FILENAME = "latest.onnx"
LATEST_METADATA_FILENAME = "latest_metadata.json"
LATEST_CHECKPOINT_FILENAME = "latest.pt"
CHECKPOINT_FILENAME_PREFIX = "checkpoint"

# Visualization / logging defaults
DEFAULT_GIF_FRAME_DURATION_MS = 300

# Piece metadata
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_COLORS = [
    (93, 173, 212),  # I - Cyan
    (219, 174, 63),  # O - Yellow
    (178, 74, 156),  # T - Magenta
    (114, 184, 65),  # S - Green
    (204, 65, 65),  # Z - Red
    (59, 84, 165),  # J - Blue
    (227, 127, 59),  # L - Orange
]

# Spawn-rotation (state 0) cell offsets — mirrors Rust TETROMINO_CELLS[][0]
PIECE_SPAWN_CELLS: list[list[tuple[int, int]]] = [
    [(0, 1), (1, 1), (2, 1), (3, 1)],  # I
    [(1, 1), (2, 1), (1, 2), (2, 2)],  # O
    [(1, 0), (0, 1), (1, 1), (2, 1)],  # T
    [(1, 0), (2, 0), (0, 1), (1, 1)],  # S
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z
    [(0, 0), (0, 1), (1, 1), (2, 1)],  # J
    [(2, 0), (0, 1), (1, 1), (2, 1)],  # L
]
