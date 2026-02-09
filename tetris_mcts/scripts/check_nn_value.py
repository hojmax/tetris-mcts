"""Check NN value estimate by manually constructing a state and running ONNX inference."""

import numpy as np
import onnxruntime as ort

from tetris_mcts.config import PROJECT_ROOT

# Piece name -> index mapping (matches constants.rs / config.py PIECE_NAMES)
PIECE_INDEX = {"I": 0, "O": 1, "T": 2, "S": 3, "Z": 4, "J": 5, "L": 6}
NUM_PIECE_TYPES = 7
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
AUX_FEATURES = 52  # 7 + 8 + 1 + 35 + 1
MAX_MOVES = 100

# ── State from MCTS visualizer (node 3, chance node) ──
CURRENT_PIECE = "J"
HOLD_PIECE = "L"
QUEUE = ["I", "Z", "O", "S", "O"]
MOVE_NUMBER = 2
HOLD_AVAILABLE = True  # hold not used this turn

# Board: T-piece rotation 1, placed in bottom-left corner
# x.
# xx
# x.
BOARD = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
BOARD[17, 0] = 1.0  # row 17, col 0
BOARD[18, 0] = 1.0  # row 18, col 0
BOARD[18, 1] = 1.0  # row 18, col 1
BOARD[19, 0] = 1.0  # row 19, col 0

EXPECTED_VALUE = 0.567
MODEL_PATH = PROJECT_ROOT / "training_runs" / "v6" / "checkpoints" / "parallel.onnx"


def encode_board(board: np.ndarray) -> np.ndarray:
    return board.reshape(1, 1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)


def encode_aux(
    current_piece: str,
    hold_piece: str | None,
    hold_available: bool,
    queue: list[str],
    move_number: int,
) -> np.ndarray:
    aux = np.zeros(AUX_FEATURES, dtype=np.float32)
    idx = 0

    # Current piece: one-hot (7)
    aux[idx + PIECE_INDEX[current_piece]] = 1.0
    idx += NUM_PIECE_TYPES

    # Hold piece: one-hot (8) — 7 pieces + empty slot
    if hold_piece is not None:
        aux[idx + PIECE_INDEX[hold_piece]] = 1.0
    else:
        aux[idx + NUM_PIECE_TYPES] = 1.0  # empty marker
    idx += NUM_PIECE_TYPES + 1

    # Hold available: binary (1)
    aux[idx] = 1.0 if hold_available else 0.0
    idx += 1

    # Queue: 5 slots x 7 one-hot = 35
    for slot, piece_name in enumerate(queue):
        aux[idx + slot * NUM_PIECE_TYPES + PIECE_INDEX[piece_name]] = 1.0
    idx += 5 * NUM_PIECE_TYPES

    # Move number: normalized
    aux[idx] = move_number / MAX_MOVES
    idx += 1

    assert idx == AUX_FEATURES
    return aux.reshape(1, AUX_FEATURES)


def main() -> None:
    board_tensor = encode_board(BOARD)
    aux_tensor = encode_aux(
        CURRENT_PIECE, HOLD_PIECE, HOLD_AVAILABLE, QUEUE, MOVE_NUMBER
    )

    print(f"Board tensor shape: {board_tensor.shape}")
    print(f"Aux tensor shape:   {aux_tensor.shape}")
    print(f"Board sum (filled cells): {board_tensor.sum():.0f}")
    print()

    # Decode aux for verification
    aux = aux_tensor[0]
    current_idx = np.argmax(aux[0:7])
    hold_slice = aux[7:15]
    hold_idx = np.argmax(hold_slice)
    hold_name = list(PIECE_INDEX.keys())[hold_idx] if hold_idx < 7 else "empty"
    print(f"Encoded current piece: {list(PIECE_INDEX.keys())[current_idx]}")
    print(f"Encoded hold piece:    {hold_name}")
    print(f"Encoded hold avail:    {aux[15]}")
    queue_names = []
    for s in range(5):
        q_idx = np.argmax(aux[16 + s * 7 : 16 + (s + 1) * 7])
        queue_names.append(list(PIECE_INDEX.keys())[q_idx])
    print(f"Encoded queue:         {queue_names}")
    print(f"Encoded move number:   {aux[51]} (raw {MOVE_NUMBER}/{MAX_MOVES})")
    print()

    # Load ONNX and run inference
    session = ort.InferenceSession(str(MODEL_PATH))
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"ONNX input names: {input_names}")

    outputs = session.run(
        None,
        {input_names[0]: board_tensor, input_names[1]: aux_tensor},
    )
    policy_logits = outputs[0][0]
    value = outputs[1][0, 0]

    print(f"\nNN value estimate: {value:.6f}")
    print(f"Expected value:    {EXPECTED_VALUE:.6f}")
    print(f"Difference:        {abs(value - EXPECTED_VALUE):.6f}")
    print(f"Match: {'YES' if abs(value - EXPECTED_VALUE) < 0.001 else 'NO'}")

    # Show top-5 policy actions
    top5 = np.argsort(policy_logits)[-5:][::-1]
    print("\nTop 5 raw logits:")
    for i, idx in enumerate(top5):
        print(f"  action {idx}: {policy_logits[idx]:.4f}")


if __name__ == "__main__":
    main()
