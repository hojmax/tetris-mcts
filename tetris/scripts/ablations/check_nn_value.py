from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from simple_parsing import parse

from tetris.ml.network import AUX_FEATURES

# Piece name -> index mapping (matches constants.rs / config.py PIECE_NAMES)
PIECE_INDEX = {"I": 0, "O": 1, "T": 2, "S": 3, "Z": 4, "J": 5, "L": 6}
NUM_PIECE_TYPES = 7
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
MAX_PLACEMENTS = 100
COMBO_NORMALIZATION_MAX = 4.0

# ── State from MCTS visualizer (node 3, chance node) ──
CURRENT_PIECE = "J"
HOLD_PIECE = "L"
QUEUE = ["I", "Z", "O", "S", "O"]
PLACEMENT_COUNT = 2
HOLD_AVAILABLE = True  # hold not used this turn
COMBO = 0
BACK_TO_BACK = False
NEXT_HIDDEN_PIECE_PROBS = [1.0 / NUM_PIECE_TYPES] * NUM_PIECE_TYPES

# Board: T-piece rotation 1, placed in bottom-left corner
# x.
# xx
# x.
BOARD = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
BOARD[17, 0] = 1.0  # row 17, col 0
BOARD[18, 0] = 1.0  # row 18, col 0
BOARD[18, 1] = 1.0  # row 18, col 1
BOARD[19, 0] = 1.0  # row 19, col 0
COLUMN_HEIGHTS = np.array(
    [3.0 / 8.0, 2.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
MAX_COLUMN_HEIGHT = 3.0 / 20.0
MIN_COLUMN_HEIGHT = 0.0 / 6.0
ROW_FILL_COUNTS = np.array(
    [0.0] * 17 + [0.1, 0.2, 0.1],
    dtype=np.float32,
)
TOTAL_BLOCKS = 4.0 / 60.0
BUMPINESS = 5.0 / 200.0
HOLES = 0.0
OVERHANG_FIELDS = 1.0 / 25.0

EXPECTED_VALUE = 0.567


@dataclass
class ScriptArgs:
    model_path: Path  # Path to ONNX model


def encode_board(board: np.ndarray) -> np.ndarray:
    return board.reshape(1, 1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)


def encode_aux(
    current_piece: str,
    hold_piece: str | None,
    hold_available: bool,
    queue: list[str],
    placement_count: int,
    combo: int,
    back_to_back: bool,
    next_hidden_piece_probs: list[float],
    column_heights: np.ndarray,
    max_column_height: float,
    min_column_height: float,
    row_fill_counts: np.ndarray,
    total_blocks: float,
    bumpiness: float,
    holes: float,
    overhang_fields: float,
) -> np.ndarray:
    if len(next_hidden_piece_probs) != NUM_PIECE_TYPES:
        raise ValueError(
            f"next_hidden_piece_probs must have length {NUM_PIECE_TYPES}, got {len(next_hidden_piece_probs)}"
        )

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

    # Placement count: normalized
    aux[idx] = placement_count / MAX_PLACEMENTS
    idx += 1

    # Combo: normalized
    aux[idx] = min(combo, COMBO_NORMALIZATION_MAX) / COMBO_NORMALIZATION_MAX
    idx += 1

    # Back-to-back: binary
    aux[idx] = 1.0 if back_to_back else 0.0
    idx += 1

    # Next hidden piece distribution: 7 probabilities
    aux[idx : idx + NUM_PIECE_TYPES] = np.asarray(
        next_hidden_piece_probs,
        dtype=np.float32,
    )
    idx += NUM_PIECE_TYPES

    aux[idx : idx + BOARD_WIDTH] = column_heights.astype(np.float32)
    idx += BOARD_WIDTH

    aux[idx] = max_column_height
    idx += 1

    aux[idx] = min_column_height
    idx += 1

    aux[idx : idx + BOARD_HEIGHT] = row_fill_counts.astype(np.float32)
    idx += BOARD_HEIGHT

    aux[idx] = total_blocks
    idx += 1

    aux[idx] = bumpiness
    idx += 1

    aux[idx] = holes
    idx += 1

    aux[idx] = overhang_fields
    idx += 1

    assert idx == AUX_FEATURES
    return aux.reshape(1, AUX_FEATURES)


def main(args: ScriptArgs) -> None:
    board_tensor = encode_board(BOARD)
    aux_tensor = encode_aux(
        CURRENT_PIECE,
        HOLD_PIECE,
        HOLD_AVAILABLE,
        QUEUE,
        PLACEMENT_COUNT,
        COMBO,
        BACK_TO_BACK,
        NEXT_HIDDEN_PIECE_PROBS,
        COLUMN_HEIGHTS,
        MAX_COLUMN_HEIGHT,
        MIN_COLUMN_HEIGHT,
        ROW_FILL_COUNTS,
        TOTAL_BLOCKS,
        BUMPINESS,
        HOLES,
        OVERHANG_FIELDS,
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
    print(
        f"Encoded placement count: {aux[51]} (raw {PLACEMENT_COUNT}/{MAX_PLACEMENTS})"
    )
    print(f"Encoded combo:         {aux[52]} (raw {COMBO})")
    print(f"Encoded back-to-back:  {aux[53]}")
    print(f"Encoded hidden dist:   {aux[54:61].tolist()}")
    print(f"Encoded column heights:{aux[61:71].tolist()}")
    print(f"Encoded max/min h:     {aux[71]:.4f} / {aux[72]:.4f}")
    print(f"Encoded row fills:     {aux[73:93].tolist()}")
    print(f"Encoded total blocks:  {aux[93]:.4f}")
    print(f"Encoded bumpiness:     {aux[94]:.4f}")
    print(f"Encoded holes:         {aux[95]:.4f}")
    print(f"Encoded overhang:      {aux[96]:.4f}")
    print()

    # Load ONNX and run inference
    session = ort.InferenceSession(str(args.model_path))
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"ONNX input names: {input_names}")

    outputs = session.run(
        None,
        {input_names[0]: board_tensor, input_names[1]: aux_tensor},
    )
    policy_logits = np.asarray(outputs[0], dtype=np.float32)[0]
    value = float(np.asarray(outputs[1], dtype=np.float32)[0, 0])

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
    main(parse(ScriptArgs))
