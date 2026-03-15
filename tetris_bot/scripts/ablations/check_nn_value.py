from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from simple_parsing import parse

from tetris_core.tetris_core import (
    TetrisEnv,
    debug_encode_state,
    debug_get_action_mask,
    debug_predict_masked_from_tensors,
)
from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, PIECE_NAMES, QUEUE_SIZE

# Hardcoded trajectory states from the checkpoint-correct
# `training_runs/v3` seed-35 playback (`nn_value_weight=1.0`).


@dataclass(frozen=True)
class HardcodedState:
    label: str
    seed: int
    action_prefix: tuple[int, ...]
    max_placements: int
    expected_viz_nn_value: float
    expected_current_piece: str
    expected_hold_piece: str | None
    expected_hold_used: bool
    expected_queue: tuple[str, ...]
    expected_board_rows: tuple[str, ...]


HARD_CODED_STATES = [
    HardcodedState(
        label="seed35_placement0",
        seed=35,
        action_prefix=(),
        max_placements=50,
        expected_viz_nn_value=13.410019874572754,
        expected_current_piece="I",
        expected_hold_piece=None,
        expected_hold_used=False,
        expected_queue=("T", "J", "S", "O", "L"),
        expected_board_rows=(
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ),
    ),
    HardcodedState(
        label="seed35_placement1",
        seed=35,
        action_prefix=(349,),
        max_placements=50,
        expected_viz_nn_value=13.740720748901367,
        expected_current_piece="T",
        expected_hold_piece=None,
        expected_hold_used=False,
        expected_queue=("J", "S", "O", "L", "Z"),
        expected_board_rows=(
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            ".1........",
            ".1........",
            ".1........",
            ".1........",
        ),
    ),
]


@dataclass
class ScriptArgs:
    model_path: Path = (
        Path(__file__).parent.parent.parent.parent
        / "training_runs/v3/checkpoints/incumbent.onnx"
    )
    state: str = "all"  # One of: all, seed35_placement0, seed35_placement1


def piece_name(piece: object | None) -> str | None:
    if piece is None:
        return None
    return PIECE_NAMES[piece.piece_type]


def board_rows_from_env(env: TetrisEnv) -> list[str]:
    return [
        "".join("1" if int(cell) != 0 else "." for cell in row)
        for row in env.get_board()
    ]


def build_env(state: HardcodedState) -> TetrisEnv:
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, state.seed)
    for action in state.action_prefix:
        attack = env.execute_action_index(action)
        if attack is None:
            raise RuntimeError(
                f"Failed to replay action {action} for state {state.label}"
            )
    return env


def validate_state(state: HardcodedState, env: TetrisEnv) -> None:
    current_piece = piece_name(env.get_current_piece())
    hold_piece = piece_name(env.get_hold_piece())
    queue = tuple(PIECE_NAMES[piece] for piece in env.get_queue(QUEUE_SIZE))
    board_rows = tuple(board_rows_from_env(env))
    hold_used = env.is_hold_used()

    if current_piece != state.expected_current_piece:
        raise ValueError(
            f"{state.label}: current piece mismatch {current_piece} != {state.expected_current_piece}"
        )
    if hold_piece != state.expected_hold_piece:
        raise ValueError(
            f"{state.label}: hold piece mismatch {hold_piece} != {state.expected_hold_piece}"
        )
    if queue != state.expected_queue:
        raise ValueError(
            f"{state.label}: queue mismatch {queue} != {state.expected_queue}"
        )
    if board_rows != state.expected_board_rows:
        raise ValueError(f"{state.label}: board rows do not match hardcoded state")
    if hold_used != state.expected_hold_used:
        raise ValueError(
            f"{state.label}: hold_used mismatch {hold_used} != {state.expected_hold_used}"
        )
    if env.placement_count != len(state.action_prefix):
        raise ValueError(
            f"{state.label}: placement_count mismatch {env.placement_count} != {len(state.action_prefix)}"
        )


def iter_selected_states(name: str) -> list[HardcodedState]:
    if name == "all":
        return HARD_CODED_STATES
    for state in HARD_CODED_STATES:
        if state.label == name:
            return [state]
    valid = ", ".join(["all", *[state.label for state in HARD_CODED_STATES]])
    raise ValueError(f"Unknown state '{name}'. Expected one of: {valid}")


def print_aux_summary(aux_tensor: np.ndarray) -> None:
    aux = aux_tensor[0]
    print(f"Encoded placement count: {aux[51]:.6f}")
    print(f"Encoded combo:           {aux[52]:.6f}")
    print(f"Encoded back-to-back:    {aux[53]:.6f}")
    print(f"Encoded hidden dist:     {aux[54:61].tolist()}")
    print(f"Encoded column heights:  {aux[61:71].tolist()}")
    print(f"Encoded max height:      {aux[71]:.6f}")
    print(f"Encoded row fills:       {aux[72:76].tolist()}")
    print(f"Encoded total blocks:    {aux[76]:.6f}")
    print(f"Encoded bumpiness:       {aux[77]:.6f}")
    print(f"Encoded holes:           {aux[78]:.6f}")
    print(f"Encoded overhang:        {aux[79]:.6f}")


def main(args: ScriptArgs) -> None:
    session = ort.InferenceSession(str(args.model_path))
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"ONNX input names: {input_names}")
    print()

    for state in iter_selected_states(args.state):
        env = build_env(state)
        validate_state(state, env)

        board_flat, aux_flat = debug_encode_state(env, state.max_placements)
        board_tensor = np.asarray(board_flat, dtype=np.float32).reshape(
            1, 1, BOARD_HEIGHT, BOARD_WIDTH
        )
        aux_tensor = np.asarray(aux_flat, dtype=np.float32).reshape(1, -1)

        outputs = session.run(
            None,
            {input_names[0]: board_tensor, input_names[1]: aux_tensor},
        )
        policy_logits = np.asarray(outputs[0], dtype=np.float32)[0]
        ort_value = float(np.asarray(outputs[1], dtype=np.float32)[0, 0])
        rust_policy, rust_value = debug_predict_masked_from_tensors(
            str(args.model_path),
            board_tensor.reshape(-1).astype(np.float32).tolist(),
            aux_tensor.reshape(-1).astype(np.float32).tolist(),
            debug_get_action_mask(env),
        )
        rust_policy_logits = np.asarray(rust_policy, dtype=np.float32)
        top5 = np.argsort(policy_logits)[-5:][::-1]
        rust_top5 = np.argsort(rust_policy_logits)[-5:][::-1]

        print(f"=== {state.label} ===")
        print(f"Seed:                   {state.seed}")
        print(f"Action prefix:          {list(state.action_prefix)}")
        print(f"Placement count:        {env.placement_count}")
        print(f"Current piece:          {piece_name(env.get_current_piece())}")
        print(f"Hold piece:             {piece_name(env.get_hold_piece())}")
        print(f"Hold used:              {env.is_hold_used()}")
        print(
            "Queue:                  "
            f"{[PIECE_NAMES[piece] for piece in env.get_queue(QUEUE_SIZE)]}"
        )
        print(f"Board sum:              {int(board_tensor.sum())}")
        print("Board rows:")
        for row in board_rows_from_env(env):
            print(f"  {row}")
        print_aux_summary(aux_tensor)
        print()
        print(f"Rust NN value:          {rust_value:.6f}")
        print(f"ORT NN value:           {ort_value:.6f}")
        print(f"Expected viz NN value:  {state.expected_viz_nn_value:.6f}")
        print(
            f"Rust difference:        {abs(rust_value - state.expected_viz_nn_value):.6f}"
        )
        print(
            "Rust match:             "
            f"{'YES' if abs(rust_value - state.expected_viz_nn_value) < 0.001 else 'NO'}"
        )
        print(f"ORT drift vs Rust:      {abs(ort_value - rust_value):.6f}")
        print("Top 5 ORT raw logits:")
        for action_idx in top5:
            print(f"  action {int(action_idx)}: {policy_logits[int(action_idx)]:.6f}")
        print("Top 5 Rust masked probs:")
        for action_idx in rust_top5:
            print(
                f"  action {int(action_idx)}: {rust_policy_logits[int(action_idx)]:.6f}"
            )
        print()


if __name__ == "__main__":
    main(parse(ScriptArgs))
