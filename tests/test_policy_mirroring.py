from __future__ import annotations

import numpy as np
import torch

import tetris_core.tetris_core as tetris_core
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    HOLD_ACTION_INDEX,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    QUEUE_SIZE,
)
from tetris_bot.ml.aux_features import AUX_FEATURE_LAYOUT
from tetris_bot.ml.network import build_aux_features
from tetris_bot.ml.policy_mirroring import (
    FLAT_VALID_INDICES_BY_PIECE,
    LEGACY_HOLD_ACTION_INDEX,
    LEGACY_NUM_ACTIONS,
    MIRROR_PIECE_TYPE_ORDER,
    NEW_NUM_ACTIONS,
    current_piece_indices_from_aux,
    legacy_action_masks_to_flat,
    mirror_aux_features,
    mirror_boards,
    mirror_flat_action_masks,
    mirror_flat_policy_targets,
    mirror_piece_indices,
    mirror_training_tensors,
    maybe_mirror_training_tensors,
)


def _piece_one_hot(piece_type: int) -> np.ndarray:
    one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    one_hot[piece_type] = 1.0
    return one_hot


def _hold_one_hot(piece_type: int | None) -> np.ndarray:
    one_hot = np.zeros(NUM_PIECE_TYPES + 1, dtype=np.float32)
    if piece_type is None:
        one_hot[NUM_PIECE_TYPES] = 1.0
    else:
        one_hot[piece_type] = 1.0
    return one_hot


def _mirror_piece_type(piece_type: int | None) -> int | None:
    if piece_type is None:
        return None
    return MIRROR_PIECE_TYPE_ORDER[piece_type]


def _mirror_queue(queue: list[int]) -> list[int]:
    return [MIRROR_PIECE_TYPE_ORDER[piece_type] for piece_type in queue]


def _flat_mask_from_env(env: tetris_core.TetrisEnv) -> torch.Tensor:
    action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)
    if action_mask.shape == (NUM_ACTIONS,):
        return torch.from_numpy(action_mask.copy())
    if action_mask.shape != (LEGACY_NUM_ACTIONS,):
        raise AssertionError(f"Unexpected env action-mask width: {action_mask.shape}")
    current_piece = env.get_current_piece()
    if current_piece is None:
        raise AssertionError("Expected env to have a current piece")
    current_pieces = torch.tensor([current_piece.piece_type], dtype=torch.long)
    return legacy_action_masks_to_flat(
        torch.from_numpy(action_mask).unsqueeze(0),
        current_pieces,
    )[0]


def _mirror_env_for_action_mask(env: tetris_core.TetrisEnv) -> tetris_core.TetrisEnv:
    mirrored = tetris_core.TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 0)
    board = env.get_board()
    mirrored.set_board([list(reversed(row)) for row in board])

    current_piece = env.get_current_piece()
    if current_piece is None:
        raise AssertionError("Expected env to have a current piece")
    mirrored.set_current_piece_type(MIRROR_PIECE_TYPE_ORDER[current_piece.piece_type])

    hold_piece = env.get_hold_piece()
    mirrored.set_hold_piece_type(_mirror_piece_type(None if hold_piece is None else hold_piece.piece_type))
    mirrored.set_hold_used(env.is_hold_used())
    mirrored.set_queue(_mirror_queue(env.get_queue(QUEUE_SIZE)))
    return mirrored


def _collect_reachable_env_states(target_count: int) -> list[tetris_core.TetrisEnv]:
    rng = np.random.default_rng(123)
    states: list[tetris_core.TetrisEnv] = []
    seed = 100
    while len(states) < target_count:
        env = tetris_core.TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)
        seed += 1
        for _ in range(12):
            states.append(env.clone_state())
            if len(states) >= target_count:
                break
            action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)
            valid_actions = np.flatnonzero(action_mask)
            if valid_actions.size == 0:
                break
            if action_mask.shape == (NUM_ACTIONS,):
                hold_action_index = HOLD_ACTION_INDEX
            elif action_mask.shape == (LEGACY_NUM_ACTIONS,):
                hold_action_index = LEGACY_HOLD_ACTION_INDEX
            else:
                raise AssertionError(
                    f"Unexpected env action-mask width while stepping: {action_mask.shape}"
                )
            if action_mask[hold_action_index] and rng.random() < 0.3:
                action = hold_action_index
            else:
                action = int(rng.choice(valid_actions))
            attack = env.execute_action_index(action)
            if attack is None:
                break
    return states


def test_mirror_aux_features_remaps_piece_planes_and_is_involution() -> None:
    aux = build_aux_features(
        current_piece=_piece_one_hot(3),
        hold_piece=_hold_one_hot(6),
        hold_available=1.0,
        next_queue=np.stack(
            [
                _piece_one_hot(0),
                _piece_one_hot(3),
                _piece_one_hot(4),
                _piece_one_hot(5),
                _piece_one_hot(6),
            ]
        ),
        placement_count=0.42,
        combo_feature=0.75,
        back_to_back=1.0,
        next_hidden_piece_probs=np.asarray(
            [0.10, 0.15, 0.20, 0.05, 0.12, 0.18, 0.20], dtype=np.float32
        ),
        column_heights=np.asarray(
            [0.01, 0.08, 0.12, 0.20, 0.35, 0.41, 0.50, 0.62, 0.78, 0.91],
            dtype=np.float32,
        ),
        max_column_height=0.91,
        row_fill_counts=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        total_blocks=0.33,
        bumpiness=0.27,
        holes=0.19,
        overhang_fields=0.11,
    )
    aux_batch = torch.from_numpy(np.stack([aux, aux], axis=0))

    mirrored = mirror_aux_features(aux_batch)
    layout = AUX_FEATURE_LAYOUT

    assert current_piece_indices_from_aux(mirrored).tolist() == [4, 4]
    assert mirrored[0, layout.hold_piece].argmax().item() == 5
    mirrored_queue = (
        mirrored[0, layout.next_queue].reshape(QUEUE_SIZE, NUM_PIECE_TYPES).argmax(dim=1)
    )
    assert mirrored_queue.tolist() == [0, 4, 3, 6, 5]
    assert torch.allclose(
        mirrored[0, layout.next_hidden_piece_probs],
        torch.tensor([0.10, 0.15, 0.20, 0.12, 0.05, 0.20, 0.18]),
    )
    assert torch.allclose(
        mirrored[0, layout.column_heights],
        aux_batch[0, layout.column_heights].flip(dims=(0,)),
    )
    assert mirrored[0, layout.max_column_height].item() == aux_batch[
        0, layout.max_column_height
    ].item()
    assert mirrored[0, layout.bumpiness].item() == aux_batch[0, layout.bumpiness].item()
    assert mirrored[0, layout.holes].item() == aux_batch[0, layout.holes].item()
    assert mirrored[0, layout.overhang_fields].item() == aux_batch[
        0, layout.overhang_fields
    ].item()

    assert torch.allclose(mirror_aux_features(mirrored), aux_batch)


def test_mirror_training_tensors_are_involutions_and_preserve_policy_support() -> None:
    states = _collect_reachable_env_states(target_count=6)
    boards = []
    aux_rows = []
    masks = []
    policies = []
    current_pieces = []
    rng = np.random.default_rng(7)

    for env in states:
        flat_mask = _flat_mask_from_env(env)
        current_piece = env.get_current_piece()
        if current_piece is None:
            raise AssertionError("Expected env to have a current piece")
        current_pieces.append(current_piece.piece_type)
        boards.append(np.asarray(env.get_board(), dtype=np.float32))

        hold_piece = env.get_hold_piece()
        next_queue = env.get_queue(QUEUE_SIZE)
        aux_rows.append(
            build_aux_features(
                current_piece=_piece_one_hot(current_piece.piece_type),
                hold_piece=_hold_one_hot(
                    None if hold_piece is None else hold_piece.piece_type
                ),
                hold_available=float(not env.is_hold_used()),
                next_queue=np.stack([_piece_one_hot(piece) for piece in next_queue]),
                placement_count=0.2,
                combo_feature=float(env.combo) / 4.0,
                back_to_back=float(env.back_to_back),
                next_hidden_piece_probs=np.full(NUM_PIECE_TYPES, 1.0 / NUM_PIECE_TYPES),
                column_heights=np.linspace(0.0, 0.9, BOARD_WIDTH, dtype=np.float32),
                max_column_height=0.9,
                row_fill_counts=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                total_blocks=0.25,
                bumpiness=0.33,
                holes=0.12,
                overhang_fields=0.07,
            )
        )
        masks.append(flat_mask)
        weights = torch.zeros(NEW_NUM_ACTIONS, dtype=torch.float32)
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).flatten().numpy()
        weights[valid_indices] = torch.from_numpy(
            rng.random(size=valid_indices.size).astype(np.float32)
        )
        weights /= weights.sum()
        policies.append(weights)

    board_tensor = torch.from_numpy(np.stack(boards, axis=0)).unsqueeze(1).bool()
    aux_tensor = torch.from_numpy(np.stack(aux_rows, axis=0))
    policy_tensor = torch.stack(policies, dim=0)
    mask_tensor = torch.stack(masks, dim=0)
    piece_tensor = torch.tensor(current_pieces, dtype=torch.long)

    mirrored_boards, mirrored_aux, mirrored_policy, mirrored_masks = (
        mirror_training_tensors(
            board_tensor,
            aux_tensor,
            policy_tensor,
            mask_tensor,
        )
    )

    assert torch.equal(mirrored_boards, mirror_boards(board_tensor))
    assert torch.equal(
        current_piece_indices_from_aux(mirrored_aux),
        mirror_piece_indices(piece_tensor),
    )
    assert torch.equal(mirrored_policy > 0, (mirrored_policy > 0) & mirrored_masks)
    assert torch.allclose(mirrored_policy.sum(dim=1), torch.ones(len(states)))
    assert torch.equal(mirrored_masks.sum(dim=1), mask_tensor.sum(dim=1))

    roundtrip = mirror_training_tensors(
        mirrored_boards,
        mirrored_aux,
        mirrored_policy,
        mirrored_masks,
    )
    assert torch.equal(roundtrip[0], board_tensor)
    assert torch.allclose(roundtrip[1], aux_tensor)
    assert torch.allclose(roundtrip[2], policy_tensor)
    assert torch.equal(roundtrip[3], mask_tensor)


def test_flat_mask_transform_matches_mirrored_env_masks() -> None:
    for env in _collect_reachable_env_states(target_count=20):
        original_mask = _flat_mask_from_env(env)
        current_piece = env.get_current_piece()
        if current_piece is None:
            raise AssertionError("Expected env to have a current piece")

        transformed_mask = mirror_flat_action_masks(
            original_mask.unsqueeze(0),
            torch.tensor([current_piece.piece_type], dtype=torch.long),
        )[0]

        mirrored_env = _mirror_env_for_action_mask(env)
        expected_mask = _flat_mask_from_env(mirrored_env)

        assert torch.equal(transformed_mask, expected_mask)
        assert transformed_mask.sum().item() == original_mask.sum().item()

        policy = torch.zeros(NEW_NUM_ACTIONS, dtype=torch.float32)
        valid_indices = torch.nonzero(original_mask, as_tuple=False).flatten()
        policy[valid_indices] = 1.0 / valid_indices.numel()
        mirrored_policy = mirror_flat_policy_targets(
            policy.unsqueeze(0),
            torch.tensor([current_piece.piece_type], dtype=torch.long),
        )[0]
        assert bool(torch.all(mirrored_policy[~expected_mask] == 0))


def test_mirror_maps_are_bijections_on_piece_valid_action_sets() -> None:
    for piece_type, valid_indices in enumerate(FLAT_VALID_INDICES_BY_PIECE):
        mask = torch.zeros((1, NEW_NUM_ACTIONS), dtype=torch.bool)
        mask[0, valid_indices] = True
        mirrored_mask = mirror_flat_action_masks(
            mask,
            torch.tensor([piece_type], dtype=torch.long),
        )[0]
        target_piece = MIRROR_PIECE_TYPE_ORDER[piece_type]
        expected_mask = torch.zeros(NEW_NUM_ACTIONS, dtype=torch.bool)
        expected_mask[FLAT_VALID_INDICES_BY_PIECE[target_piece]] = True
        assert torch.equal(mirrored_mask, expected_mask)


def test_maybe_mirror_training_tensors_is_noop_for_legacy_width() -> None:
    boards = torch.zeros((2, 1, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.bool)
    aux = torch.zeros((2, 80), dtype=torch.float32)
    aux[:, 0] = 1.0
    policy_targets = torch.zeros((2, LEGACY_NUM_ACTIONS), dtype=torch.float32)
    action_masks = torch.zeros((2, LEGACY_NUM_ACTIONS), dtype=torch.bool)
    action_masks[:, 0] = True
    policy_targets[:, 0] = 1.0

    mirrored = maybe_mirror_training_tensors(
        boards,
        aux,
        policy_targets,
        action_masks,
        probability=1.0,
    )

    assert torch.equal(mirrored[0], boards)
    assert torch.equal(mirrored[1], aux)
    assert torch.equal(mirrored[2], policy_targets)
    assert torch.equal(mirrored[3], action_masks)
