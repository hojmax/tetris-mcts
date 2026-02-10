from pathlib import Path

import numpy as np
import torch

import tetris_core
from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    QUEUE_SIZE,
)
from tetris_mcts.ml.network import ConvBackbone, HeadsModel, TetrisNet
from tetris_mcts.ml.weights import export_onnx, export_split_models


def _encode_state_python(
    env: tetris_core.TetrisEnv,
    move_number: int,
    max_moves: int,
) -> tuple[np.ndarray, np.ndarray]:
    board = np.array(
        [1.0 if cell != 0 else 0.0 for row in env.get_board() for cell in row],
        dtype=np.float32,
    )

    current = env.get_current_piece()
    hold = env.get_hold_piece()
    queue = env.get_queue(QUEUE_SIZE)

    aux: list[float] = []

    current_piece = 0 if current is None else current.piece_type
    for piece_type in range(NUM_PIECE_TYPES):
        aux.append(1.0 if piece_type == current_piece else 0.0)

    hold_piece = None if hold is None else hold.piece_type
    for piece_type in range(NUM_PIECE_TYPES):
        aux.append(1.0 if hold_piece == piece_type else 0.0)
    aux.append(1.0 if hold_piece is None else 0.0)

    aux.append(1.0 if not env.is_hold_used() else 0.0)

    for slot in range(QUEUE_SIZE):
        queue_piece = queue[slot] if slot < len(queue) else None
        for piece_type in range(NUM_PIECE_TYPES):
            aux.append(1.0 if queue_piece == piece_type else 0.0)

    aux.append(float(move_number) / float(max_moves))

    return board, np.asarray(aux, dtype=np.float32)


def _python_masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_logits = np.where(mask, logits, -np.inf).astype(np.float32)
    probs = torch.softmax(torch.from_numpy(masked_logits), dim=0)
    return probs.numpy()


def _build_env_variants() -> list[tuple[tetris_core.TetrisEnv, int, int]]:
    env_a = tetris_core.TetrisEnv.with_seed(10, 20, 1)
    env_b = tetris_core.TetrisEnv.with_seed(10, 20, 2)
    env_b.hold()
    env_c = tetris_core.TetrisEnv.with_seed(10, 20, 3)
    env_c.rotate_cw()
    env_c.move_left()
    env_c.hard_drop()
    env_d = tetris_core.TetrisEnv.with_seed(10, 20, 4)
    env_d.rotate_ccw()
    env_d.move_right()
    env_d.hard_drop()
    env_d.hold()
    env_d.rotate_cw()

    return [
        (env_a, 0, 100),
        (env_b, 1, 100),
        (env_c, 37, 100),
        (env_d, 99, 100),
    ]


def test_encode_state_matches_between_rust_inference_and_python_training_view() -> None:
    for env, move_number, max_moves in _build_env_variants():
        rust_board, rust_aux = tetris_core.debug_encode_state(
            env, move_number, max_moves
        )
        py_board, py_aux = _encode_state_python(env, move_number, max_moves)

        np.testing.assert_array_equal(
            np.asarray(rust_board, dtype=np.float32), py_board
        )
        np.testing.assert_array_equal(np.asarray(rust_aux, dtype=np.float32), py_aux)


def test_masked_softmax_matches_between_rust_and_python() -> None:
    rng = np.random.default_rng(123)
    logits = rng.normal(size=NUM_ACTIONS).astype(np.float32)
    mask = rng.random(NUM_ACTIONS) > 0.2
    mask[0] = True

    rust_probs = np.asarray(
        tetris_core.debug_masked_softmax(logits.tolist(), mask.tolist()),
        dtype=np.float32,
    )
    py_probs = _python_masked_softmax(logits, mask)

    np.testing.assert_allclose(rust_probs, py_probs, rtol=1e-6, atol=1e-7)
    assert np.all(rust_probs[~mask] == 0.0)
    np.testing.assert_allclose(np.sum(rust_probs), 1.0, rtol=1e-6, atol=1e-7)


def test_pytorch_and_rust_tract_inference_match_on_same_onnx(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = TetrisNet(
        conv_filters=[4, 8],
        fc_hidden=128,
        conv_kernel_size=3,
        conv_padding=1,
    )
    model.eval()

    onnx_path = tmp_path / "parity_model.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    for env, move_number, max_moves in _build_env_variants():
        board, aux = _encode_state_python(env, move_number, max_moves)
        action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)

        board_tensor = torch.from_numpy(board).reshape(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)
        mask_tensor = torch.from_numpy(action_mask).reshape(1, -1)

        with torch.no_grad():
            policy_logits, value_tensor = model(board_tensor, aux_tensor)
            masked_logits = policy_logits.masked_fill(~mask_tensor, float("-inf"))
            expected_policy = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()
            expected_value = value_tensor.item()

        rust_policy, rust_value = tetris_core.debug_predict_masked_from_tensors(
            str(onnx_path),
            board.tolist(),
            aux.tolist(),
            action_mask.tolist(),
        )

        np.testing.assert_allclose(
            np.asarray(rust_policy, dtype=np.float32),
            expected_policy.astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            float(rust_value), float(expected_value), rtol=1e-4, atol=1e-5
        )


def test_split_model_matches_end_to_end_pytorch(tmp_path: Path) -> None:
    """Verify split computation (conv + manual FC + heads) matches TetrisNet.forward()."""
    rng = np.random.default_rng(42)

    for seed in range(10):
        torch.manual_seed(seed)
        model = TetrisNet(
            conv_filters=[4, 8],
            fc_hidden=128,
            conv_kernel_size=3,
            conv_padding=1,
        )
        model.eval()

        conv_backbone = ConvBackbone(model)
        conv_backbone.eval()
        heads = HeadsModel(model)
        heads.eval()

        # Extract FC weight/bias split
        fc_weight = model.fc1.weight.detach()  # (128, 1652)
        fc_bias = model.fc1.bias.detach()  # (128,)
        w_board = fc_weight[:, :1600]  # (128, 1600)
        w_aux = fc_weight[:, 1600:]  # (128, 52)

        # Generate random inputs
        for _ in range(20):
            board = torch.from_numpy(
                rng.integers(0, 2, size=(1, 1, BOARD_HEIGHT, BOARD_WIDTH)).astype(
                    np.float32
                )
            )
            aux = torch.from_numpy(rng.standard_normal((1, 52)).astype(np.float32))

            with torch.no_grad():
                # End-to-end
                expected_logits, expected_value = model(board, aux)

                # Split path
                conv_out = conv_backbone(board)  # (1, 1600)
                board_embed = (w_board @ conv_out.squeeze(0)) + fc_bias  # (128,)
                fc_out = board_embed + (w_aux @ aux.squeeze(0))  # (128,)
                split_logits, split_value = heads(fc_out.unsqueeze(0))

            np.testing.assert_allclose(
                split_logits.numpy(),
                expected_logits.numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Policy logits mismatch (seed={seed})",
            )
            np.testing.assert_allclose(
                split_value.numpy(),
                expected_value.numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Value mismatch (seed={seed})",
            )


def test_split_onnx_rust_matches_end_to_end_pytorch(tmp_path: Path) -> None:
    """Verify Rust split ONNX inference matches PyTorch end-to-end on real game states."""
    torch.manual_seed(99)
    model = TetrisNet(
        conv_filters=[4, 8],
        fc_hidden=128,
        conv_kernel_size=3,
        conv_padding=1,
    )
    model.eval()

    onnx_path = tmp_path / "split_parity.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    # Build diverse game states
    envs: list[tuple[tetris_core.TetrisEnv, int, int]] = _build_env_variants()
    # Add more states with different seeds and more drops
    for game_seed in range(10, 30):
        env = tetris_core.TetrisEnv.with_seed(10, 20, game_seed)
        for _ in range(game_seed % 5):
            env.hard_drop()
        if game_seed % 3 == 0:
            env.hold()
        envs.append((env, game_seed % 100, 100))

    for env, move_number, max_moves in envs:
        board, aux = _encode_state_python(env, move_number, max_moves)
        action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)
        if not action_mask.any():
            continue

        # PyTorch end-to-end
        board_tensor = torch.from_numpy(board).reshape(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)
        mask_tensor = torch.from_numpy(action_mask).reshape(1, -1)

        with torch.no_grad():
            policy_logits, value_tensor = model(board_tensor, aux_tensor)
            masked_logits = policy_logits.masked_fill(~mask_tensor, float("-inf"))
            expected_policy = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()
            expected_value = value_tensor.item()

        # Rust split inference
        rust_policy, rust_value = tetris_core.debug_predict_masked_from_tensors(
            str(onnx_path),
            board.tolist(),
            aux.tolist(),
            action_mask.tolist(),
        )

        np.testing.assert_allclose(
            np.asarray(rust_policy, dtype=np.float32),
            expected_policy.astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Policy mismatch (seed={move_number})",
        )
        np.testing.assert_allclose(
            float(rust_value),
            float(expected_value),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Value mismatch (seed={move_number})",
        )


def _piece_state(piece: object | None) -> tuple[int, int, int, int] | None:
    if piece is None:
        return None
    return (piece.piece_type, piece.x, piece.y, piece.rotation)


def _assert_env_states_equal(
    lhs: tetris_core.TetrisEnv, rhs: tetris_core.TetrisEnv
) -> None:
    assert lhs.width == rhs.width
    assert lhs.height == rhs.height
    assert lhs.lines_cleared == rhs.lines_cleared
    assert lhs.game_over == rhs.game_over
    assert lhs.attack == rhs.attack
    assert lhs.combo == rhs.combo
    assert lhs.back_to_back == rhs.back_to_back

    assert lhs.get_board() == rhs.get_board()
    assert lhs.get_board_piece_types() == rhs.get_board_piece_types()
    assert _piece_state(lhs.get_current_piece()) == _piece_state(
        rhs.get_current_piece()
    )
    assert _piece_state(lhs.get_hold_piece()) == _piece_state(rhs.get_hold_piece())
    assert lhs.is_hold_used() == rhs.is_hold_used()
    assert lhs.get_pieces_spawned() == rhs.get_pieces_spawned()
    assert lhs.get_queue_len() == rhs.get_queue_len()
    assert lhs.get_queue(lhs.get_queue_len()) == rhs.get_queue(rhs.get_queue_len())
    assert lhs.get_possible_next_pieces() == rhs.get_possible_next_pieces()


def _assert_tree_exports_equal(
    lhs: tetris_core.MCTSTreeExport, rhs: tetris_core.MCTSTreeExport
) -> None:
    assert lhs.root_id == rhs.root_id
    assert lhs.num_simulations == rhs.num_simulations
    assert lhs.selected_action == rhs.selected_action
    np.testing.assert_array_equal(
        np.asarray(lhs.policy, dtype=np.float32),
        np.asarray(rhs.policy, dtype=np.float32),
    )

    assert len(lhs.nodes) == len(rhs.nodes)
    for lhs_node, rhs_node in zip(lhs.nodes, rhs.nodes, strict=True):
        assert lhs_node.id == rhs_node.id
        assert lhs_node.node_type == rhs_node.node_type
        assert lhs_node.visit_count == rhs_node.visit_count
        assert lhs_node.value_sum == rhs_node.value_sum
        assert lhs_node.mean_value == rhs_node.mean_value
        np.testing.assert_array_equal(
            np.asarray(lhs_node.value_history, dtype=np.float32),
            np.asarray(rhs_node.value_history, dtype=np.float32),
        )
        assert lhs_node.nn_value == rhs_node.nn_value
        assert lhs_node.is_terminal == rhs_node.is_terminal
        assert lhs_node.move_number == rhs_node.move_number
        assert lhs_node.attack == rhs_node.attack
        assert lhs_node.parent_id == rhs_node.parent_id
        assert lhs_node.edge_from_parent == rhs_node.edge_from_parent
        assert lhs_node.children == rhs_node.children
        assert lhs_node.valid_actions == rhs_node.valid_actions
        np.testing.assert_array_equal(
            np.asarray(lhs_node.action_priors, dtype=np.float32),
            np.asarray(rhs_node.action_priors, dtype=np.float32),
        )
        _assert_env_states_equal(lhs_node.state, rhs_node.state)


def test_mcts_tree_cache_parity_matches_uncached_search(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    model = TetrisNet(
        conv_filters=[4, 8],
        fc_hidden=128,
        conv_kernel_size=3,
        conv_padding=1,
    )
    model.eval()

    onnx_path = tmp_path / "mcts_cache_parity.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    env = tetris_core.TetrisEnv.with_seed(10, 20, 4242)
    for _ in range(5):
        env.rotate_cw()
        env.move_left()
        env.hard_drop()
    env.hold()
    env.rotate_ccw()
    env.move_right()

    config = tetris_core.MCTSConfig()
    config.num_simulations = 256
    config.c_puct = 1.5
    config.temperature = 1.0
    config.dirichlet_alpha = 0.15
    config.dirichlet_epsilon = 0.25
    config.seed = 999
    config.max_moves = 100
    config.track_value_history = True
    config.death_penalty = 0.0

    cached_agent = tetris_core.MCTSAgent(config)
    assert cached_agent.load_model(str(onnx_path))
    assert cached_agent.set_board_cache_enabled(True)

    warmup = cached_agent.search_with_tree(
        env.clone_state(), add_noise=False, move_number=17
    )
    assert warmup is not None
    warmup_stats = cached_agent.get_and_reset_cache_stats()
    assert warmup_stats is not None

    cached_search = cached_agent.search_with_tree(
        env.clone_state(), add_noise=False, move_number=17
    )
    assert cached_search is not None
    cached_result, cached_tree = cached_search
    cached_stats = cached_agent.get_and_reset_cache_stats()
    assert cached_stats is not None
    cache_hits, cache_misses, cache_size = cached_stats
    assert cache_hits > 0
    assert cache_misses == 0
    assert cache_size > 0

    uncached_agent = tetris_core.MCTSAgent(config)
    assert uncached_agent.load_model(str(onnx_path))
    assert uncached_agent.set_board_cache_enabled(False)
    uncached_search = uncached_agent.search_with_tree(
        env.clone_state(), add_noise=False, move_number=17
    )
    assert uncached_search is not None
    uncached_result, uncached_tree = uncached_search
    uncached_stats = uncached_agent.get_and_reset_cache_stats()
    assert uncached_stats is not None
    uncached_hits, uncached_misses, uncached_size = uncached_stats
    assert uncached_hits == 0
    assert uncached_misses == 0
    assert uncached_size == 0

    assert cached_result.action == uncached_result.action
    assert cached_result.value == uncached_result.value
    assert cached_result.num_simulations == uncached_result.num_simulations
    np.testing.assert_array_equal(
        np.asarray(cached_result.policy, dtype=np.float32),
        np.asarray(uncached_result.policy, dtype=np.float32),
    )

    _assert_tree_exports_equal(cached_tree, uncached_tree)
