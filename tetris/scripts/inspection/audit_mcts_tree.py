"""Audit MCTS tree structure and environment transitions step-by-step."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import structlog
from pydantic.dataclasses import dataclass
from simple_parsing import field as sp_field
from simple_parsing import parse

from tetris_core import MCTSAgent, MCTSConfig, TetrisEnv
from tetris.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    PIECE_NAMES,
    QUEUE_SIZE,
)
from tetris.ml.config import TrainingConfig

logger = structlog.get_logger()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
HOLD_ACTION_INDEX = NUM_ACTIONS - 1


def piece_to_dict(piece: object | None) -> dict | None:
    """Convert optional piece object to a JSON-serializable dict."""
    if piece is None:
        return None
    return {
        "piece_type": int(piece.piece_type),
        "piece_name": PIECE_NAMES[int(piece.piece_type)],
        "x": int(piece.x),
        "y": int(piece.y),
        "rotation": int(piece.rotation),
    }


def piece_key(piece: object | None) -> tuple[int, int, int, int] | None:
    """Create comparable tuple key for optional piece object."""
    if piece is None:
        return None
    return (
        int(piece.piece_type),
        int(piece.x),
        int(piece.y),
        int(piece.rotation),
    )


def board_hash(board: list[list[int]]) -> str:
    """Stable hash for board occupancy."""
    as_bytes = bytes(cell for row in board for cell in row)
    return hashlib.sha256(as_bytes).hexdigest()[:16]


def get_lines_and_attack_from_state(env: TetrisEnv) -> tuple[int, int]:
    """Extract line-clear and attack for the state's most recent placement."""
    attack_result = env.get_last_attack_result()
    if attack_result is None:
        return 0, 0
    return int(attack_result.lines_cleared), int(attack_result.total_attack)


def get_state_snapshot(env: TetrisEnv) -> dict:
    """Small readable state snapshot for logging."""
    board = env.get_board()
    return {
        "board_hash": board_hash(board),
        "queue": [int(x) for x in env.get_queue(QUEUE_SIZE)],
        "queue_len": int(env.get_queue_len()),
        "current_piece": piece_to_dict(env.get_current_piece()),
        "hold_piece": piece_to_dict(env.get_hold_piece()),
        "hold_available": bool(not env.is_hold_used()),
        "pieces_spawned": int(env.get_pieces_spawned()),
    }


def safe_float(value: float) -> float:
    """Round for compact logs while preserving enough precision."""
    return float(round(float(value), 6))


def check_and_record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Append a failure message when condition is False."""
    if not condition:
        failures.append(message)


def compute_policy_entropy(policy: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    import math

    entropy = 0.0
    for p in policy:
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def sorted_piece_counts(counts_by_piece: dict[int, int]) -> list[dict]:
    """Convert piece->count mapping into sorted JSON rows."""
    rows: list[dict] = []
    for piece, count in sorted(counts_by_piece.items(), key=lambda row: row[0]):
        rows.append(
            {
                "piece": int(piece),
                "piece_name": PIECE_NAMES[int(piece)],
                "count": int(count),
            }
        )
    return rows


def run_tree_math_audit(
    nodes: list[object],
    root_id: int,
    args: "ScriptArgs",
    failures: list[str],
) -> dict:
    """Validate tree-level accounting and chance-node behavior."""
    node_by_id: dict[int, object] = {}
    decision_nodes = 0
    chance_nodes = 0
    decision_backup_mismatch_count = 0
    reward_attack_mismatch_count = 0
    chance_uniform_violation_count = 0
    chance_weight_rows: list[dict] = []

    # Node ID and mean-value consistency checks.
    for idx, node in enumerate(nodes):
        node_id = int(node.id)
        node_by_id[node_id] = node
        check_and_record(
            node_id == idx,
            f"Node index/id mismatch: idx={idx} node.id={node_id}",
            failures,
        )

        if int(node.visit_count) > 0:
            recomputed_mean = float(node.value_sum) / float(node.visit_count)
            check_and_record(
                abs(recomputed_mean - float(node.mean_value))
                <= args.value_mean_tolerance,
                (
                    f"Node {node_id} mean mismatch: exported={node.mean_value} "
                    f"recomputed={recomputed_mean}"
                ),
                failures,
            )
        else:
            check_and_record(
                abs(float(node.mean_value)) <= args.value_mean_tolerance,
                f"Node {node_id} has zero visits but non-zero mean {node.mean_value}",
                failures,
            )

    check_and_record(
        int(root_id) in node_by_id,
        f"Root id {root_id} not present in exported nodes",
        failures,
    )

    # Topology + backup/value accounting checks.
    for node in nodes:
        node_id = int(node.id)
        node_type = str(node.node_type)

        # Parent pointers and edge labels.
        for child_id_raw in node.children:
            child_id = int(child_id_raw)
            check_and_record(
                child_id in node_by_id,
                f"Node {node_id} references missing child {child_id}",
                failures,
            )
            if child_id not in node_by_id:
                continue

            child = node_by_id[child_id]
            check_and_record(
                child.parent_id is not None and int(child.parent_id) == node_id,
                (
                    f"Parent pointer mismatch: parent={node_id} child={child_id} "
                    f"child.parent_id={child.parent_id}"
                ),
                failures,
            )

            if node_type == "decision":
                check_and_record(
                    str(child.node_type) == "chance",
                    (
                        f"Decision node {node_id} has non-chance child {child_id} "
                        f"type={child.node_type}"
                    ),
                    failures,
                )
                if child.edge_from_parent is not None:
                    check_and_record(
                        int(child.edge_from_parent)
                        in {int(x) for x in node.valid_actions},
                        (
                            f"Decision node {node_id} child {child_id} edge "
                            f"{child.edge_from_parent} not in valid_actions"
                        ),
                        failures,
                    )
            elif node_type == "chance":
                check_and_record(
                    str(child.node_type) == "decision",
                    (
                        f"Chance node {node_id} has non-decision child {child_id} "
                        f"type={child.node_type}"
                    ),
                    failures,
                )
                possible_pieces = {
                    int(x) for x in node.state.get_possible_next_pieces()
                }
                if child.edge_from_parent is not None:
                    check_and_record(
                        int(child.edge_from_parent) in possible_pieces,
                        (
                            f"Chance node {node_id} child {child_id} piece "
                            f"{child.edge_from_parent} not in possible pieces "
                            f"{sorted(possible_pieces)}"
                        ),
                        failures,
                    )

        if node_type == "decision":
            decision_nodes += 1
            child_visits = 0
            child_value_sum = 0.0
            for child_id_raw in node.children:
                child_id = int(child_id_raw)
                if child_id not in node_by_id:
                    continue
                child = node_by_id[child_id]
                child_visits += int(child.visit_count)
                child_value_sum += float(child.value_sum)

            if int(node.visit_count) > 0:
                visits_ok = child_visits == int(node.visit_count)
                value_ok = (
                    abs(child_value_sum - float(node.value_sum))
                    <= args.value_sum_tolerance
                )
                if not visits_ok or not value_ok:
                    decision_backup_mismatch_count += 1
                check_and_record(
                    visits_ok,
                    (
                        f"Decision node {node_id} visit accounting mismatch: "
                        f"node={node.visit_count} child_sum={child_visits}"
                    ),
                    failures,
                )
                check_and_record(
                    value_ok,
                    (
                        f"Decision node {node_id} value accounting mismatch: "
                        f"node={node.value_sum} child_sum={child_value_sum}"
                    ),
                    failures,
                )
            else:
                check_and_record(
                    child_visits == 0,
                    (
                        f"Decision node {node_id} has zero visits but "
                        f"child visits={child_visits}"
                    ),
                    failures,
                )
                check_and_record(
                    abs(child_value_sum) <= args.value_sum_tolerance,
                    (
                        f"Decision node {node_id} has zero visits but "
                        f"child value sum={child_value_sum}"
                    ),
                    failures,
                )
        elif node_type == "chance":
            chance_nodes += 1

            reward_delta = abs(float(node.reward) - float(node.attack))
            reward_ok = reward_delta <= args.reward_attack_tolerance
            if not reward_ok:
                reward_attack_mismatch_count += 1
            check_and_record(
                reward_ok,
                (
                    f"Chance node {node_id} reward/attack mismatch: "
                    f"reward={node.reward} attack={node.attack}"
                ),
                failures,
            )

            possible_pieces = sorted(
                int(x) for x in node.state.get_possible_next_pieces()
            )
            piece_visit_counts: dict[int, int] = {piece: 0 for piece in possible_pieces}
            for child_id_raw in node.children:
                child_id = int(child_id_raw)
                if child_id not in node_by_id:
                    continue
                child = node_by_id[child_id]
                piece_edge = child.edge_from_parent
                if piece_edge is None:
                    continue
                piece = int(piece_edge)
                if piece in piece_visit_counts:
                    piece_visit_counts[piece] += int(child.visit_count)

            accounted_visits = sum(piece_visit_counts.values())
            unaccounted_visits = int(node.visit_count) - accounted_visits
            check_and_record(
                unaccounted_visits >= 0,
                (
                    f"Chance node {node_id} has negative unaccounted visits: "
                    f"node_visits={node.visit_count} accounted={accounted_visits}"
                ),
                failures,
            )

            expected_uniform_count = (
                (float(node.visit_count) / float(len(possible_pieces)))
                if possible_pieces
                else 0.0
            )
            max_abs_deviation = 0.0
            max_rel_deviation = 0.0
            if expected_uniform_count > 0.0:
                max_abs_deviation = max(
                    abs(float(piece_visit_counts[piece]) - expected_uniform_count)
                    for piece in possible_pieces
                )
                max_rel_deviation = max_abs_deviation / expected_uniform_count

            should_check_uniform = (
                args.check_uniform_piece_sampling
                and len(possible_pieces) > 1
                and int(node.visit_count) >= args.min_visits_for_uniform_check
                and unaccounted_visits == 0
            )
            if should_check_uniform:
                uniform_ok = max_rel_deviation <= args.max_uniform_rel_deviation
                if not uniform_ok:
                    chance_uniform_violation_count += 1
                check_and_record(
                    uniform_ok,
                    (
                        f"Chance node {node_id} non-uniform piece sampling: "
                        f"max_rel_deviation={max_rel_deviation:.3f} "
                        f"threshold={args.max_uniform_rel_deviation:.3f} "
                        f"piece_visits={piece_visit_counts}"
                    ),
                    failures,
                )

            chance_weight_rows.append(
                {
                    "node_id": int(node_id),
                    "visit_count": int(node.visit_count),
                    "reward": safe_float(node.reward),
                    "attack": int(node.attack),
                    "reward_attack_delta": safe_float(reward_delta),
                    "possible_pieces": possible_pieces,
                    "piece_visit_counts": sorted_piece_counts(piece_visit_counts),
                    "accounted_piece_visits": int(accounted_visits),
                    "unaccounted_visits": int(unaccounted_visits),
                    "expected_uniform_count": safe_float(expected_uniform_count),
                    "max_abs_deviation": safe_float(max_abs_deviation),
                    "max_rel_deviation": safe_float(max_rel_deviation),
                }
            )

    chance_weight_rows.sort(
        key=lambda row: (row["visit_count"], row["max_rel_deviation"], -row["node_id"]),
        reverse=True,
    )

    return {
        "num_nodes": int(len(nodes)),
        "num_decision_nodes": int(decision_nodes),
        "num_chance_nodes": int(chance_nodes),
        "decision_backup_mismatch_count": int(decision_backup_mismatch_count),
        "reward_attack_mismatch_count": int(reward_attack_mismatch_count),
        "chance_uniform_violation_count": int(chance_uniform_violation_count),
        "top_chance_weight_nodes": chance_weight_rows[: args.top_k_actions],
    }


def analyze_all_valid_actions(
    env: TetrisEnv,
    valid_actions: list[int],
) -> list[dict]:
    """Execute each valid action in a cloned env and capture immediate outcomes."""
    action_rows: list[dict] = []
    for action_idx in valid_actions:
        clone_env = env.clone_state()
        attack = clone_env.execute_action_index(int(action_idx))
        if attack is None:
            action_rows.append(
                {
                    "action": int(action_idx),
                    "is_executable": False,
                    "attack": 0,
                    "lines": 0,
                }
            )
            continue

        lines, total_attack = get_lines_and_attack_from_state(clone_env)
        action_rows.append(
            {
                "action": int(action_idx),
                "is_executable": True,
                "attack": int(attack),
                "lines": int(lines),
                "total_attack": int(total_attack),
            }
        )

    return action_rows


def serialize_tree_node(node: object) -> dict:
    """Convert exported tree node to a compact JSON dict."""
    node_state = node.state
    lines, total_attack = get_lines_and_attack_from_state(node_state)
    return {
        "id": int(node.id),
        "node_type": str(node.node_type),
        "parent_id": None if node.parent_id is None else int(node.parent_id),
        "edge_from_parent": None
        if node.edge_from_parent is None
        else int(node.edge_from_parent),
        "visit_count": int(node.visit_count),
        "value_sum": safe_float(node.value_sum),
        "mean_value": safe_float(node.mean_value),
        "nn_value": safe_float(node.nn_value),
        "prior": safe_float(node.prior),
        "attack": int(node.attack),
        "reward": safe_float(node.reward),
        "lines_cleared": int(lines),
        "total_attack": int(total_attack),
        "children": [int(x) for x in node.children],
        "num_children": int(len(node.children)),
        "num_valid_actions": int(len(node.valid_actions)),
        "state": get_state_snapshot(node_state),
    }


def write_json(path: Path, data: dict) -> None:
    """Write JSON file with stable formatting."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def run_seed_audit(
    agent: MCTSAgent,
    args: "ScriptArgs",
    seed: int,
    seed_dir: Path,
) -> dict:
    """Audit one deterministic game seed."""
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed)

    moves_jsonl_path = seed_dir / "moves.jsonl"
    failures_jsonl_path = seed_dir / "failures.jsonl"
    if moves_jsonl_path.exists():
        moves_jsonl_path.unlink()
    if failures_jsonl_path.exists():
        failures_jsonl_path.unlink()

    total_lines = 0
    total_attack = 0
    total_failures = 0
    moves_played = 0
    selected_line_clear_moves = 0
    moves_with_any_line_option = 0
    moves_with_expanded_line_option = 0
    selected_attack_moves = 0
    moves_with_any_attack_option = 0
    moves_with_expanded_attack_option = 0
    decision_backup_mismatch_count = 0
    reward_attack_mismatch_count = 0
    chance_uniform_violation_count = 0
    chance_model_hidden_piece_counts: dict[tuple[int, ...], dict[int, int]] = (
        defaultdict(lambda: defaultdict(int))
    )

    logger.info("Auditing seed", seed=seed, max_placements=args.max_placements)

    frame_idx = 0
    placement_count = 0
    while placement_count < args.max_placements:
        if env.game_over:
            logger.info("Game over reached", seed=seed, move=frame_idx)
            break

        pre_snapshot = get_state_snapshot(env)

        result = agent.search_with_tree(
            env, add_noise=False, placement_count=placement_count
        )
        if result is None:
            logger.warning(
                "Search returned None (no valid action mask)",
                seed=seed,
                move=frame_idx,
            )
            break

        mcts_result, tree = result
        nodes = tree.nodes
        root = nodes[tree.root_id]
        failures: list[str] = []

        tree_math = run_tree_math_audit(
            nodes=nodes,
            root_id=int(tree.root_id),
            args=args,
            failures=failures,
        )
        decision_backup_mismatch_count += int(
            tree_math["decision_backup_mismatch_count"]
        )
        reward_attack_mismatch_count += int(tree_math["reward_attack_mismatch_count"])
        chance_uniform_violation_count += int(
            tree_math["chance_uniform_violation_count"]
        )

        check_and_record(
            root.node_type == "decision",
            f"Root node_type expected 'decision', got {root.node_type}",
            failures,
        )

        policy_sum = sum(float(x) for x in mcts_result.policy)
        check_and_record(
            abs(policy_sum - 1.0) < 1e-4,
            f"Policy sum expected 1.0, got {policy_sum}",
            failures,
        )

        if root.visit_count > 0:
            check_and_record(
                int(root.visit_count) == int(mcts_result.num_simulations),
                (
                    "Root visit count mismatch: "
                    f"root={root.visit_count} sims={mcts_result.num_simulations}"
                ),
                failures,
            )
            check_and_record(
                abs(float(mcts_result.value) - float(root.mean_value))
                <= args.value_mean_tolerance,
                (
                    "Root value mismatch: "
                    f"mcts_result.value={mcts_result.value} root.mean_value={root.mean_value}"
                ),
                failures,
            )

        prior_by_action = {
            int(action): float(prior)
            for action, prior in zip(root.valid_actions, root.action_priors)
        }

        root_children_by_action: dict[int, object] = {}
        root_action_rows: list[dict] = []
        chance_child_visit_sum = 0

        for child_id in root.children:
            check_and_record(
                0 <= int(child_id) < len(nodes),
                f"Root child id out of bounds: {child_id}",
                failures,
            )
            if not (0 <= int(child_id) < len(nodes)):
                continue

            child = nodes[child_id]
            check_and_record(
                child.node_type == "chance",
                f"Root child {child_id} expected chance node, got {child.node_type}",
                failures,
            )
            if child.node_type != "chance":
                continue

            action_idx = child.edge_from_parent
            check_and_record(
                action_idx is not None,
                f"Chance node {child_id} missing edge_from_parent action index",
                failures,
            )
            if action_idx is None:
                continue

            action = int(action_idx)
            root_children_by_action[action] = child
            chance_child_visit_sum += int(child.visit_count)

            # Re-execute action from pre-search env and compare to exported chance node.
            check_env = env.clone_state()
            step_attack = check_env.execute_action_index(action)
            check_and_record(
                step_attack is not None,
                f"Action {action} from root is not executable in env clone",
                failures,
            )

            sim_lines = 0
            sim_total_attack = 0
            true_hidden_piece: int | None = None
            possible_next_pieces = sorted(
                int(x) for x in child.state.get_possible_next_pieces()
            )
            if step_attack is not None:
                full_queue = [int(x) for x in check_env.get_queue(6)]
                if len(full_queue) >= 6:
                    true_hidden_piece = int(full_queue[5])
                    check_and_record(
                        true_hidden_piece in set(possible_next_pieces),
                        (
                            f"Action {action} true hidden piece {true_hidden_piece} "
                            f"not in possible set {possible_next_pieces}"
                        ),
                        failures,
                    )
                    chance_model_hidden_piece_counts[tuple(possible_next_pieces)][
                        true_hidden_piece
                    ] += 1

                sim_lines, sim_total_attack = get_lines_and_attack_from_state(check_env)
                check_and_record(
                    int(step_attack) == int(child.attack),
                    (
                        f"Action {action} attack mismatch: "
                        f"env={step_attack} tree={child.attack}"
                    ),
                    failures,
                )

                child_lines, child_total_attack = get_lines_and_attack_from_state(
                    child.state
                )
                check_and_record(
                    sim_lines == child_lines,
                    (
                        f"Action {action} lines mismatch: "
                        f"env={sim_lines} tree_state={child_lines}"
                    ),
                    failures,
                )
                check_and_record(
                    sim_total_attack == child_total_attack,
                    (
                        f"Action {action} total_attack mismatch: "
                        f"env={sim_total_attack} tree_state={child_total_attack}"
                    ),
                    failures,
                )

                check_env.truncate_queue(5)
                check_and_record(
                    check_env.get_board() == child.state.get_board(),
                    f"Action {action} board mismatch after execute+truncate",
                    failures,
                )
                check_and_record(
                    check_env.get_queue(5) == child.state.get_queue(5),
                    f"Action {action} queue mismatch after execute+truncate",
                    failures,
                )

            # Validate chance -> decision transitions for explored piece outcomes.
            possible_pieces = set(possible_next_pieces)
            for grandchild_id in child.children:
                check_and_record(
                    0 <= int(grandchild_id) < len(nodes),
                    f"Chance child id out of bounds: {grandchild_id}",
                    failures,
                )
                if not (0 <= int(grandchild_id) < len(nodes)):
                    continue

                decision_child = nodes[grandchild_id]
                check_and_record(
                    decision_child.node_type == "decision",
                    (
                        f"Chance node {child_id} child {grandchild_id} "
                        f"expected decision node, got {decision_child.node_type}"
                    ),
                    failures,
                )
                if decision_child.node_type != "decision":
                    continue

                piece_outcome = decision_child.edge_from_parent
                check_and_record(
                    piece_outcome is not None,
                    f"Decision node {grandchild_id} missing piece edge",
                    failures,
                )
                if piece_outcome is None:
                    continue

                piece = int(piece_outcome)
                check_and_record(
                    piece in possible_pieces,
                    (
                        f"Piece outcome {piece} on node {grandchild_id} "
                        "not in get_possible_next_pieces()"
                    ),
                    failures,
                )

                transition_env = child.state.clone_state()
                transition_env.push_queue_piece(piece)

                check_and_record(
                    transition_env.get_board() == decision_child.state.get_board(),
                    (
                        f"Chance->decision board mismatch for action {action}, "
                        f"piece {piece}"
                    ),
                    failures,
                )
                check_and_record(
                    transition_env.get_queue_len()
                    == decision_child.state.get_queue_len(),
                    (
                        f"Chance->decision queue_len mismatch for action {action}, "
                        f"piece {piece}"
                    ),
                    failures,
                )
                check_and_record(
                    transition_env.get_queue(5) == decision_child.state.get_queue(5),
                    (
                        f"Chance->decision visible queue mismatch for action {action}, "
                        f"piece {piece}"
                    ),
                    failures,
                )
                check_and_record(
                    piece_key(transition_env.get_current_piece())
                    == piece_key(decision_child.state.get_current_piece()),
                    (
                        f"Chance->decision current piece mismatch for action {action}, "
                        f"piece {piece}"
                    ),
                    failures,
                )
                check_and_record(
                    piece_key(transition_env.get_hold_piece())
                    == piece_key(decision_child.state.get_hold_piece()),
                    (
                        f"Chance->decision hold piece mismatch for action {action}, "
                        f"piece {piece}"
                    ),
                    failures,
                )

            child_lines, _child_total_attack = get_lines_and_attack_from_state(
                child.state
            )
            policy_prob = (
                float(mcts_result.policy[action])
                if 0 <= action < len(mcts_result.policy)
                else 0.0
            )
            row = {
                "action": action,
                "visit_count": int(child.visit_count),
                "mean_value": safe_float(child.mean_value),
                "value_sum": safe_float(child.value_sum),
                "nn_value": safe_float(child.nn_value),
                "prior": safe_float(prior_by_action.get(action, 0.0)),
                "policy_prob": safe_float(policy_prob),
                "attack": int(child.attack),
                "reward": safe_float(child.reward),
                "reward_attack_delta": safe_float(
                    abs(float(child.reward) - float(child.attack))
                ),
                "lines": int(child_lines),
                "q_minus_reward": safe_float(
                    float(child.mean_value) - float(child.reward)
                ),
                "q_minus_reward_minus_nn": safe_float(
                    float(child.mean_value)
                    - float(child.reward)
                    - float(child.nn_value)
                ),
                "piece_children": int(len(child.children)),
                "possible_next_pieces": sorted(int(x) for x in possible_pieces),
                "true_hidden_piece_from_env": true_hidden_piece,
            }
            root_action_rows.append(row)

        if root.visit_count > 0:
            check_and_record(
                chance_child_visit_sum == int(root.visit_count),
                (
                    "Root child visit sum mismatch: "
                    f"sum={chance_child_visit_sum} root={root.visit_count}"
                ),
                failures,
            )

        selected_action = int(mcts_result.action)
        check_and_record(
            selected_action in root_children_by_action,
            f"Selected action {selected_action} not present among root children",
            failures,
        )

        # Evaluate full valid action set to see if line clears were available but not selected.
        valid_action_rows: list[dict] = []
        if args.check_all_valid_actions:
            valid_action_rows = analyze_all_valid_actions(
                env,
                [int(x) for x in root.valid_actions],
            )
            for row in valid_action_rows:
                row["prior"] = safe_float(prior_by_action.get(int(row["action"]), 0.0))

        any_line_option = any(row.get("lines", 0) > 0 for row in valid_action_rows)
        if any_line_option:
            moves_with_any_line_option += 1

        expanded_line_option = any(row["lines"] > 0 for row in root_action_rows)
        if expanded_line_option:
            moves_with_expanded_line_option += 1

        any_attack_option = any(row.get("attack", 0) > 0 for row in valid_action_rows)
        if any_attack_option:
            moves_with_any_attack_option += 1

        expanded_attack_option = any(row["attack"] > 0 for row in root_action_rows)
        if expanded_attack_option:
            moves_with_expanded_attack_option += 1

        selected_child = root_children_by_action.get(selected_action)
        selected_tree_lines = 0
        selected_tree_attack = 0
        selected_tree_q = 0.0
        if selected_child is not None:
            selected_tree_lines, selected_tree_total_attack = (
                get_lines_and_attack_from_state(selected_child.state)
            )
            selected_tree_attack = int(selected_child.attack)
            check_and_record(
                selected_tree_attack == int(selected_tree_total_attack),
                (
                    f"Selected action {selected_action} attack mismatch between "
                    "chance.attack and chance.state.last_attack_result"
                ),
                failures,
            )
            selected_tree_q = float(selected_child.mean_value)

        # Execute selected action in live environment and verify against tree.
        live_attack = env.execute_action_index(selected_action)
        check_and_record(
            live_attack is not None,
            f"Selected action {selected_action} could not be executed in live env",
            failures,
        )
        if live_attack is None:
            break

        live_lines, live_total_attack = get_lines_and_attack_from_state(env)
        total_lines += int(live_lines)
        total_attack += int(live_attack)
        moves_played += 1

        check_and_record(
            int(live_attack) == int(live_total_attack),
            (
                f"Live env attack mismatch: execute returned {live_attack}, "
                f"last_attack_result.total_attack={live_total_attack}"
            ),
            failures,
        )

        if selected_child is not None:
            check_and_record(
                int(live_attack) == int(selected_tree_attack),
                (
                    f"Selected action {selected_action} attack mismatch: "
                    f"live={live_attack} tree={selected_tree_attack}"
                ),
                failures,
            )
            check_and_record(
                int(live_lines) == int(selected_tree_lines),
                (
                    f"Selected action {selected_action} lines mismatch: "
                    f"live={live_lines} tree={selected_tree_lines}"
                ),
                failures,
            )

        if live_lines > 0:
            selected_line_clear_moves += 1
        if live_attack > 0:
            selected_attack_moves += 1

        root_action_rows.sort(
            key=lambda row: (row["visit_count"], row["mean_value"], -row["action"]),
            reverse=True,
        )

        top_valid_line_row: dict | None = None
        if valid_action_rows:
            line_candidates = [
                row for row in valid_action_rows if row.get("lines", 0) > 0
            ]
            if line_candidates:
                line_candidates.sort(
                    key=lambda row: (
                        row["lines"],
                        row.get("attack", 0),
                        -row["action"],
                    ),
                    reverse=True,
                )
                top_valid_line_row = line_candidates[0]

        top_valid_attack_row: dict | None = None
        if valid_action_rows:
            attack_candidates = [
                row for row in valid_action_rows if row.get("attack", 0) > 0
            ]
            if attack_candidates:
                attack_candidates.sort(
                    key=lambda row: (
                        row["attack"],
                        row.get("lines", 0),
                        -row["action"],
                    ),
                    reverse=True,
                )
                top_valid_attack_row = attack_candidates[0]

        move_record = {
            "seed": int(seed),
            "move": int(frame_idx),
            "placement_count": int(placement_count),
            "pre_state": pre_snapshot,
            "search": {
                "num_nodes": int(len(nodes)),
                "num_root_children": int(len(root.children)),
                "root_visit_count": int(root.visit_count),
                "num_simulations": int(mcts_result.num_simulations),
                "selected_action": int(selected_action),
                "selected_policy_prob": safe_float(mcts_result.policy[selected_action]),
                "policy_entropy": safe_float(
                    compute_policy_entropy(mcts_result.policy)
                ),
                "root_value": safe_float(mcts_result.value),
            },
            "selected_action": {
                "action": int(selected_action),
                "live_attack": int(live_attack),
                "live_lines": int(live_lines),
                "tree_q": safe_float(selected_tree_q),
                "prior": safe_float(prior_by_action.get(selected_action, 0.0)),
                "tree_attack": int(selected_tree_attack),
                "tree_lines": int(selected_tree_lines),
            },
            "line_clear_opportunities": {
                "any_valid_line_action": bool(any_line_option),
                "any_expanded_line_action": bool(expanded_line_option),
                "selected_clears_line": bool(live_lines > 0),
                "valid_line_action_count": int(
                    sum(1 for row in valid_action_rows if row.get("lines", 0) > 0)
                ),
                "expanded_line_action_count": int(
                    sum(1 for row in root_action_rows if row["lines"] > 0)
                ),
                "top_valid_line_action": top_valid_line_row,
            },
            "attack_opportunities": {
                "any_valid_attack_action": bool(any_attack_option),
                "any_expanded_attack_action": bool(expanded_attack_option),
                "selected_scores_attack": bool(live_attack > 0),
                "valid_attack_action_count": int(
                    sum(1 for row in valid_action_rows if row.get("attack", 0) > 0)
                ),
                "expanded_attack_action_count": int(
                    sum(1 for row in root_action_rows if row["attack"] > 0)
                ),
                "top_valid_attack_action": top_valid_attack_row,
            },
            "tree_math_audit": tree_math,
            "root_actions_topk": root_action_rows[: args.top_k_actions],
            "checks": {
                "num_failures": int(len(failures)),
                "failures": failures,
            },
            "post_state": get_state_snapshot(env),
        }

        with moves_jsonl_path.open("a") as f:
            f.write(json.dumps(move_record) + "\n")

        if failures:
            with failures_jsonl_path.open("a") as f:
                for failure in failures:
                    payload = {
                        "seed": int(seed),
                        "move": int(frame_idx),
                        "failure": failure,
                    }
                    f.write(json.dumps(payload) + "\n")

        if args.dump_full_tree:
            tree_payload = {
                "seed": int(seed),
                "move": int(frame_idx),
                "selected_action": int(selected_action),
                "num_nodes": int(len(nodes)),
                "num_simulations": int(mcts_result.num_simulations),
                "root_id": int(tree.root_id),
                "policy": [safe_float(x) for x in mcts_result.policy],
                "nodes": [serialize_tree_node(node) for node in nodes],
            }
            write_json(seed_dir / f"tree_move_{frame_idx:03d}.json", tree_payload)

        total_failures += len(failures)

        logger.info(
            "Audited move",
            seed=seed,
            move=frame_idx,
            selected_action=selected_action,
            selected_lines=int(live_lines),
            selected_attack=int(live_attack),
            root_children=int(len(root.children)),
            num_nodes=int(len(nodes)),
            failures=int(len(failures)),
        )
        if selected_action != HOLD_ACTION_INDEX:
            placement_count += 1
        frame_idx += 1

    seed_summary = {
        "seed": int(seed),
        "moves_played": int(moves_played),
        "total_lines": int(total_lines),
        "total_attack": int(total_attack),
        "selected_line_clear_moves": int(selected_line_clear_moves),
        "moves_with_any_line_option": int(moves_with_any_line_option),
        "moves_with_expanded_line_option": int(moves_with_expanded_line_option),
        "selected_attack_moves": int(selected_attack_moves),
        "moves_with_any_attack_option": int(moves_with_any_attack_option),
        "moves_with_expanded_attack_option": int(moves_with_expanded_attack_option),
        "total_failures": int(total_failures),
        "line_clear_selection_rate": safe_float(
            (selected_line_clear_moves / moves_played) if moves_played > 0 else 0.0
        ),
        "line_option_rate": safe_float(
            (moves_with_any_line_option / moves_played) if moves_played > 0 else 0.0
        ),
        "attack_selection_rate": safe_float(
            (selected_attack_moves / moves_played) if moves_played > 0 else 0.0
        ),
        "attack_option_rate": safe_float(
            (moves_with_any_attack_option / moves_played) if moves_played > 0 else 0.0
        ),
        "decision_backup_mismatch_count": int(decision_backup_mismatch_count),
        "reward_attack_mismatch_count": int(reward_attack_mismatch_count),
        "chance_uniform_violation_count": int(chance_uniform_violation_count),
        "chance_model_hidden_piece_counts": [
            {
                "possible_pieces": [int(piece) for piece in key],
                "observed_hidden_piece_counts": sorted_piece_counts(
                    {int(piece): int(count) for piece, count in counts.items()}
                ),
                "num_observations": int(sum(counts.values())),
            }
            for key, counts in sorted(
                chance_model_hidden_piece_counts.items(),
                key=lambda row: (len(row[0]), row[0]),
            )
        ],
    }

    write_json(seed_dir / "summary.json", seed_summary)
    return seed_summary


def build_mcts_config(args: "ScriptArgs") -> MCTSConfig:
    """Build MCTSConfig from script args."""
    config = MCTSConfig()
    config.num_simulations = args.num_simulations
    config.c_puct = args.c_puct
    config.temperature = args.temperature
    config.dirichlet_alpha = args.dirichlet_alpha
    config.dirichlet_epsilon = args.dirichlet_epsilon
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed
    return config


def get_seeds(args: "ScriptArgs") -> list[int]:
    """Resolve list of seeds from explicit list or range args."""
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed_start + offset) for offset in range(args.num_seeds)]


@dataclass
class ScriptArgs:
    model_path: Path = sp_field(positional=True)  # Path to ONNX model to audit
    output_dir: Path = (
        Path(__file__).parent / "outputs" / "mcts_tree_audits"
    )  # Directory for audit artifacts
    run_name: str = ""  # Optional explicit run directory name (empty = auto-generated)
    max_placements: int = 30  # Maximum placements to audit per seed
    seeds: list[int] = sp_field(
        default_factory=list
    )  # Explicit seed list (overrides seed_start/num_seeds)
    seed_start: int = 0  # First seed when seeds list is empty
    num_seeds: int = 3  # Number of sequential seeds when seeds list is empty
    num_simulations: int = 400  # MCTS simulations per audited move
    c_puct: float = (  # PUCT exploration constant
        DEFAULT_TRAINING_CONFIG.self_play.c_puct
    )
    temperature: float = 0.0  # Action selection temperature (0 = argmax)
    dirichlet_alpha: float = (
        DEFAULT_TRAINING_CONFIG.self_play.dirichlet_alpha
    )  # Dirichlet alpha (unused with epsilon=0)
    dirichlet_epsilon: float = 0.0  # Dirichlet epsilon (0 for deterministic audits)
    mcts_seed: int = 12345  # MCTS RNG seed for reproducibility
    top_k_actions: int = 8  # Number of top root actions logged per move
    dump_full_tree: bool = True  # If true, write full tree JSON for each move
    check_all_valid_actions: bool = (
        True  # If true, evaluate all valid root actions for immediate lines
    )
    value_mean_tolerance: float = (
        1e-5  # Tolerance for exported mean-value consistency checks
    )
    value_sum_tolerance: float = (
        1e-3  # Tolerance for decision-node value-sum accounting checks
    )
    reward_attack_tolerance: float = (
        1e-6  # Tolerance for chance reward vs attack consistency checks
    )
    check_uniform_piece_sampling: bool = (
        True  # If true, check chance-node piece-visit uniformity at high visits
    )
    min_visits_for_uniform_check: int = (
        140  # Minimum visits before enforcing uniformity checks on a chance node
    )
    max_uniform_rel_deviation: float = 0.9  # Max relative deviation from uniform piece counts for high-visit chance nodes
    fail_on_invariant_error: bool = (
        True  # If true, exit non-zero on any invariant failure
    )


def main(args: ScriptArgs) -> None:
    if not args.model_path.exists():
        raise ValueError(f"Model path does not exist: {args.model_path}")

    seeds = get_seeds(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        run_name = args.run_name
    else:
        if len(seeds) <= 4:
            seed_tag = "-".join(str(seed) for seed in seeds)
        else:
            seed_tag = f"{seeds[0]}to{seeds[-1]}n{len(seeds)}"
        run_name = (
            f"{args.model_path.stem}_sim{args.num_simulations}_temp{args.temperature:.2f}"
            f"_mcts{args.mcts_seed}_m{args.max_placements}_seeds{seed_tag}"
        )
    run_dir = args.output_dir / run_name
    if run_dir.exists():
        raise ValueError(
            f"Run directory already exists: {run_dir}. "
            "Use --run_name to provide a unique name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    mcts_config = build_mcts_config(args)
    agent = MCTSAgent(mcts_config)

    if not agent.load_model(str(args.model_path)):
        raise RuntimeError(f"Failed to load model from {args.model_path}")

    logger.info(
        "Starting MCTS tree audit",
        model_path=str(args.model_path),
        run_dir=str(run_dir),
        seeds=seeds,
        max_placements=args.max_placements,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        check_uniform_piece_sampling=args.check_uniform_piece_sampling,
        min_visits_for_uniform_check=args.min_visits_for_uniform_check,
        max_uniform_rel_deviation=args.max_uniform_rel_deviation,
    )

    per_seed_summaries: list[dict] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        summary = run_seed_audit(agent, args, seed, seed_dir)
        per_seed_summaries.append(summary)

    total_moves = sum(int(row["moves_played"]) for row in per_seed_summaries)
    total_lines = sum(int(row["total_lines"]) for row in per_seed_summaries)
    total_attack = sum(int(row["total_attack"]) for row in per_seed_summaries)
    total_failures = sum(int(row["total_failures"]) for row in per_seed_summaries)
    total_selected_line_clears = sum(
        int(row["selected_line_clear_moves"]) for row in per_seed_summaries
    )
    total_line_option_moves = sum(
        int(row["moves_with_any_line_option"]) for row in per_seed_summaries
    )
    total_selected_attacks = sum(
        int(row["selected_attack_moves"]) for row in per_seed_summaries
    )
    total_attack_option_moves = sum(
        int(row["moves_with_any_attack_option"]) for row in per_seed_summaries
    )
    total_decision_backup_mismatches = sum(
        int(row["decision_backup_mismatch_count"]) for row in per_seed_summaries
    )
    total_reward_attack_mismatches = sum(
        int(row["reward_attack_mismatch_count"]) for row in per_seed_summaries
    )
    total_chance_uniform_violations = sum(
        int(row["chance_uniform_violation_count"]) for row in per_seed_summaries
    )

    hidden_piece_counts: dict[tuple[int, ...], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for seed_summary in per_seed_summaries:
        for row in seed_summary["chance_model_hidden_piece_counts"]:
            key = tuple(int(piece) for piece in row["possible_pieces"])
            for piece_row in row["observed_hidden_piece_counts"]:
                piece = int(piece_row["piece"])
                count = int(piece_row["count"])
                hidden_piece_counts[key][piece] += count

    overall_summary = {
        "model_path": str(args.model_path),
        "run_dir": str(run_dir),
        "seeds": [int(seed) for seed in seeds],
        "num_seeds": int(len(seeds)),
        "max_placements": int(args.max_placements),
        "num_simulations": int(args.num_simulations),
        "temperature": float(args.temperature),
        "mcts_seed": int(args.mcts_seed),
        "total_moves": int(total_moves),
        "total_lines": int(total_lines),
        "total_attack": int(total_attack),
        "total_failures": int(total_failures),
        "total_decision_backup_mismatches": int(total_decision_backup_mismatches),
        "total_reward_attack_mismatches": int(total_reward_attack_mismatches),
        "total_chance_uniform_violations": int(total_chance_uniform_violations),
        "selected_line_clear_rate": safe_float(
            (total_selected_line_clears / total_moves) if total_moves > 0 else 0.0
        ),
        "line_option_rate": safe_float(
            (total_line_option_moves / total_moves) if total_moves > 0 else 0.0
        ),
        "selected_attack_rate": safe_float(
            (total_selected_attacks / total_moves) if total_moves > 0 else 0.0
        ),
        "attack_option_rate": safe_float(
            (total_attack_option_moves / total_moves) if total_moves > 0 else 0.0
        ),
        "chance_model_hidden_piece_counts": [
            {
                "possible_pieces": [int(piece) for piece in key],
                "observed_hidden_piece_counts": sorted_piece_counts(
                    {int(piece): int(count) for piece, count in counts.items()}
                ),
                "num_observations": int(sum(counts.values())),
            }
            for key, counts in sorted(
                hidden_piece_counts.items(),
                key=lambda row: (len(row[0]), row[0]),
            )
        ],
        "per_seed": per_seed_summaries,
    }

    write_json(run_dir / "summary.json", overall_summary)

    logger.info(
        "MCTS tree audit complete",
        run_dir=str(run_dir),
        total_moves=total_moves,
        total_lines=total_lines,
        total_attack=total_attack,
        total_failures=total_failures,
        total_decision_backup_mismatches=total_decision_backup_mismatches,
        total_reward_attack_mismatches=total_reward_attack_mismatches,
        total_chance_uniform_violations=total_chance_uniform_violations,
        selected_line_clear_rate=overall_summary["selected_line_clear_rate"],
        line_option_rate=overall_summary["line_option_rate"],
        selected_attack_rate=overall_summary["selected_attack_rate"],
        attack_option_rate=overall_summary["attack_option_rate"],
    )

    if args.fail_on_invariant_error and total_failures > 0:
        raise RuntimeError(
            f"Audit found {total_failures} invariant failures. See {run_dir} for details."
        )


if __name__ == "__main__":
    main(parse(ScriptArgs))
