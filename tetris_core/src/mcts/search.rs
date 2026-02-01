//! MCTS Search Implementation
//!
//! Core search algorithm including simulation, expansion, and backpropagation.

use crate::constants::QUEUE_SIZE;
use crate::nn::TetrisNN;

use super::action_space::{get_action_space, NUM_ACTIONS};
use super::config::MCTSConfig;
use super::nodes::{ChanceNode, DecisionNode, MCTSNode};
use super::results::MCTSResult;
use super::utils::sample_action;

/// Run a single MCTS simulation
///
/// Uses raw pointers for tree traversal to track the path from root to leaf.
/// This is a common pattern in tree structures where we need mutable access
/// to nodes at multiple levels simultaneously.
///
/// # Safety
/// The unsafe pointer operations are sound because:
/// 1. All pointers are derived from valid mutable references to tree nodes
/// 2. The tree structure is not modified during traversal (no reallocation)
/// 3. Each node is accessed through exactly one pointer at a time
/// 4. Pointers remain valid for the entire duration of a single simulation
pub(super) fn simulate(
    config: &MCTSConfig,
    nn: &TetrisNN,
    root: &mut DecisionNode,
    root_move_number: u32,
) {
    // Selection: traverse tree, tracking path for backpropagation
    // Store (node_ptr, action_idx, attack_at_this_step)
    let mut path: Vec<(*mut DecisionNode, usize, f32)> = Vec::new();
    let mut current = root as *mut DecisionNode;
    let mut depth: u32 = 0;

    loop {
        // SAFETY: `current` is always derived from a valid &mut DecisionNode.
        // The tree structure doesn't change during simulation, so the pointer
        // remains valid. We only hold one mutable reference at a time.
        let node = unsafe { &mut *current };

        if node.is_terminal {
            // Terminal - backpropagate with 0 future value (game over)
            backup_with_value(&path, 0.0);
            return;
        }

        if node.valid_actions.is_empty() {
            backup_with_value(&path, 0.0);
            return;
        }

        // Select action
        let action_idx = node.select_action(config.c_puct);

        // Check if child exists
        if !node.children.contains_key(&action_idx) {
            // Expansion: create new child (NN evaluation happens inside expand_action)
            let child = match expand_action(nn, node, action_idx, root_move_number + depth + 1) {
                Some(c) => c,
                None => {
                    // Expansion should never fail for a valid action selected by MCTS.
                    // If this happens, it indicates a bug in action mask generation or
                    // action space mapping. In release builds, treat as terminal to avoid
                    // crashing during self-play.
                    debug_assert!(
                        false,
                        "BUG: expand_action failed for action {} - action mask is inconsistent",
                        action_idx
                    );
                    backup_with_value(&path, 0.0);
                    return;
                }
            };
            node.children.insert(action_idx, child);

            // Get attack and nn_value from the new node
            let chance_node = match node.children.get(&action_idx) {
                Some(MCTSNode::Chance(cn)) => cn,
                Some(MCTSNode::Decision(_)) => {
                    debug_assert!(false, "BUG: expand_action should create ChanceNode");
                    backup_with_value(&path, 0.0);
                    return;
                }
                None => {
                    debug_assert!(false, "BUG: just inserted child but it's missing");
                    backup_with_value(&path, 0.0);
                    return;
                }
            };
            let leaf_attack = chance_node.attack as f32;
            let leaf_value = chance_node.nn_value; // Use stored NN value

            // Add this step to path with its attack
            path.push((current, action_idx, leaf_attack));

            // Backpropagate: total = attack_along_path + leaf_value
            backup_with_value(&path, leaf_value);
            return;
        }

        // Traverse to child - get attack at this step
        let chance_node = match node.children.get_mut(&action_idx) {
            Some(MCTSNode::Chance(cn)) => cn,
            Some(MCTSNode::Decision(_)) => {
                debug_assert!(false, "BUG: Decision node child should be ChanceNode");
                backup_with_value(&path, 0.0);
                return;
            }
            None => {
                debug_assert!(false, "BUG: child should exist after contains_key check");
                backup_with_value(&path, 0.0);
                return;
            }
        };
        let step_attack = chance_node.attack as f32;
        path.push((current, action_idx, step_attack));
        depth += 1;

        // Randomly select which piece outcome to explore
        let piece = chance_node.select_piece_random();

        // Get or create decision node for this piece
        if !chance_node.children.contains_key(&piece) {
            let decision_child = expand_chance(chance_node, piece, root_move_number + depth);
            chance_node.children.insert(piece, decision_child);
        }

        match chance_node.children.get_mut(&piece) {
            Some(MCTSNode::Decision(decision_node)) => {
                current = decision_node as *mut DecisionNode;
            }
            Some(MCTSNode::Chance(_)) => {
                debug_assert!(false, "BUG: ChanceNode child should be DecisionNode");
                backup_with_value(&path, 0.0);
                return;
            }
            None => {
                debug_assert!(false, "BUG: child should exist after insertion");
                backup_with_value(&path, 0.0);
                return;
            }
        }
    }
}

/// Expand an action from a decision node (creates chance node)
///
/// Returns None if expansion fails (invalid action, missing placement, or NN error).
fn expand_action(
    nn: &TetrisNN,
    parent: &DecisionNode,
    action_idx: usize,
    move_number: u32,
) -> Option<MCTSNode> {
    let mut new_state = parent.state.clone();

    // Get placement coordinates from action index
    let (x, y, rot) = get_action_space().index_to_placement(action_idx)?;

    // Find the matching placement to get move sequence for T-spin detection
    let placements = new_state.get_possible_placements();
    let placement = placements
        .iter()
        .find(|p| p.piece.x == x && p.piece.y == y && p.piece.rotation == rot)?;
    let attack = new_state.execute_placement(placement);

    // Truncate to visible queue length FIRST.
    // This ensures expand_chance pushes to position 5 (the first "unseen" position).
    new_state.truncate_queue(QUEUE_SIZE);

    // Get possible pieces from the 7-bag rule.
    // At bag boundaries, multiple pieces are possible, creating stochastic branching.
    let bag_remaining = new_state.get_possible_next_pieces();

    // Get NN policy and value - cached for all DecisionNode children
    // (They all see the same visible state, only differing in the hidden 6th queue piece)
    let mask = crate::nn::get_action_mask(&new_state);
    let (policy, nn_value) = nn
        .predict_masked(&new_state, move_number as usize, &mask)
        .ok()?;

    Some(MCTSNode::Chance(ChanceNode::new(
        new_state,
        attack,
        bag_remaining,
        nn_value,
        policy,
    )))
}

/// Expand a chance node for a specific piece (creates decision node)
///
/// The "piece" parameter represents the piece that appears at the END of the visible
/// queue (the next unseen piece). This is the actual "chance" in Tetris - we know
/// the current piece and visible queue, but not what comes after.
///
/// Uses cached policy/value from parent ChanceNode since the NN only sees the visible
/// queue (5 pieces) - the hidden 6th piece doesn't affect the NN output.
fn expand_chance(parent: &ChanceNode, piece: usize, move_number: u32) -> MCTSNode {
    let mut new_state = parent.state.clone();

    // Add the selected piece to the end of the queue
    // This represents the "chance" outcome - which piece appears next in the queue
    new_state.push_queue_piece(piece);

    let mut node = DecisionNode::new(new_state, move_number);

    // Use cached policy and value from parent ChanceNode
    // (All children see the same visible state - only the hidden 6th queue piece differs)
    node.set_nn_output(&parent.cached_policy, parent.nn_value);

    MCTSNode::Decision(node)
}

/// Backpropagate total episode value through the path
///
/// All nodes in the path receive the SAME total value:
///   total_value = cumulative_attack_along_path + leaf_value
///
/// The NN predicts future reward from each state. By adding the cumulative
/// attack collected along the path to the leaf's NN value, we get the total
/// expected episode return. This same total is backed up to all nodes so that
/// Q values represent "expected total return when passing through this node".
fn backup_with_value(path: &[(*mut DecisionNode, usize, f32)], leaf_value: f32) {
    if path.is_empty() {
        return;
    }

    // Compute total value = all attacks along path + leaf's future value estimate
    let total_attack: f32 = path.iter().map(|(_, _, attack)| attack).sum();
    let total_value = total_attack + leaf_value;

    // All nodes on the path get the same total value
    for &(node_ptr, action_idx, _) in path.iter() {
        // SAFETY: node_ptr was stored during the simulation traversal from valid
        // &mut DecisionNode references. The tree hasn't been modified, so pointers
        // remain valid. Each pointer in the path refers to a distinct node.
        let node = unsafe { &mut *node_ptr };

        // Update child stats (the ChanceNode we traversed through)
        if let Some(child) = node.children.get_mut(&action_idx) {
            match child {
                MCTSNode::Decision(d) => {
                    d.visit_count += 1;
                    d.value_sum += total_value;
                }
                MCTSNode::Chance(c) => {
                    c.visit_count += 1;
                    c.value_sum += total_value;
                }
            }
        }

        // Update the DecisionNode itself
        node.visit_count += 1;
        node.value_sum += total_value;
    }
}

/// Run MCTS search and return both the result and root node.
pub(super) fn search_internal(
    config: &MCTSConfig,
    nn: &TetrisNN,
    env: &crate::env::TetrisEnv,
    policy: Vec<f32>,
    nn_value: f32,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode) {
    // Create root node (keep full queue - truncation breaks 7-bag tracking)
    let mut root = DecisionNode::new(env.clone(), move_number);
    root.set_nn_output(&policy, nn_value);

    if add_noise {
        root.add_dirichlet_noise(config.dirichlet_alpha, config.dirichlet_epsilon);
    }

    // Run simulations
    for _ in 0..config.num_simulations {
        simulate(config, nn, &mut root, move_number);
    }

    // Build result policy from visit counts
    let mut result_policy = vec![0.0; NUM_ACTIONS];
    let total_visits: u32 = root.children.values().map(|c| c.visit_count()).sum();

    debug_assert!(total_visits > 0, "MCTS should have visits after simulations");

    let action = if config.temperature == 0.0 {
        let (best_action, _) = root
            .children
            .iter()
            .max_by_key(|(_, child)| child.visit_count())
            .map(|(&idx, child)| (idx, child.visit_count()))
            .expect("MCTS root should have children after simulations");
        result_policy[best_action] = 1.0;
        best_action
    } else {
        for (&action_idx, child) in &root.children {
            result_policy[action_idx] = (child.visit_count() as f32).powf(1.0 / config.temperature);
        }
        let sum: f32 = result_policy.iter().sum();
        if sum > 0.0 {
            for p in &mut result_policy {
                *p /= sum;
            }
        }
        sample_action(&result_policy)
    };

    let root_value = if root.visit_count > 0 {
        root.value_sum / root.visit_count as f32
    } else {
        0.0
    };

    let mcts_result = MCTSResult {
        policy: result_policy,
        action,
        value: root_value,
        num_simulations: config.num_simulations,
    };

    (mcts_result, root)
}
