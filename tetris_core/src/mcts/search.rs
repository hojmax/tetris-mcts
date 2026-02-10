//! MCTS Search Implementation
//!
//! Core search algorithm including simulation, expansion, and backpropagation.

use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};

use crate::constants::QUEUE_SIZE;
use crate::nn::TetrisNN;

use super::action_space::NUM_ACTIONS;
use super::config::MCTSConfig;
use super::nodes::{ChanceNode, DecisionNode, MCTSNode};
use super::results::{MCTSResult, TreeStats};

fn sample_action_from_policy(policy: &[f32], rng: &mut StdRng) -> Option<usize> {
    let total_mass: f32 = policy.iter().sum();
    if total_mass <= 0.0 {
        return None;
    }

    let mut threshold = rng.gen::<f32>() * total_mass;
    for (idx, &prob) in policy.iter().enumerate() {
        if prob <= 0.0 {
            continue;
        }
        threshold -= prob;
        if threshold <= 0.0 {
            return Some(idx);
        }
    }

    policy
        .iter()
        .enumerate()
        .rfind(|(_, prob)| **prob > 0.0)
        .map(|(idx, _)| idx)
}

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
    rng: &mut StdRng,
) {
    // Selection: traverse tree, tracking path for backpropagation
    // Store (node_ptr, action_idx, reward_at_this_step)
    let mut path: Vec<(*mut DecisionNode, usize, f32)> = Vec::new();
    let mut current = root as *mut DecisionNode;
    let mut depth: u32 = 0;

    loop {
        // SAFETY: `current` is always derived from a valid &mut DecisionNode.
        // The tree structure doesn't change during simulation, so the pointer
        // remains valid. We only hold one mutable reference at a time.
        let node = unsafe { &mut *current };
        node.visit_count += 1;

        if node.is_terminal {
            let penalty = super::utils::compute_death_penalty(
                node.move_number,
                config.max_moves,
                config.death_penalty,
            );
            backup_with_value(&path, -penalty, config.track_value_history);
            return;
        }

        if node.valid_actions.is_empty() {
            backup_with_value(&path, 0.0, config.track_value_history);
            return;
        }

        // Select action
        let action_idx = node.select_action(config.c_puct);

        // Check if child exists
        if !node.children.contains_key(&action_idx) {
            // Expansion: create new child (NN evaluation happens inside expand_action)
            let child = match expand_action(
                nn,
                node,
                action_idx,
                root_move_number + depth + 1,
                config.max_moves,
            ) {
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
                    backup_with_value(&path, 0.0, config.track_value_history);
                    return;
                }
            };
            node.children.insert(action_idx, child);
            if let Some(MCTSNode::Chance(chance_node)) = node.children.get_mut(&action_idx) {
                chance_node.visit_count += 1;
            }

            // Get step reward and nn_value from the new node
            let chance_node = match node.children.get(&action_idx) {
                Some(MCTSNode::Chance(cn)) => cn,
                Some(MCTSNode::Decision(_)) => {
                    debug_assert!(false, "BUG: expand_action should create ChanceNode");
                    backup_with_value(&path, 0.0, config.track_value_history);
                    return;
                }
                None => {
                    debug_assert!(false, "BUG: just inserted child but it's missing");
                    backup_with_value(&path, 0.0, config.track_value_history);
                    return;
                }
            };
            let leaf_reward = chance_node.attack as f32
                - super::utils::compute_overhang_penalty(
                    chance_node.overhang_fields,
                    config.overhang_penalty_weight,
                );
            let leaf_value = chance_node.nn_value; // Use stored NN value

            // Add this step to path with its reward
            path.push((current, action_idx, leaf_reward));

            // Backpropagate: total = reward_along_path + leaf_value
            backup_with_value(&path, leaf_value, config.track_value_history);
            return;
        }

        // Traverse to child - get reward at this step
        let chance_node = match node.children.get_mut(&action_idx) {
            Some(MCTSNode::Chance(cn)) => cn,
            Some(MCTSNode::Decision(_)) => {
                debug_assert!(false, "BUG: Decision node child should be ChanceNode");
                backup_with_value(&path, 0.0, config.track_value_history);
                return;
            }
            None => {
                debug_assert!(false, "BUG: child should exist after contains_key check");
                backup_with_value(&path, 0.0, config.track_value_history);
                return;
            }
        };
        chance_node.visit_count += 1;
        let step_reward = chance_node.attack as f32
            - super::utils::compute_overhang_penalty(
                chance_node.overhang_fields,
                config.overhang_penalty_weight,
            );
        path.push((current, action_idx, step_reward));
        depth += 1;

        // Randomly select which piece outcome to explore
        let piece = chance_node.select_piece_random(rng);

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
                backup_with_value(&path, 0.0, config.track_value_history);
                return;
            }
            None => {
                debug_assert!(false, "BUG: child should exist after insertion");
                backup_with_value(&path, 0.0, config.track_value_history);
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
    max_moves: u32,
) -> Option<MCTSNode> {
    let mut new_state = parent.state.clone();

    // Execute by action index so hold actions are handled consistently.
    let attack = match new_state.execute_action_index(action_idx) {
        Some(attack) => attack,
        None => {
            debug_assert!(
                false,
                "BUG: action {} selected by MCTS is not executable",
                action_idx
            );
            return None;
        }
    };

    let overhang_fields = super::utils::count_overhang_fields(&new_state);

    // Truncate to visible queue length FIRST.
    // This ensures expand_chance pushes to position 5 (the first "unseen" position).
    new_state.truncate_queue(QUEUE_SIZE);

    // Get possible pieces from the 7-bag rule.
    // At bag boundaries, multiple pieces are possible, creating stochastic branching.
    let bag_remaining = new_state.get_possible_next_pieces();

    // Get NN policy and value - cached for all DecisionNode children
    // (They all see the same visible state, only differing in the hidden 6th queue piece)
    let mask = crate::nn::get_action_mask(&new_state);
    let (policy, nn_value) =
        match nn.predict_masked(&new_state, move_number as usize, &mask, max_moves as usize) {
            Ok(result) => result,
            Err(e) => {
                eprintln!(
                    "[MCTS] NN prediction failed during expansion at move {}: {}",
                    move_number, e
                );
                return None;
            }
        };

    Some(MCTSNode::Chance(ChanceNode::new(
        new_state,
        attack,
        overhang_fields,
        move_number,
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
///   total_value = cumulative_reward_along_path + leaf_value
///
/// The NN predicts future reward from each state. By adding the cumulative
/// reward collected along the path to the leaf's NN value, we get the total
/// expected episode return. This same total is backed up to all nodes so that
/// Q values represent "expected total return when passing through this node".
fn backup_with_value(
    path: &[(*mut DecisionNode, usize, f32)],
    leaf_value: f32,
    track_value_history: bool,
) {
    if path.is_empty() {
        return;
    }

    // Compute total value = all step rewards along path + leaf's future value estimate
    let total_reward: f32 = path.iter().map(|(_, _, reward)| reward).sum();
    let total_value = total_reward + leaf_value;

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
                    d.value_sum += total_value;
                    if track_value_history {
                        d.value_history
                            .get_or_insert_with(Vec::new)
                            .push(total_value);
                    }
                }
                MCTSNode::Chance(c) => {
                    c.value_sum += total_value;
                    if track_value_history {
                        c.value_history
                            .get_or_insert_with(Vec::new)
                            .push(total_value);
                    }
                }
            }
        }

        // Update the DecisionNode itself
        node.value_sum += total_value;
        if track_value_history {
            node.value_history
                .get_or_insert_with(Vec::new)
                .push(total_value);
        }
    }
}

/// Compute statistics about the MCTS tree structure.
pub(super) fn compute_tree_stats(root: &DecisionNode) -> TreeStats {
    let mut total_nodes: u32 = 0;
    let mut num_leaves: u32 = 0;
    let mut children_sum: u32 = 0;
    let mut non_leaf_count: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut max_depth: u32 = 0;

    enum NodeRef<'a> {
        Decision(&'a DecisionNode),
        Chance(&'a ChanceNode),
    }

    let mut stack: Vec<(NodeRef, u32)> = vec![(NodeRef::Decision(root), 0)];

    while let Some((node, depth)) = stack.pop() {
        total_nodes += 1;
        max_depth = max_depth.max(depth);

        match node {
            NodeRef::Decision(d) => {
                if d.children.is_empty() {
                    num_leaves += 1;
                } else {
                    non_leaf_count += 1;
                    children_sum += d.children.len() as u32;
                    for child in d.children.values() {
                        if let MCTSNode::Chance(cn) = child {
                            stack.push((NodeRef::Chance(cn), depth + 1));
                        }
                    }
                }
            }
            NodeRef::Chance(c) => {
                max_attack = max_attack.max(c.attack);
                if c.children.is_empty() {
                    num_leaves += 1;
                } else {
                    non_leaf_count += 1;
                    children_sum += c.children.len() as u32;
                    for child in c.children.values() {
                        if let MCTSNode::Decision(dn) = child {
                            stack.push((NodeRef::Decision(dn), depth + 1));
                        }
                    }
                }
            }
        }
    }

    let branching_factor = if non_leaf_count > 0 {
        children_sum as f32 / non_leaf_count as f32
    } else {
        0.0
    };

    TreeStats {
        branching_factor,
        num_leaves,
        total_nodes,
        max_depth,
        max_attack,
    }
}

/// Run MCTS search and return the result, root node, and tree statistics.
pub(super) fn search_internal(
    config: &MCTSConfig,
    nn: &TetrisNN,
    env: &crate::env::TetrisEnv,
    policy: Vec<f32>,
    nn_value: f32,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode, TreeStats) {
    // Create root node (keep full queue - truncation breaks 7-bag tracking)
    let mut root = DecisionNode::new(env.clone(), move_number);
    root.set_nn_output(&policy, nn_value);

    // Create RNG (seeded if config.seed is Some, otherwise thread_rng)
    // Combine MCTS seed with env seed and move number for unique RNG per (game, move)
    let mut rng = if let Some(mcts_seed) = config.seed {
        let combined_seed = mcts_seed
            .wrapping_add(env.seed)
            .wrapping_add(move_number as u64);
        StdRng::seed_from_u64(combined_seed)
    } else {
        StdRng::from_rng(thread_rng()).expect("Failed to create RNG from thread_rng")
    };

    if add_noise {
        root.add_dirichlet_noise(config.dirichlet_alpha, config.dirichlet_epsilon, &mut rng);
    }

    // Run simulations
    for _ in 0..config.num_simulations {
        simulate(config, nn, &mut root, move_number, &mut rng);
    }

    // Build result policy from visit counts
    let mut result_policy = vec![0.0; NUM_ACTIONS];

    debug_assert!(
        root.children.values().map(|c| c.visit_count()).sum::<u32>() > 0,
        "MCTS should have visits after simulations"
    );

    // Greedy action from highest visit count.
    // Use action index as tiebreaker for deterministic behavior
    let greedy_action = root
        .children
        .iter()
        .max_by_key(|(&idx, child)| (child.visit_count(), idx))
        .map(|(&idx, _)| idx)
        .expect("MCTS root should have children after simulations");

    // Build policy from visit counts with temperature
    // Temperature=0 means deterministic (one-hot on best action)
    // Temperature sharpens (T<1) or softens (T>1) the distribution
    if config.temperature == 0.0 {
        // One-hot policy on the best action (for evaluation)
        result_policy[greedy_action] = 1.0;
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
    }

    let should_sample_action =
        config.visit_sampling_epsilon > 0.0 && rng.gen::<f32>() < config.visit_sampling_epsilon;
    let action = if should_sample_action {
        sample_action_from_policy(&result_policy, &mut rng).unwrap_or(greedy_action)
    } else {
        greedy_action
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

    let tree_stats = compute_tree_stats(&root);

    (mcts_result, root, tree_stats)
}
