//! MCTS Search Implementation
//!
//! Core search algorithm including simulation, expansion, and backpropagation.

use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};

use crate::constants::QUEUE_SIZE;
use crate::env::TetrisEnv;
use crate::nn::TetrisNN;

use super::action_space::{HOLD_ACTION_INDEX, NUM_ACTIONS};
use super::config::MCTSConfig;
use super::nodes::{ChanceNode, DecisionNode, MCTSNode};
use super::results::{MCTSResult, TreeStats};

fn sample_action_from_policy(policy: &[f32], rng: &mut StdRng) -> Option<usize> {
    let total_mass: f32 = policy.iter().sum();
    if total_mass <= 0.0 {
        return None;
    }

    let mut threshold = rng.gen::<f32>() * total_mass;
    for (idx, &probability) in policy.iter().enumerate() {
        if probability <= 0.0 {
            continue;
        }
        threshold -= probability;
        if threshold <= 0.0 {
            return Some(idx);
        }
    }

    policy
        .iter()
        .enumerate()
        .rfind(|(_, probability)| **probability > 0.0)
        .map(|(idx, _)| idx)
}

fn uniform_policy_from_mask(mask: &[bool]) -> Vec<f32> {
    let mut policy = vec![0.0; NUM_ACTIONS];
    let valid_count = mask.iter().filter(|&&is_valid| is_valid).count();
    if valid_count == 0 {
        return policy;
    }

    let probability = 1.0 / valid_count as f32;
    for (action_idx, is_valid) in mask.iter().enumerate() {
        if *is_valid {
            policy[action_idx] = probability;
        }
    }
    policy
}

fn maybe_ignore_nn_value(value: f32, ignore_nn_value_head: bool) -> f32 {
    if ignore_nn_value_head {
        0.0
    } else {
        value
    }
}

trait LeafEvaluator {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        move_number: u32,
        max_placements: u32,
    ) -> Option<(Vec<f32>, f32)>;
}

struct NeuralLeafEvaluator<'a> {
    nn: &'a TetrisNN,
    ignore_nn_value_head: bool,
}

impl LeafEvaluator for NeuralLeafEvaluator<'_> {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        move_number: u32,
        max_placements: u32,
    ) -> Option<(Vec<f32>, f32)> {
        let mask = crate::nn::get_action_mask(state);
        match self
            .nn
            .predict_masked(state, move_number as usize, &mask, max_placements as usize)
        {
            Ok((policy, value)) => Some((
                policy,
                maybe_ignore_nn_value(value, self.ignore_nn_value_head),
            )),
            Err(error) => {
                eprintln!(
                    "[MCTS] NN prediction failed during expansion at move {}: {}",
                    move_number, error
                );
                None
            }
        }
    }
}

struct BootstrapLeafEvaluator;

impl LeafEvaluator for BootstrapLeafEvaluator {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        _move_number: u32,
        _max_placements: u32,
    ) -> Option<(Vec<f32>, f32)> {
        let mask = crate::nn::get_action_mask(state);
        Some((uniform_policy_from_mask(&mask), 0.0))
    }
}

/// Run a single MCTS simulation using the provided leaf evaluator.
///
/// Uses raw pointers for tree traversal to track the path from root to leaf.
/// This is a common pattern in tree structures where we need mutable access
/// to nodes at multiple levels simultaneously.
fn simulate<E: LeafEvaluator>(
    config: &MCTSConfig,
    evaluator: &E,
    root: &mut DecisionNode,
    rng: &mut StdRng,
) {
    let mut path: Vec<(*mut DecisionNode, usize, f32)> = Vec::new();
    let mut current = root as *mut DecisionNode;

    loop {
        let node = unsafe { &mut *current };
        node.visit_count += 1;

        // Align search objective with training targets: no value beyond episode horizon.
        if node.move_number >= config.max_placements {
            backup_with_value(&path, 0.0, config.track_value_history);
            return;
        }

        if node.is_terminal {
            let penalty = super::utils::compute_death_penalty(
                node.move_number,
                config.max_placements,
                config.death_penalty,
            );
            backup_with_value(&path, -penalty, config.track_value_history);
            return;
        }

        if node.valid_actions.is_empty() {
            backup_with_value(&path, 0.0, config.track_value_history);
            return;
        }

        let action_idx = node.select_action(config.c_puct);
        let next_move_number = node.move_number
            + if action_idx == HOLD_ACTION_INDEX {
                0
            } else {
                1
            };
        if !node.children.contains_key(&action_idx) {
            let child = match expand_action(
                evaluator,
                node,
                action_idx,
                next_move_number,
                config.max_placements,
            ) {
                Some(child) => child,
                None => {
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

            let chance_node = match node.children.get(&action_idx) {
                Some(MCTSNode::Chance(chance_node)) => chance_node,
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
            path.push((current, action_idx, leaf_reward));
            let leaf_value = if chance_node.state.game_over {
                -super::utils::compute_death_penalty(
                    chance_node.move_number,
                    config.max_placements,
                    config.death_penalty,
                )
            } else if chance_node.move_number >= config.max_placements {
                // No value tail beyond the placement horizon.
                0.0
            } else {
                chance_node.nn_value
            };
            backup_with_value(&path, leaf_value, config.track_value_history);
            return;
        }

        let chance_node = match node.children.get_mut(&action_idx) {
            Some(MCTSNode::Chance(chance_node)) => chance_node,
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
        if chance_node.state.game_over {
            let penalty = super::utils::compute_death_penalty(
                chance_node.move_number,
                config.max_placements,
                config.death_penalty,
            );
            backup_with_value(&path, -penalty, config.track_value_history);
            return;
        }
        if chance_node.move_number >= config.max_placements {
            backup_with_value(&path, 0.0, config.track_value_history);
            return;
        }

        let piece = chance_node.select_piece_random(rng);
        if !chance_node.children.contains_key(&piece) {
            let decision_child = expand_chance(chance_node, piece, chance_node.move_number);
            chance_node.children.insert(piece, decision_child);
        }

        match chance_node.children.get_mut(&piece) {
            Some(MCTSNode::Decision(decision_node)) => current = decision_node as *mut DecisionNode,
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
/// Returns None if expansion fails (invalid action, missing placement, or evaluator error).
fn expand_action<E: LeafEvaluator>(
    evaluator: &E,
    parent: &DecisionNode,
    action_idx: usize,
    move_number: u32,
    max_placements: u32,
) -> Option<MCTSNode> {
    let mut new_state = parent.state.clone();
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
    new_state.truncate_queue(QUEUE_SIZE);
    let bag_remaining = new_state.get_possible_next_pieces();
    let (policy, value) = evaluator.evaluate(&new_state, move_number, max_placements)?;

    Some(MCTSNode::Chance(ChanceNode::new(
        new_state,
        attack,
        overhang_fields,
        move_number,
        bag_remaining,
        value,
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

fn create_search_rng(config: &MCTSConfig, env: &TetrisEnv, move_number: u32) -> StdRng {
    if let Some(mcts_seed) = config.seed {
        let combined_seed = mcts_seed
            .wrapping_add(env.seed)
            .wrapping_add(move_number as u64);
        StdRng::seed_from_u64(combined_seed)
    } else {
        StdRng::from_rng(thread_rng()).expect("Failed to create RNG from thread_rng")
    }
}

fn build_result_from_root(
    config: &MCTSConfig,
    root: &DecisionNode,
    rng: &mut StdRng,
) -> MCTSResult {
    let mut result_policy = vec![0.0; NUM_ACTIONS];

    debug_assert!(
        root.children.values().map(|c| c.visit_count()).sum::<u32>() > 0,
        "MCTS should have visits after simulations"
    );

    let greedy_action = root
        .children
        .iter()
        .max_by(|(&idx_a, child_a), (&idx_b, child_b)| {
            child_a
                .visit_count()
                .cmp(&child_b.visit_count())
                .then_with(|| idx_b.cmp(&idx_a))
        })
        .map(|(&idx, _)| idx)
        .expect("MCTS root should have children after simulations");

    if config.temperature == 0.0 {
        result_policy[greedy_action] = 1.0;
    } else {
        for (&action_idx, child) in &root.children {
            result_policy[action_idx] = (child.visit_count() as f32).powf(1.0 / config.temperature);
        }
        let sum: f32 = result_policy.iter().sum();
        if sum > 0.0 {
            for probability in &mut result_policy {
                *probability /= sum;
            }
        }
    }

    let should_sample_action =
        config.visit_sampling_epsilon > 0.0 && rng.gen::<f32>() < config.visit_sampling_epsilon;
    let action = if should_sample_action {
        sample_action_from_policy(&result_policy, rng).unwrap_or(greedy_action)
    } else {
        greedy_action
    };

    let root_value = if root.visit_count > 0 {
        root.value_sum / root.visit_count as f32
    } else {
        0.0
    };

    MCTSResult {
        policy: result_policy,
        action,
        value: root_value,
        num_simulations: config.num_simulations,
    }
}

fn search_internal_with_evaluator<E: LeafEvaluator>(
    config: &MCTSConfig,
    evaluator: &E,
    env: &TetrisEnv,
    policy: Vec<f32>,
    value: f32,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode, TreeStats) {
    let mut root = DecisionNode::new(env.clone(), move_number);
    root.set_nn_output(&policy, value);

    let mut rng = create_search_rng(config, env, move_number);
    if add_noise {
        root.add_dirichlet_noise(config.dirichlet_alpha, config.dirichlet_epsilon, &mut rng);
    }

    for _ in 0..config.num_simulations {
        simulate(config, evaluator, &mut root, &mut rng);
    }

    let mcts_result = build_result_from_root(config, &root, &mut rng);
    let tree_stats = compute_tree_stats(&root);
    (mcts_result, root, tree_stats)
}

/// Run MCTS search and return the result, root node, and tree statistics.
pub(super) fn search_internal(
    config: &MCTSConfig,
    nn: &TetrisNN,
    env: &TetrisEnv,
    policy: Vec<f32>,
    nn_value: f32,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode, TreeStats) {
    let evaluator = NeuralLeafEvaluator {
        nn,
        ignore_nn_value_head: config.ignore_nn_value_head,
    };
    search_internal_with_evaluator(
        config,
        &evaluator,
        env,
        policy,
        maybe_ignore_nn_value(nn_value, config.ignore_nn_value_head),
        add_noise,
        move_number,
    )
}

/// Run MCTS search without NN guidance (uniform priors and zero value).
pub(crate) fn search_internal_without_nn(
    config: &MCTSConfig,
    env: &TetrisEnv,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode, TreeStats) {
    let root_mask = crate::nn::get_action_mask(env);
    let root_policy = uniform_policy_from_mask(&root_mask);
    let evaluator = BootstrapLeafEvaluator;
    search_internal_with_evaluator(
        config,
        &evaluator,
        env,
        root_policy,
        0.0,
        add_noise,
        move_number,
    )
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::mcts::action_space::HOLD_ACTION_INDEX;

    use super::*;

    struct ConstantEvaluator {
        value: f32,
    }

    impl LeafEvaluator for ConstantEvaluator {
        fn evaluate(
            &self,
            state: &TetrisEnv,
            _move_number: u32,
            _max_placements: u32,
        ) -> Option<(Vec<f32>, f32)> {
            let mask = crate::nn::get_action_mask(state);
            Some((uniform_policy_from_mask(&mask), self.value))
        }
    }

    #[test]
    fn test_simulate_terminal_after_expansion_uses_death_penalty() {
        let mut env = TetrisEnv::new(10, 20);
        for x in 0..env.width {
            env.board[x] = 1;
            env.board[env.width + x] = 1;
        }

        let mut root = DecisionNode::new(env, 0);
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[HOLD_ACTION_INDEX] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 100;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 123.0 };
        let mut rng = StdRng::seed_from_u64(1234);
        simulate(&config, &evaluator, &mut root, &mut rng);

        let expected_value = -super::super::utils::compute_death_penalty(
            0,
            config.max_placements,
            config.death_penalty,
        );
        assert!(
            (root.value_sum - expected_value).abs() < 1e-6,
            "Expected root value_sum to use death penalty ({}), got {}",
            expected_value,
            root.value_sum
        );
    }

    #[test]
    fn test_simulate_stops_at_max_placements_horizon() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env, 3);
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[HOLD_ACTION_INDEX] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 3;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 42.0 };
        let mut rng = StdRng::seed_from_u64(7);
        simulate(&config, &evaluator, &mut root, &mut rng);

        assert_eq!(root.value_sum, 0.0);
        assert!(root.children.is_empty());
    }

    #[test]
    fn test_simulate_horizon_after_last_placement_uses_immediate_reward_only() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env, 2);
        let placement_action = root
            .valid_actions
            .iter()
            .copied()
            .find(|&action_idx| action_idx != HOLD_ACTION_INDEX)
            .expect("Expected at least one placement action");
        let mut root_policy = vec![0.0; NUM_ACTIONS];
        root_policy[placement_action] = 1.0;
        root.set_nn_output(&root_policy, 0.0);

        let mut config = MCTSConfig::default();
        config.max_placements = 3;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 123.0 };
        let mut rng = StdRng::seed_from_u64(99);
        simulate(&config, &evaluator, &mut root, &mut rng);

        let chance_node = match root.children.get(&placement_action) {
            Some(MCTSNode::Chance(chance_node)) => chance_node,
            _ => panic!("Expected expanded child to be a ChanceNode"),
        };
        let expected_value = chance_node.attack as f32;
        assert!(
            (root.value_sum - expected_value).abs() < 1e-6,
            "Expected horizon backup to ignore NN tail value and use immediate reward only (expected {}, got {})",
            expected_value,
            root.value_sum
        );
    }

    #[test]
    fn test_root_greedy_tie_break_prefers_lower_action_index() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env.clone(), 0);
        let root_policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        root.set_nn_output(&root_policy, 0.0);

        let mut left_child = ChanceNode::new(
            env.clone(),
            0,
            0,
            1,
            Vec::new(),
            0.0,
            vec![0.0; NUM_ACTIONS],
        );
        left_child.visit_count = 5;
        let mut right_child =
            ChanceNode::new(env, 0, 0, 1, Vec::new(), 0.0, vec![0.0; NUM_ACTIONS]);
        right_child.visit_count = 5;
        root.children.insert(10, MCTSNode::Chance(left_child));
        root.children.insert(20, MCTSNode::Chance(right_child));

        let mut config = MCTSConfig::default();
        config.temperature = 0.0;
        config.visit_sampling_epsilon = 0.0;

        let mut rng = StdRng::seed_from_u64(11);
        let result = build_result_from_root(&config, &root, &mut rng);
        assert_eq!(result.action, 10);
    }
}
