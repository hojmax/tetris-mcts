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
use super::results::{MCTSResult, TraversalStats, TreeStats};

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

enum ExpandActionError {
    InvariantViolation { action_idx: usize },
    EvaluatorFailed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SimulationOutcome {
    Expansion,
    TerminalEnd,
    HorizonEnd,
}

fn uniform_action_priors_for_valid_actions(valid_action_count: usize) -> Vec<f32> {
    if valid_action_count == 0 {
        return Vec::new();
    }

    vec![1.0 / valid_action_count as f32; valid_action_count]
}

fn scale_nn_value(value: f32, nn_value_weight: f32) -> f32 {
    value * nn_value_weight
}

pub(super) trait LeafEvaluator {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        move_number: u32,
        max_placements: u32,
    ) -> Result<(Vec<f32>, f32), String>;
}

pub(super) struct NeuralLeafEvaluator<'a> {
    pub(super) nn: &'a TetrisNN,
    pub(super) nn_value_weight: f32,
}

impl LeafEvaluator for NeuralLeafEvaluator<'_> {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        move_number: u32,
        max_placements: u32,
    ) -> Result<(Vec<f32>, f32), String> {
        let valid_actions = state.get_cached_valid_action_indices_arc();
        match self.nn.predict_with_valid_actions(
            state,
            move_number as usize,
            valid_actions.as_slice(),
            max_placements as usize,
        ) {
            Ok((policy, value)) => Ok((policy, scale_nn_value(value, self.nn_value_weight))),
            Err(error) => Err(format!(
                "NN prediction failed during expansion at move {}: {}",
                move_number, error
            )),
        }
    }
}

pub(super) struct BootstrapLeafEvaluator;

impl LeafEvaluator for BootstrapLeafEvaluator {
    fn evaluate(
        &self,
        state: &TetrisEnv,
        _move_number: u32,
        _max_placements: u32,
    ) -> Result<(Vec<f32>, f32), String> {
        let valid_actions = state.get_cached_valid_action_indices_arc();
        Ok((
            uniform_action_priors_for_valid_actions(valid_actions.len()),
            0.0,
        ))
    }
}

fn update_q_bounds(q_bounds: &mut (f32, f32), value: f32) {
    q_bounds.0 = q_bounds.0.min(value);
    q_bounds.1 = q_bounds.1.max(value);
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
    q_bounds: &mut (f32, f32),
) -> SimulationOutcome {
    let root_cumulative_attack = root.state.attack as f32;
    let mut path: Vec<(*mut DecisionNode, usize)> = Vec::new();
    let mut path_attack_sum: f32 = 0.0;
    let mut current = root as *mut DecisionNode;

    loop {
        let node = unsafe { &mut *current };
        node.visit_count += 1;

        // No value tail beyond horizon.
        if node.move_number >= config.max_placements {
            let total_value = root_cumulative_attack + path_attack_sum;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(&path, total_value, config.track_value_history);
            return SimulationOutcome::HorizonEnd;
        }

        if node.is_terminal {
            let penalty = super::utils::compute_death_penalty(
                node.move_number,
                config.max_placements,
                config.death_penalty,
            );
            let total_value = root_cumulative_attack + path_attack_sum - penalty;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(&path, total_value, config.track_value_history);
            return SimulationOutcome::TerminalEnd;
        }

        if node.valid_actions.is_empty() {
            panic!("BUG: non-terminal DecisionNode has no valid actions");
        }

        let action_idx = node.select_action(config.c_puct, config.q_scale, *q_bounds);
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
                Ok(child) => child,
                Err(ExpandActionError::InvariantViolation { action_idx }) => {
                    panic!(
                        "BUG: expand_action failed for action {} - action mask is inconsistent",
                        action_idx
                    );
                }
                Err(ExpandActionError::EvaluatorFailed(error)) => {
                    panic!("MCTS leaf evaluation failed during expansion: {}", error);
                }
            };
            node.children.insert(action_idx, child);
            if let Some(MCTSNode::Chance(chance_node)) = node.children.get_mut(&action_idx) {
                chance_node.visit_count += 1;
            }

            let chance_node = match node.children.get(&action_idx) {
                Some(MCTSNode::Chance(chance_node)) => chance_node,
                Some(MCTSNode::Decision(_)) => {
                    unreachable!("BUG: expand_action should create ChanceNode");
                }
                None => {
                    unreachable!("BUG: just inserted child but it's missing");
                }
            };
            let leaf_overhang = super::utils::compute_overhang_penalty(
                chance_node.overhang_fields,
                config.overhang_penalty_weight,
            );
            path.push((current, action_idx));
            path_attack_sum += chance_node.attack as f32;
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
                chance_node.nn_value - leaf_overhang
            };
            let total_value = root_cumulative_attack + path_attack_sum + leaf_value;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(&path, total_value, config.track_value_history);
            return SimulationOutcome::Expansion;
        }

        let chance_node = match node.children.get_mut(&action_idx) {
            Some(MCTSNode::Chance(chance_node)) => chance_node,
            Some(MCTSNode::Decision(_)) => {
                unreachable!("BUG: Decision node child should be ChanceNode");
            }
            None => {
                unreachable!("BUG: child should exist after contains_key check");
            }
        };
        chance_node.visit_count += 1;
        path.push((current, action_idx));
        path_attack_sum += chance_node.attack as f32;
        if chance_node.state.game_over {
            let penalty = super::utils::compute_death_penalty(
                chance_node.move_number,
                config.max_placements,
                config.death_penalty,
            );
            let total_value = root_cumulative_attack + path_attack_sum - penalty;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(&path, total_value, config.track_value_history);
            return SimulationOutcome::TerminalEnd;
        }
        if chance_node.move_number >= config.max_placements {
            let total_value = root_cumulative_attack + path_attack_sum;
            update_q_bounds(q_bounds, total_value);
            backup_with_value(&path, total_value, config.track_value_history);
            return SimulationOutcome::HorizonEnd;
        }

        let piece = chance_node.select_piece_random(rng);
        if !chance_node.children.contains_key(&piece) {
            let decision_child = expand_chance(chance_node, piece, chance_node.move_number);
            chance_node.children.insert(piece, decision_child);
        }

        match chance_node.children.get_mut(&piece) {
            Some(MCTSNode::Decision(decision_node)) => current = decision_node as *mut DecisionNode,
            Some(MCTSNode::Chance(_)) => {
                unreachable!("BUG: ChanceNode child should be DecisionNode");
            }
            None => {
                unreachable!("BUG: child should exist after insertion");
            }
        }
    }
}

/// Expand an action from a decision node (creates chance node)
///
/// Returns an error if action execution or leaf evaluation fails.
fn expand_action<E: LeafEvaluator>(
    evaluator: &E,
    parent: &DecisionNode,
    action_idx: usize,
    move_number: u32,
    max_placements: u32,
) -> Result<MCTSNode, ExpandActionError> {
    let mut new_state = parent.state.mcts_clone();
    let attack = match new_state.execute_action_index(action_idx) {
        Some(attack) => attack,
        None => {
            return Err(ExpandActionError::InvariantViolation { action_idx });
        }
    };

    let overhang_fields = super::utils::count_overhang_fields(&new_state);
    new_state.truncate_queue(QUEUE_SIZE);
    let bag_remaining = new_state.get_possible_next_pieces();
    let (action_priors, value) = evaluator
        .evaluate(&new_state, move_number, max_placements)
        .map_err(ExpandActionError::EvaluatorFailed)?;

    Ok(MCTSNode::Chance(ChanceNode::new(
        new_state,
        attack,
        overhang_fields,
        move_number,
        bag_remaining,
        value,
        action_priors,
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
    let mut new_state = parent.state.mcts_clone();

    // Add the selected piece to the end of the queue
    // This represents the "chance" outcome - which piece appears next in the queue
    new_state.push_queue_piece(piece);

    let mut node = DecisionNode::new(new_state, move_number);

    // Use cached action priors and value from parent ChanceNode
    // (All children see the same visible state - only the hidden 6th queue piece differs)
    node.set_nn_output_for_valid_actions(&parent.cached_action_priors, parent.nn_value);

    MCTSNode::Decision(node)
}

/// Backpropagate total episode value through the path.
///
/// Callers precompute:
///   total_value = root_cumulative_attack + path_rewards + leaf_value
/// and all nodes in the path receive that same total value.
fn backup_with_value(
    path: &[(*mut DecisionNode, usize)],
    total_value: f32,
    track_value_history: bool,
) {
    if path.is_empty() {
        return;
    }

    // All nodes on the path get the same total value
    for &(node_ptr, action_idx) in path.iter() {
        // SAFETY: node_ptr was stored during the simulation traversal from valid
        // &mut DecisionNode references. The tree hasn't been modified, so pointers
        // remain valid. Each pointer in the path refers to a distinct node.
        let node = unsafe { &mut *node_ptr };

        // Update child stats (the ChanceNode we traversed through)
        if let Some(child) = node.children.get_mut(&action_idx) {
            let chance_child = match child {
                MCTSNode::Chance(chance_child) => chance_child,
                MCTSNode::Decision(_) => {
                    unreachable!("DecisionNode child must be a ChanceNode")
                }
            };
            chance_child.value_sum += total_value;
            if track_value_history {
                chance_child
                    .value_history
                    .get_or_insert_with(Vec::new)
                    .push(total_value);
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

/// Core search loop: run simulations on a root and return results.
///
/// Works for both fresh roots and reused subtree roots. Takes ownership
/// of the root and returns it alongside the result for tree reuse.
pub(super) fn run_search<E: LeafEvaluator>(
    config: &MCTSConfig,
    evaluator: &E,
    mut root: DecisionNode,
    add_noise: bool,
) -> (MCTSResult, DecisionNode, TreeStats, TraversalStats) {
    let mut rng = create_search_rng(config, &root.state, root.move_number);
    if add_noise {
        root.add_dirichlet_noise(config.dirichlet_alpha, config.dirichlet_epsilon, &mut rng);
    }

    let mut traversal_stats = TraversalStats::default();
    let mut q_bounds = (f32::INFINITY, f32::NEG_INFINITY);
    for _ in 0..config.num_simulations {
        match simulate(config, evaluator, &mut root, &mut rng, &mut q_bounds) {
            SimulationOutcome::Expansion => traversal_stats.expansions += 1,
            SimulationOutcome::TerminalEnd => traversal_stats.terminal_ends += 1,
            SimulationOutcome::HorizonEnd => traversal_stats.horizon_ends += 1,
        }
    }
    debug_assert_eq!(traversal_stats.total(), config.num_simulations);

    let mcts_result = build_result_from_root(config, &root, &mut rng);
    let tree_stats = compute_tree_stats(&root);
    (mcts_result, root, tree_stats, traversal_stats)
}

/// Run MCTS search with NN guidance from a fresh root.
pub(super) fn search_internal(
    config: &MCTSConfig,
    nn: &TetrisNN,
    env: &TetrisEnv,
    policy: Vec<f32>,
    nn_value: f32,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode, TreeStats, TraversalStats) {
    let evaluator = NeuralLeafEvaluator {
        nn,
        nn_value_weight: config.nn_value_weight,
    };
    let mut root = DecisionNode::new(env.clone(), move_number);
    root.set_nn_output(&policy, scale_nn_value(nn_value, config.nn_value_weight));
    run_search(config, &evaluator, root, add_noise)
}

/// Run MCTS search without NN guidance (uniform priors and zero value).
pub(crate) fn search_internal_without_nn(
    config: &MCTSConfig,
    env: &TetrisEnv,
    add_noise: bool,
    move_number: u32,
) -> (MCTSResult, DecisionNode, TreeStats, TraversalStats) {
    let root_policy = env.get_cached_uniform_policy().as_ref().clone();
    let mut root = DecisionNode::new(env.clone(), move_number);
    root.set_nn_output(&root_policy, 0.0);
    run_search(config, &BootstrapLeafEvaluator, root, add_noise)
}

/// Extract the subtree corresponding to a chosen action and actual piece spawn.
///
/// After MCTS selects `action_idx`, the real environment executes it and a piece spawns.
/// `actual_hidden_piece` is the piece type (0-6) that was drawn from the bag to fill
/// the hidden queue slot (the 6th piece beyond the visible 5).
///
/// Returns the extracted DecisionNode, or None if the subtree path wasn't explored.
/// Value sums require no adjustment: each backed-up value equals
/// `root.state.attack + path_attack_sum + leaf_value`, and because
/// `old_root.state.attack + step_attack == new_root.state.attack`, old and new
/// simulations contribute identical magnitudes for the same leaf path.
pub(super) fn extract_subtree(
    mut root: DecisionNode,
    action_idx: usize,
    actual_hidden_piece: usize,
) -> Option<DecisionNode> {
    // Take the ChanceNode child for the chosen action
    let mut chance_node = match root.children.remove(&action_idx)? {
        MCTSNode::Chance(cn) => cn,
        MCTSNode::Decision(_) => return None,
    };

    // Take the DecisionNode child for the actual piece that spawned
    match chance_node.children.remove(&actual_hidden_piece)? {
        MCTSNode::Decision(dn) => Some(dn),
        MCTSNode::Chance(_) => None,
    }
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
        ) -> Result<(Vec<f32>, f32), String> {
            let valid_actions = state.get_cached_valid_action_indices_arc();
            Ok((
                uniform_action_priors_for_valid_actions(valid_actions.len()),
                self.value,
            ))
        }
    }

    #[test]
    fn test_scale_nn_value() {
        assert_eq!(scale_nn_value(7.0, 0.0), 0.0);
        assert!((scale_nn_value(7.0, 0.01) - 0.07).abs() < 1e-6);
        assert_eq!(scale_nn_value(-3.0, 1.0), -3.0);
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
        let outcome = simulate(&config, &evaluator, &mut root, &mut rng, &mut (f32::INFINITY, f32::NEG_INFINITY));
        assert_eq!(outcome, SimulationOutcome::Expansion);

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
        let outcome = simulate(&config, &evaluator, &mut root, &mut rng, &mut (f32::INFINITY, f32::NEG_INFINITY));
        assert_eq!(outcome, SimulationOutcome::HorizonEnd);

        assert_eq!(root.value_sum, 0.0);
        assert!(root.children.is_empty());
    }

    #[test]
    #[should_panic(expected = "non-terminal DecisionNode has no valid actions")]
    fn test_simulate_panics_when_non_terminal_node_has_no_valid_actions() {
        let env = TetrisEnv::new(10, 20);
        let mut root = DecisionNode::new(env, 0);
        root.valid_actions.clear();
        root.action_priors.clear();

        let mut config = MCTSConfig::default();
        config.max_placements = 3;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 0.0;

        let evaluator = ConstantEvaluator { value: 0.0 };
        let mut rng = StdRng::seed_from_u64(123);
        simulate(&config, &evaluator, &mut root, &mut rng, &mut (f32::INFINITY, f32::NEG_INFINITY));
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
        let outcome = simulate(&config, &evaluator, &mut root, &mut rng, &mut (f32::INFINITY, f32::NEG_INFINITY));
        assert_eq!(outcome, SimulationOutcome::Expansion);

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

    #[test]
    fn test_extract_subtree_returns_matching_decision_node() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 200;
        config.reuse_tree = true;

        let evaluator = ConstantEvaluator { value: 1.0 };

        // Build a tree with a fresh search
        let root_policy = uniform_action_priors_for_valid_actions(
            env.get_cached_valid_action_indices_arc().len(),
        );
        let mut root = DecisionNode::new(env.clone(), 0);
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (f32::INFINITY, f32::NEG_INFINITY);
        for _ in 0..config.num_simulations {
            simulate(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }

        // Find the best action (most visits)
        let best_action = root
            .children
            .iter()
            .max_by_key(|(_, child)| child.visit_count())
            .map(|(&idx, _)| idx)
            .expect("Root should have children");

        // Get the chance node for the best action and find an explored piece
        let chance_node = match root.children.get(&best_action).unwrap() {
            MCTSNode::Chance(cn) => cn,
            _ => panic!("Expected ChanceNode"),
        };
        let explored_piece = chance_node
            .children
            .keys()
            .next()
            .copied()
            .expect("ChanceNode should have at least one child");

        let old_visit_count = match chance_node.children.get(&explored_piece).unwrap() {
            MCTSNode::Decision(dn) => dn.visit_count,
            _ => panic!("Expected DecisionNode"),
        };

        // Extract the subtree
        let subtree = extract_subtree(root, best_action, explored_piece);
        assert!(
            subtree.is_some(),
            "Should extract subtree for explored piece"
        );

        let reused_root = subtree.unwrap();
        assert_eq!(
            reused_root.visit_count, old_visit_count,
            "Reused root should preserve visit count"
        );
        assert!(
            !reused_root.valid_actions.is_empty(),
            "Reused root should have valid actions"
        );
    }

    #[test]
    fn test_extract_subtree_returns_none_for_unexplored_piece() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 5; // Very few sims to leave some pieces unexplored

        let evaluator = ConstantEvaluator { value: 1.0 };
        let root_policy = uniform_action_priors_for_valid_actions(
            env.get_cached_valid_action_indices_arc().len(),
        );
        let mut root = DecisionNode::new(env.clone(), 0);
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (f32::INFINITY, f32::NEG_INFINITY);
        for _ in 0..config.num_simulations {
            simulate(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }

        // Find an action that was explored
        let action = root
            .children
            .keys()
            .next()
            .copied()
            .expect("Should have at least one child");

        let chance_node = match root.children.get(&action).unwrap() {
            MCTSNode::Chance(cn) => cn,
            _ => panic!("Expected ChanceNode"),
        };

        // Find a piece that was NOT explored in this chance node
        let unexplored_piece = (0..7).find(|p| !chance_node.children.contains_key(p));
        if let Some(piece) = unexplored_piece {
            let result = extract_subtree(root, action, piece);
            assert!(result.is_none(), "Should return None for unexplored piece");
        }
        // If all pieces were explored (unlikely with 5 sims), test is vacuously true
    }

    #[test]
    fn test_run_search_on_reused_tree_adds_simulations() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 50;

        let evaluator = ConstantEvaluator { value: 1.0 };
        let root_policy = uniform_action_priors_for_valid_actions(
            env.get_cached_valid_action_indices_arc().len(),
        );
        let mut root = DecisionNode::new(env.clone(), 0);
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);

        // Run initial search
        let mut rng = StdRng::seed_from_u64(42);
        let mut q_bounds = (f32::INFINITY, f32::NEG_INFINITY);
        for _ in 0..config.num_simulations {
            simulate(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        }
        let visits_after_first = root.visit_count;

        // Continue search via run_search (simulates tree reuse)
        let (result, root, _tree_stats, traversal_stats) =
            run_search(&config, &evaluator, root, false);
        assert_eq!(result.num_simulations, config.num_simulations);
        assert_eq!(traversal_stats.total(), config.num_simulations);

        assert_eq!(
            root.visit_count,
            visits_after_first + config.num_simulations,
            "Visit count should increase by num_simulations"
        );
        assert!(
            result.policy.iter().any(|&p| p > 0.0),
            "Policy should have non-zero entries"
        );
    }

    #[test]
    fn test_simulate_revisiting_terminal_child_reports_terminal_outcome() {
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

        let evaluator = ConstantEvaluator { value: 0.0 };
        let mut rng = StdRng::seed_from_u64(1234);
        let mut q_bounds = (f32::INFINITY, f32::NEG_INFINITY);
        let first = simulate(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);
        let second = simulate(&config, &evaluator, &mut root, &mut rng, &mut q_bounds);

        assert_eq!(first, SimulationOutcome::Expansion);
        assert_eq!(second, SimulationOutcome::TerminalEnd);
    }

    #[test]
    fn test_run_search_traversal_stats_sum_to_num_simulations() {
        let env = TetrisEnv::new(10, 20);
        let mut config = MCTSConfig::default();
        config.num_simulations = 40;
        config.max_placements = 6;

        let evaluator = ConstantEvaluator { value: 0.0 };
        let root_policy = uniform_action_priors_for_valid_actions(
            env.get_cached_valid_action_indices_arc().len(),
        );
        let mut root = DecisionNode::new(env.clone(), 0);
        root.set_nn_output_for_valid_actions(&root_policy, 0.0);

        let (_result, _root, _tree_stats, traversal_stats) =
            run_search(&config, &evaluator, root, false);

        assert_eq!(traversal_stats.total(), config.num_simulations);
        assert!(traversal_stats.expansions > 0);
    }
}
