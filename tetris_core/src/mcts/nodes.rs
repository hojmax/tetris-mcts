//! MCTS Node Types
//!
//! Decision nodes (player moves) and chance nodes (piece spawns).

use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::constants::NUM_PIECE_TYPES;
use crate::env::TetrisEnv;

#[cfg(test)]
use super::action_space::NUM_ACTIONS;
use super::utils::sample_dirichlet;

const Q_NORMALIZATION_EPSILON: f32 = 1e-6;

fn normalize_q_value(q: f32, q_min: f32, q_max: f32) -> f32 {
    let range = q_max - q_min;
    if range.abs() < Q_NORMALIZATION_EPSILON {
        // If all sibling Q values are effectively identical, keep this term neutral.
        return 0.5;
    }
    (q - q_min) / range
}

/// MCTS Node types
#[derive(Clone)]
pub enum MCTSNode {
    /// Decision node - player chooses an action
    Decision(DecisionNode),
    /// Chance node - random piece spawn
    Chance(ChanceNode),
}

impl MCTSNode {
    pub fn visit_count(&self) -> u32 {
        match self {
            MCTSNode::Decision(n) => n.visit_count,
            MCTSNode::Chance(n) => n.visit_count,
        }
    }

    pub fn mean_value(&self) -> f32 {
        match self {
            MCTSNode::Decision(n) => n.mean_value(),
            MCTSNode::Chance(n) => n.mean_value(),
        }
    }
}

/// Decision node where player chooses an action
#[derive(Clone)]
pub struct DecisionNode {
    /// Game state at this node
    pub state: TetrisEnv,
    /// Visit count
    pub visit_count: u32,
    /// Sum of values from all visits
    pub value_sum: f32,
    /// Children: action index -> child node (ChanceNode after action)
    pub children: HashMap<usize, MCTSNode>,
    /// Valid action indices for this state
    pub valid_actions: Vec<usize>,
    /// Cached priors for valid actions (from neural network)
    pub action_priors: Vec<f32>,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Placement count in the game (hold actions do not increment this)
    pub move_number: u32,
    /// Raw neural network value estimate (stored when node is expanded)
    pub nn_value: f32,
    /// All backed-up total values that contributed to value_sum (only when track_value_history=true)
    pub value_history: Option<Vec<f32>>,
}

impl DecisionNode {
    pub fn new(state: TetrisEnv, move_number: u32) -> Self {
        let is_terminal = state.game_over;

        // Get valid actions
        let valid_actions = if is_terminal {
            Vec::new()
        } else {
            get_valid_action_indices(&state)
        };

        DecisionNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            valid_actions,
            action_priors: Vec::new(),
            is_terminal,
            move_number,
            nn_value: 0.0,
            value_history: None,
        }
    }

    /// Set priors and value from neural network output
    pub fn set_nn_output(&mut self, policy: &[f32], value: f32) {
        self.action_priors = self.valid_actions.iter().map(|&idx| policy[idx]).collect();

        // Normalize priors over valid actions
        let sum: f32 = self.action_priors.iter().sum();
        if sum > 0.0 {
            for p in &mut self.action_priors {
                *p /= sum;
            }
        }

        self.nn_value = value;
    }

    /// Add Dirichlet noise to priors (for root exploration)
    pub fn add_dirichlet_noise(&mut self, alpha: f32, epsilon: f32, rng: &mut StdRng) {
        let noise = sample_dirichlet(alpha, self.action_priors.len(), rng);
        for (prior, n) in self.action_priors.iter_mut().zip(noise.iter()) {
            *prior = (1.0 - epsilon) * *prior + epsilon * n;
        }
    }

    /// Select best action using PUCT formula with sibling min-max normalized Q values.
    pub fn select_action(&self, c_puct: f32) -> usize {
        let sqrt_total = (self.visit_count as f32).sqrt();
        let mut q_min = f32::INFINITY;
        let mut q_max = f32::NEG_INFINITY;
        let mut action_stats: Vec<(usize, f32, f32, u32)> =
            Vec::with_capacity(self.valid_actions.len());

        for (i, &action_idx) in self.valid_actions.iter().enumerate() {
            let prior = self.action_priors[i];
            let (q, n) = if let Some(child) = self.children.get(&action_idx) {
                (child.mean_value(), child.visit_count())
            } else {
                (0.0, 0)
            };
            q_min = q_min.min(q);
            q_max = q_max.max(q);
            action_stats.push((action_idx, prior, q, n));
        }

        let mut best_action = self.valid_actions[0];
        let mut best_value = f32::NEG_INFINITY;

        for (action_idx, prior, q, n) in action_stats {
            let normalized_q = normalize_q_value(q, q_min, q_max);
            // PUCT formula: Q_norm + c * P * sqrt(N_parent) / (1 + N_child)
            let u = c_puct * prior * sqrt_total / (1.0 + n as f32);
            let value = normalized_q + u;

            // Use action_idx as tiebreaker for more consistent selection
            if value > best_value || (value == best_value && action_idx < best_action) {
                best_value = value;
                best_action = action_idx;
            }
        }

        best_action
    }
}

impl DecisionNode {
    pub fn mean_value(&self) -> f32 {
        if self.visit_count > 0 {
            self.value_sum / self.visit_count as f32
        } else {
            0.0
        }
    }
}

/// Chance node for stochastic piece spawns
#[derive(Clone)]
pub struct ChanceNode {
    /// Game state (after piece placement, before new piece spawn)
    pub state: TetrisEnv,
    /// Visit count
    pub visit_count: u32,
    /// Sum of values from all visits
    pub value_sum: f32,
    /// Children: piece type -> child DecisionNode
    pub children: HashMap<usize, MCTSNode>,
    /// Attack gained from the action that led to this node
    pub attack: u32,
    /// Overhang fields in the board state at this node
    pub overhang_fields: u32,
    /// Placement count in the game (hold actions do not increment this)
    pub move_number: u32,
    /// Pieces remaining in current bag (possible next pieces)
    pub bag_remaining: Vec<usize>,
    /// Raw neural network value estimate (stored when node is expanded)
    pub nn_value: f32,
    /// Cached policy from NN (shared by all DecisionNode children)
    pub cached_policy: Vec<f32>,
    /// All backed-up total values that contributed to value_sum (only when track_value_history=true)
    pub value_history: Option<Vec<f32>>,
}

impl ChanceNode {
    pub fn new(
        state: TetrisEnv,
        attack: u32,
        overhang_fields: u32,
        move_number: u32,
        bag_remaining: Vec<usize>,
        nn_value: f32,
        cached_policy: Vec<f32>,
    ) -> Self {
        ChanceNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            attack,
            overhang_fields,
            move_number,
            bag_remaining,
            nn_value,
            cached_policy,
            value_history: None,
        }
    }

    /// Randomly select a piece from the possible outcomes
    pub fn select_piece_random(&self, rng: &mut StdRng) -> usize {
        if self.bag_remaining.is_empty() {
            // New bag - any piece is equally likely
            rng.gen_range(0..NUM_PIECE_TYPES)
        } else {
            // Select from remaining pieces in current bag
            *self.bag_remaining.choose(rng).unwrap()
        }
    }
}

impl ChanceNode {
    pub fn mean_value(&self) -> f32 {
        if self.visit_count > 0 {
            self.value_sum / self.visit_count as f32
        } else {
            0.0
        }
    }
}

/// Get valid action indices for a state
pub fn get_valid_action_indices(env: &TetrisEnv) -> Vec<usize> {
    env.get_cached_valid_action_indices()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_node_creation() {
        let env = TetrisEnv::new(10, 20);
        let node = DecisionNode::new(env, 0);

        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert!(node.children.is_empty());
        assert!(!node.valid_actions.is_empty());
        assert!(!node.is_terminal);
        assert_eq!(node.move_number, 0);
    }

    #[test]
    fn test_decision_node_terminal_state() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let node = DecisionNode::new(env, 5);

        assert!(node.is_terminal);
        assert!(node.valid_actions.is_empty());
        assert_eq!(node.move_number, 5);
    }

    #[test]
    fn test_decision_node_set_nn_output() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env, 0);

        // Create mock policy over all actions
        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.5);

        // Priors should be normalized to sum to 1
        let sum: f32 = node.action_priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Priors should sum to 1, got {}",
            sum
        );

        // Each valid action should have a prior
        assert_eq!(node.action_priors.len(), node.valid_actions.len());

        // NN value should be stored
        assert!((node.nn_value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_nn_output_ignores_invalid_action_mass() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env, 0);

        let invalid_action = (0..NUM_ACTIONS)
            .find(|idx| !node.valid_actions.contains(idx))
            .expect("Expected at least one invalid action");
        let preferred_valid_action = node.valid_actions[0];

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[invalid_action] = 1000.0;
        policy[preferred_valid_action] = 1.0;

        node.set_nn_output(&policy, 0.0);

        let preferred_prior_index = node
            .valid_actions
            .iter()
            .position(|&idx| idx == preferred_valid_action)
            .expect("Preferred valid action should be present in valid_actions");

        for (i, prior) in node.action_priors.iter().enumerate() {
            if i == preferred_prior_index {
                assert!(
                    (*prior - 1.0).abs() < 1e-6,
                    "Preferred valid action should have all probability mass"
                );
            } else {
                assert_eq!(
                    *prior, 0.0,
                    "Other valid actions should have zero mass after renormalization"
                );
            }
        }
    }

    #[test]
    fn test_decision_node_add_dirichlet_noise() {
        use rand::SeedableRng;
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env, 0);
        let mut rng = StdRng::seed_from_u64(42);

        // Set uniform priors
        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.0);

        let priors_before: Vec<f32> = node.action_priors.clone();

        // Add Dirichlet noise
        node.add_dirichlet_noise(0.3, 0.25, &mut rng);

        // Priors should still sum to approximately 1
        let sum: f32 = node.action_priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Priors should still sum to 1 after noise"
        );

        // At least some priors should have changed
        let changed = node
            .action_priors
            .iter()
            .zip(priors_before.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(changed, "Dirichlet noise should modify priors");
    }

    #[test]
    fn test_decision_node_select_action_unvisited() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env, 0);

        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.0);

        // With no visits, selection should return a valid action
        let action = node.select_action(1.0);
        assert!(node.valid_actions.contains(&action));
    }

    #[test]
    fn test_decision_node_select_action_with_children() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone(), 0);

        let policy = vec![1.0; NUM_ACTIONS];
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 10;

        // Add a child with high value
        let action_idx = node.valid_actions[0];
        let mut child = ChanceNode::new(env.clone(), 0, 0, 0, vec![], 0.0, vec![]);
        child.visit_count = 5;
        child.value_sum = 10.0; // Mean value = 2.0
        node.children.insert(action_idx, MCTSNode::Chance(child));

        // Selection should consider the child's value
        let selected = node.select_action(1.0);
        assert!(node.valid_actions.contains(&selected));
    }

    #[test]
    fn test_chance_node_creation() {
        let env = TetrisEnv::new(10, 20);
        let bag: Vec<usize> = vec![0, 1, 2];
        let policy = vec![0.1; NUM_ACTIONS];
        let node = ChanceNode::new(env, 5, 0, 0, bag.clone(), 0.0, policy.clone());

        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert_eq!(node.attack, 5);
        assert_eq!(node.overhang_fields, 0);
        assert!(node.children.is_empty());
        assert_eq!(node.bag_remaining, bag);
        assert_eq!(node.cached_policy.len(), NUM_ACTIONS);
    }

    #[test]
    fn test_chance_node_empty_bag() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, 0, 0, vec![], 0.0, vec![]);

        // Empty bag - any piece should be selectable
        let piece = node.select_piece_random(&mut rng);
        assert!(piece < NUM_PIECE_TYPES);
    }

    #[test]
    fn test_chance_node_select_piece_random() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, 0, 0, vec![1, 3, 5], 0.0, vec![]); // O, S, J remaining

        // Select many pieces and verify they're from the bag
        for _ in 0..20 {
            let piece = node.select_piece_random(&mut rng);
            assert!(
                piece == 1 || piece == 3 || piece == 5,
                "Piece {} should be from bag [1, 3, 5]",
                piece
            );
        }
    }

    #[test]
    fn test_mcts_node_visit_count_decision() {
        let env = TetrisEnv::new(10, 20);
        let mut decision = DecisionNode::new(env, 0);
        decision.visit_count = 42;

        let node = MCTSNode::Decision(decision);
        assert_eq!(node.visit_count(), 42);
    }

    #[test]
    fn test_mcts_node_visit_count_chance() {
        let env = TetrisEnv::new(10, 20);
        let mut chance = ChanceNode::new(env, 0, 0, 0, vec![], 0.0, vec![]);
        chance.visit_count = 17;

        let node = MCTSNode::Chance(chance);
        assert_eq!(node.visit_count(), 17);
    }

    #[test]
    fn test_mcts_node_mean_value_unvisited() {
        let env = TetrisEnv::new(10, 20);
        let decision = DecisionNode::new(env.clone(), 0);
        let chance = ChanceNode::new(env, 0, 0, 0, vec![], 0.0, vec![]);

        assert_eq!(MCTSNode::Decision(decision).mean_value(), 0.0);
        assert_eq!(MCTSNode::Chance(chance).mean_value(), 0.0);
    }

    #[test]
    fn test_mcts_node_mean_value_visited() {
        let env = TetrisEnv::new(10, 20);
        let mut decision = DecisionNode::new(env.clone(), 0);
        decision.visit_count = 10;
        decision.value_sum = 25.0;

        assert!((MCTSNode::Decision(decision).mean_value() - 2.5).abs() < 0.01);

        let mut chance = ChanceNode::new(env, 0, 0, 0, vec![], 0.0, vec![]);
        chance.visit_count = 4;
        chance.value_sum = 10.0;

        assert!((MCTSNode::Chance(chance).mean_value() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_get_valid_action_indices() {
        let env = TetrisEnv::new(10, 20);
        let indices = get_valid_action_indices(&env);

        // Should have at least some valid actions
        assert!(!indices.is_empty());

        // All indices should be within action space bounds
        for idx in &indices {
            assert!(*idx < NUM_ACTIONS, "Action index {} out of bounds", idx);
        }

        // No duplicate indices
        let mut unique = indices.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            indices.len(),
            "Should have no duplicate action indices"
        );
    }

    #[test]
    fn test_get_valid_action_indices_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        env.current_piece = None; // Clear the current piece as would happen in game over

        let indices = get_valid_action_indices(&env);

        // Game over state with no current piece should have no valid actions
        assert!(indices.is_empty());
    }

    #[test]
    fn test_decision_node_puct_exploration() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone(), 0);

        // Create non-uniform priors
        let mut policy = vec![0.001; NUM_ACTIONS];
        if let Some(&first_action) = node.valid_actions.first() {
            policy[first_action] = 0.9; // High prior for first action
        }
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 1;

        // With high c_puct, should explore high-prior actions
        let action = node.select_action(10.0);
        assert!(node.valid_actions.contains(&action));
    }

    #[test]
    fn test_normalize_q_value_tiny_range_is_neutral() {
        let q_norm = normalize_q_value(3.0, 2.0, 2.0 + (Q_NORMALIZATION_EPSILON * 0.5));
        assert!((q_norm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_decision_node_select_action_uses_min_max_q_normalization() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone(), 0);

        let high_q_action = node.valid_actions[0];
        let high_prior_action = node.valid_actions[1];
        node.valid_actions = vec![high_q_action, high_prior_action];

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[high_q_action] = 0.01;
        policy[high_prior_action] = 0.99;
        node.set_nn_output(&policy, 0.0);
        node.visit_count = 100;

        let mut high_q_child = ChanceNode::new(env.clone(), 0, 0, 0, vec![], 0.0, vec![]);
        high_q_child.visit_count = 10;
        high_q_child.value_sum = 2000.0; // mean Q = 200
        node.children
            .insert(high_q_action, MCTSNode::Chance(high_q_child));

        let mut high_prior_child = ChanceNode::new(env, 0, 0, 0, vec![], 0.0, vec![]);
        high_prior_child.visit_count = 1;
        high_prior_child.value_sum = 100.0; // mean Q = 100
        node.children
            .insert(high_prior_action, MCTSNode::Chance(high_prior_child));

        // Raw-Q PUCT would choose high_q_action; normalized-Q PUCT should
        // let the strong prior/exploration term win here.
        let selected = node.select_action(1.0);
        assert_eq!(selected, high_prior_action);
    }

    #[test]
    fn test_chance_node_random_selection_variety() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, 0, 0, vec![], 0.0, vec![]); // Empty bag = all 7 pieces possible

        // Select many pieces and verify we get variety
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            seen.insert(node.select_piece_random(&mut rng));
        }

        // With 100 selections from 7 pieces, we should see most of them
        assert!(
            seen.len() >= 5,
            "Random selection should produce variety, got {} unique pieces",
            seen.len()
        );
    }
}
