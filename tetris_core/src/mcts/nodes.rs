//! MCTS Node Types
//!
//! Decision nodes (player moves) and chance nodes (piece spawns).

use rand::prelude::*;
use std::collections::HashMap;

use crate::env::TetrisEnv;
use crate::piece::NUM_PIECE_TYPES;

use super::action_space::get_action_space;
use super::utils::sample_dirichlet;

/// MCTS Node types
#[derive(Clone)]
pub enum MCTSNode {
    /// Decision node - player chooses an action
    Decision(DecisionNode),
    /// Chance node - random piece spawn
    Chance(ChanceNode),
}

impl MCTSNode {
    /// Get visit count
    pub fn visit_count(&self) -> u32 {
        match self {
            MCTSNode::Decision(n) => n.visit_count,
            MCTSNode::Chance(n) => n.visit_count,
        }
    }

    /// Get mean value
    pub fn mean_value(&self) -> f32 {
        match self {
            MCTSNode::Decision(n) => {
                if n.visit_count > 0 {
                    n.value_sum / n.visit_count as f32
                } else {
                    0.0
                }
            }
            MCTSNode::Chance(n) => {
                if n.visit_count > 0 {
                    n.value_sum / n.visit_count as f32
                } else {
                    0.0
                }
            }
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
    /// Prior probability from neural network
    pub prior: f32,
    /// Children: action index -> child node (ChanceNode after action)
    pub children: HashMap<usize, MCTSNode>,
    /// Valid action indices for this state
    pub valid_actions: Vec<usize>,
    /// Cached priors for valid actions (from neural network)
    pub action_priors: Vec<f32>,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Move number in the game
    pub move_number: u32,
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
            prior: 1.0,
            children: HashMap::new(),
            valid_actions,
            action_priors: Vec::new(),
            is_terminal,
            move_number,
        }
    }

    /// Set priors from neural network output
    pub fn set_priors(&mut self, policy: &[f32]) {
        self.action_priors = self.valid_actions.iter().map(|&idx| policy[idx]).collect();

        // Normalize priors over valid actions
        let sum: f32 = self.action_priors.iter().sum();
        for p in &mut self.action_priors {
            *p /= sum;
        }
    }

    /// Add Dirichlet noise to priors (for root exploration)
    pub fn add_dirichlet_noise(&mut self, alpha: f32, epsilon: f32) {
        let noise = sample_dirichlet(alpha, self.action_priors.len());
        for (prior, n) in self.action_priors.iter_mut().zip(noise.iter()) {
            *prior = (1.0 - epsilon) * *prior + epsilon * n;
        }
    }

    /// Select best action using PUCT formula
    pub fn select_action(&self, c_puct: f32) -> usize {
        let sqrt_total = (self.visit_count as f32).sqrt();
        let mut best_action = self.valid_actions[0];
        let mut best_value = f32::NEG_INFINITY;

        for (i, &action_idx) in self.valid_actions.iter().enumerate() {
            let prior = self.action_priors[i];

            let (q, n) = if let Some(child) = self.children.get(&action_idx) {
                (child.mean_value(), child.visit_count())
            } else {
                (0.0, 0)
            };

            // PUCT formula: Q + c * P * sqrt(N_parent) / (1 + N_child)
            let u = c_puct * prior * sqrt_total / (1.0 + n as f32);
            let value = q + u;

            if value > best_value {
                best_value = value;
                best_action = action_idx;
            }
        }

        best_action
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
    /// Pieces remaining in current bag (for probability computation)
    pub bag_remaining: Vec<usize>,
    /// Randomized order for round-robin piece selection
    pub piece_order: Vec<usize>,
    /// Current index in round-robin sequence
    pub round_robin_idx: usize,
}

impl ChanceNode {
    pub fn new(state: TetrisEnv, attack: u32, bag_remaining: Vec<usize>) -> Self {
        let mut rng = thread_rng();

        // Create randomized piece order for round-robin
        let mut piece_order: Vec<usize> = if bag_remaining.is_empty() {
            // New bag - all pieces
            (0..NUM_PIECE_TYPES).collect()
        } else {
            bag_remaining.clone()
        };
        piece_order.shuffle(&mut rng);

        ChanceNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            attack,
            bag_remaining,
            piece_order,
            round_robin_idx: 0,
        }
    }

    /// Select next piece using round-robin on randomized order
    /// Returns the piece type and advances the index
    pub fn select_piece_round_robin(&mut self) -> usize {
        let piece = self.piece_order[self.round_robin_idx % self.piece_order.len()];
        self.round_robin_idx += 1;
        piece
    }
}

/// Get valid action indices for a state
pub fn get_valid_action_indices(env: &TetrisEnv) -> Vec<usize> {
    let action_space = get_action_space();
    let placements = env.get_possible_placements();

    let mut indices = Vec::new();
    for p in placements {
        let piece = &p.piece;
        if let Some(idx) = action_space.placement_to_index(piece.x, piece.y, piece.rotation) {
            indices.push(idx);
        }
    }

    indices
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
        assert_eq!(node.prior, 1.0);
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
    fn test_decision_node_set_priors() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env, 0);

        // Create mock policy with 734 actions
        let policy = vec![1.0; 734];
        node.set_priors(&policy);

        // Priors should be normalized to sum to 1
        let sum: f32 = node.action_priors.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Priors should sum to 1, got {}", sum);

        // Each valid action should have a prior
        assert_eq!(node.action_priors.len(), node.valid_actions.len());
    }

    #[test]
    fn test_decision_node_add_dirichlet_noise() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env, 0);

        // Set uniform priors
        let policy = vec![1.0; 734];
        node.set_priors(&policy);

        let priors_before: Vec<f32> = node.action_priors.clone();

        // Add Dirichlet noise
        node.add_dirichlet_noise(0.3, 0.25);

        // Priors should still sum to approximately 1
        let sum: f32 = node.action_priors.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Priors should still sum to 1 after noise");

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

        let policy = vec![1.0; 734];
        node.set_priors(&policy);

        // With no visits, selection should return a valid action
        let action = node.select_action(1.0);
        assert!(node.valid_actions.contains(&action));
    }

    #[test]
    fn test_decision_node_select_action_with_children() {
        let env = TetrisEnv::new(10, 20);
        let mut node = DecisionNode::new(env.clone(), 0);

        let policy = vec![1.0; 734];
        node.set_priors(&policy);
        node.visit_count = 10;

        // Add a child with high value
        let action_idx = node.valid_actions[0];
        let mut child = ChanceNode::new(env.clone(), 0, vec![]);
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
        let node = ChanceNode::new(env, 5, bag.clone());

        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert_eq!(node.attack, 5);
        assert!(node.children.is_empty());
        assert_eq!(node.bag_remaining, bag);
        assert_eq!(node.round_robin_idx, 0);
    }

    #[test]
    fn test_chance_node_empty_bag() {
        let env = TetrisEnv::new(10, 20);
        let node = ChanceNode::new(env, 0, vec![]);

        // Empty bag should use all 7 pieces
        assert_eq!(node.piece_order.len(), NUM_PIECE_TYPES);
    }

    #[test]
    fn test_chance_node_select_piece_round_robin() {
        let env = TetrisEnv::new(10, 20);
        let mut node = ChanceNode::new(env, 0, vec![]);

        // Select pieces and verify round-robin behavior
        let mut selected = Vec::new();
        for _ in 0..7 {
            selected.push(node.select_piece_round_robin());
        }

        // Should have selected 7 pieces
        assert_eq!(selected.len(), 7);
        assert_eq!(node.round_robin_idx, 7);

        // After 7 selections, should wrap around
        let next_piece = node.select_piece_round_robin();
        assert_eq!(next_piece, selected[0], "Should wrap around to first piece");
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
        let mut chance = ChanceNode::new(env, 0, vec![]);
        chance.visit_count = 17;

        let node = MCTSNode::Chance(chance);
        assert_eq!(node.visit_count(), 17);
    }

    #[test]
    fn test_mcts_node_mean_value_unvisited() {
        let env = TetrisEnv::new(10, 20);
        let decision = DecisionNode::new(env.clone(), 0);
        let chance = ChanceNode::new(env, 0, vec![]);

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

        let mut chance = ChanceNode::new(env, 0, vec![]);
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
            assert!(*idx < 734, "Action index {} out of bounds", idx);
        }

        // No duplicate indices
        let mut unique = indices.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), indices.len(), "Should have no duplicate action indices");
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
        let mut policy = vec![0.001; 734];
        if let Some(&first_action) = node.valid_actions.first() {
            policy[first_action] = 0.9; // High prior for first action
        }
        node.set_priors(&policy);
        node.visit_count = 1;

        // With high c_puct, should explore high-prior actions
        let action = node.select_action(10.0);
        assert!(node.valid_actions.contains(&action));
    }

    #[test]
    fn test_chance_node_piece_order_randomization() {
        let env = TetrisEnv::new(10, 20);

        // Create multiple nodes and check that piece orders vary
        let mut orders: Vec<Vec<usize>> = Vec::new();
        for _ in 0..10 {
            let node = ChanceNode::new(env.clone(), 0, vec![]);
            orders.push(node.piece_order.clone());
        }

        // At least some orders should be different (probabilistic)
        let first_order = &orders[0];
        let different = orders.iter().any(|o| o != first_order);
        // Note: This test could occasionally fail due to random chance,
        // but with 10 samples it's extremely unlikely
        assert!(different, "Piece orders should be randomized");
    }
}
