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
