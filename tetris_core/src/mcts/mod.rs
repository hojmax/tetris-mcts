//! Monte Carlo Tree Search for Tetris
//!
//! Implements AlphaZero-style MCTS with:
//! - Decision nodes (player moves)
//! - Chance nodes (piece spawns from 7-bag)
//! - Neural network priors
//! - PUCT selection

mod action_space;
mod agent;
mod config;
mod export;
mod nodes;
mod results;
mod search;
mod utils;

// Re-export public API
pub use action_space::{
    get_action_space, ActionSpace, HOLD_ACTION_INDEX, NUM_ACTIONS, NUM_PLACEMENT_ACTIONS,
};
pub use agent::MCTSAgent;
pub use config::MCTSConfig;
pub use nodes::{get_valid_action_indices, ChanceNode, DecisionNode, MCTSNode, NodeStats};
pub use results::{
    GameResult, GameStats, GameTreeStats, MCTSResult, MCTSTreeExport, TrainingExample,
    TreeNodeExport,
};
pub use utils::sample_dirichlet;
