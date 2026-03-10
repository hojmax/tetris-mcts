//! Monte Carlo Tree Search (MCTS) for Tetris.

mod agent;
mod config;
mod export;
mod nodes;
mod results;
pub(crate) mod search;
mod utils;

// Re-export action-space types from the game domain.
pub use crate::game::action_space::{
    get_action_space, ActionSpace, HOLD_ACTION_INDEX, NUM_ACTIONS, NUM_PLACEMENT_ACTIONS,
};

pub use agent::MCTSAgent;
pub use config::MCTSConfig;
pub use nodes::{get_valid_action_indices, ChanceNode, DecisionNode, MCTSNode};
pub use results::{
    GameResult, GameStats, GameTreePlayback, GameTreeStats, GameTreeStep, MCTSResult,
    MCTSTreeExport, TrainingExample, TreeNodeExport,
};
pub use utils::{
    compute_bumpiness, count_overhang_fields_and_holes, normalize_bumpiness,
    normalize_column_heights, normalize_holes, normalize_max_column_height,
    normalize_overhang_fields, normalize_row_fill_counts, normalize_total_blocks, sample_dirichlet,
};
