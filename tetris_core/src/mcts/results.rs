//! MCTS Result Types
//!
//! Data structures for search results and training data.

use pyo3::prelude::*;

/// MCTS search result
#[pyclass]
#[derive(Clone)]
pub struct MCTSResult {
    /// Policy (visit counts normalized) over all 734 actions
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Selected action index
    #[pyo3(get)]
    pub action: usize,
    /// Root value estimate
    #[pyo3(get)]
    pub value: f32,
    /// Number of simulations run
    #[pyo3(get)]
    pub num_simulations: u32,
}

#[pymethods]
impl MCTSResult {
    fn __repr__(&self) -> String {
        format!(
            "MCTSResult(action={}, value={:.3}, simulations={})",
            self.action, self.value, self.num_simulations
        )
    }
}

/// Training example returned from self-play
#[pyclass]
#[derive(Clone)]
pub struct TrainingExample {
    /// Board state flattened (20*10 = 200 values, 0 or 1)
    #[pyo3(get)]
    pub board: Vec<u8>,
    /// Current piece type (0-6)
    #[pyo3(get)]
    pub current_piece: usize,
    /// Hold piece type (0-6) or 7 if empty
    #[pyo3(get)]
    pub hold_piece: usize,
    /// Whether hold is available
    #[pyo3(get)]
    pub hold_available: bool,
    /// Next queue (up to 5 piece types)
    #[pyo3(get)]
    pub next_queue: Vec<usize>,
    /// Move number (0-99)
    #[pyo3(get)]
    pub move_number: u32,
    /// MCTS policy target (734 values)
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Value target (cumulative attack from this point)
    #[pyo3(get)]
    pub value: f32,
    /// Action mask (734 values, true = valid)
    #[pyo3(get)]
    pub action_mask: Vec<bool>,
}

/// Result from playing a full game
#[pyclass]
#[derive(Clone)]
pub struct GameResult {
    /// Training examples from this game
    #[pyo3(get)]
    pub examples: Vec<TrainingExample>,
    /// Total attack scored
    #[pyo3(get)]
    pub total_attack: u32,
    /// Number of moves played
    #[pyo3(get)]
    pub num_moves: u32,
}
