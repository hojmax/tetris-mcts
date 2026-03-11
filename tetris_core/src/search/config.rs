//! MCTS Configuration

use pyo3::prelude::*;

/// Configuration for MCTS
#[pyclass]
#[derive(Clone, Debug)]
pub struct MCTSConfig {
    /// Number of simulations per move
    #[pyo3(get, set)]
    pub num_simulations: u32,
    /// PUCT exploration constant
    #[pyo3(get, set)]
    pub c_puct: f32,
    /// Temperature for action selection (0 = argmax, higher = more exploration)
    #[pyo3(get, set)]
    pub temperature: f32,
    /// Dirichlet noise alpha (for root exploration)
    #[pyo3(get, set)]
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight (epsilon)
    #[pyo3(get, set)]
    pub dirichlet_epsilon: f32,
    /// Epsilon for sampling root action from visit-policy instead of argmax
    #[pyo3(get, set)]
    pub visit_sampling_epsilon: f32,
    /// Optional RNG seed for deterministic behavior (None = non-deterministic)
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    /// Maximum placements per episode (hold actions do not count toward this limit).
    /// Used for placement-count normalization in NN features
    #[pyo3(get, set)]
    pub max_placements: u32,
    /// Whether to store per-visit backed-up values for visualization/debugging
    #[pyo3(get, set)]
    pub track_value_history: bool,
    /// Penalty applied when a simulation reaches game over (backs up as negative value)
    #[pyo3(get, set)]
    pub death_penalty: f32,
    /// Weight for normalized overhang penalty subtracted during MCTS search scoring
    #[pyo3(get, set)]
    pub overhang_penalty_weight: f32,
    /// Scale factor applied to neural value-head output during search (0 = ignore value head)
    #[pyo3(get, set)]
    pub nn_value_weight: f32,
    /// Scale for tanh Q squashing in PUCT selection. Some(scale) uses tanh(Q/scale),
    /// None uses global min-max normalization. Set to None for bootstrap (no-NN) mode.
    #[pyo3(get, set)]
    pub q_scale: Option<f32>,
    /// If true, unvisited action children start with the parent node's initial backed-up
    /// total-value estimate instead of zero Q (first-play urgency).
    #[pyo3(get, set)]
    pub use_parent_value_for_unvisited_q: bool,
    /// Reuse the MCTS subtree from the previous move instead of building a fresh tree.
    /// After selecting an action, the subtree corresponding to that action + actual piece
    /// spawn is extracted and used as the starting point for the next search.
    #[pyo3(get, set)]
    pub reuse_tree: bool,
}

#[pymethods]
impl MCTSConfig {
    #[new]
    pub fn new() -> Self {
        MCTSConfig {
            num_simulations: 100,
            c_puct: 1.5,
            temperature: 1.0,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.25,
            visit_sampling_epsilon: 0.0,
            seed: None,
            max_placements: 100,
            track_value_history: false,
            death_penalty: 0.0,
            overhang_penalty_weight: 0.0,
            nn_value_weight: 1.0,
            q_scale: Some(8.0),
            use_parent_value_for_unvisited_q: false,
            reuse_tree: true,
        }
    }
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = MCTSConfig::default();
        assert_eq!(config.num_simulations, 100);
        assert_eq!(config.c_puct, 1.5);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.dirichlet_alpha, 0.15);
        assert_eq!(config.dirichlet_epsilon, 0.25);
        assert_eq!(config.visit_sampling_epsilon, 0.0);
        assert_eq!(config.seed, None);
        assert_eq!(config.max_placements, 100);
        assert!(!config.track_value_history);
        assert_eq!(config.death_penalty, 0.0);
        assert_eq!(config.overhang_penalty_weight, 0.0);
        assert_eq!(config.nn_value_weight, 1.0);
        assert_eq!(config.q_scale, Some(8.0));
        assert!(!config.use_parent_value_for_unvisited_q);
        assert!(config.reuse_tree);
    }
}
