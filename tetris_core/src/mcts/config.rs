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
    /// Optional RNG seed for deterministic behavior (None = non-deterministic)
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    /// Maximum moves per episode. Used for move-number normalization in NN features
    #[pyo3(get, set)]
    pub max_moves: u32,
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
            seed: None,
            max_moves: 100,
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
        assert_eq!(config.seed, None);
        assert_eq!(config.max_moves, 100);
    }
}
