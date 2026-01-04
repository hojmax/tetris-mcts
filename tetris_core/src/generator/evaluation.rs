//! Model Evaluation
//!
//! Evaluate models on fixed seeds for consistent benchmarking.

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;
use crate::mcts::{get_action_space, MCTSAgent, MCTSConfig};

/// Result of evaluating a model on fixed seeds.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Number of games played
    #[pyo3(get)]
    pub num_games: u32,
    /// Total attack across all games
    #[pyo3(get)]
    pub total_attack: u32,
    /// Maximum attack in any single game
    #[pyo3(get)]
    pub max_attack: u32,
    /// Total moves across all games
    #[pyo3(get)]
    pub total_moves: u32,
    /// Average attack per game
    #[pyo3(get)]
    pub avg_attack: f32,
    /// Average moves per game
    #[pyo3(get)]
    pub avg_moves: f32,
    /// Attack per piece (efficiency)
    #[pyo3(get)]
    pub attack_per_piece: f32,
    /// Individual game results: (attack, moves) for each seed
    #[pyo3(get)]
    pub game_results: Vec<(u32, u32)>,
}

#[pymethods]
impl EvalResult {
    fn __repr__(&self) -> String {
        format!(
            "EvalResult(games={}, avg_attack={:.1}, max_attack={}, attack_per_piece={:.3})",
            self.num_games, self.avg_attack, self.max_attack, self.attack_per_piece
        )
    }

    /// Convert to dictionary for logging.
    pub fn to_dict(&self) -> HashMap<String, f32> {
        let mut d = HashMap::new();
        d.insert("eval/num_games".to_string(), self.num_games as f32);
        d.insert("eval/total_attack".to_string(), self.total_attack as f32);
        d.insert("eval/max_attack".to_string(), self.max_attack as f32);
        d.insert("eval/avg_attack".to_string(), self.avg_attack);
        d.insert("eval/avg_moves".to_string(), self.avg_moves);
        d.insert("eval/attack_per_piece".to_string(), self.attack_per_piece);
        d
    }
}

/// Evaluate a model on fixed seeds for consistent benchmarking.
///
/// Plays games using MCTS with the specified model on deterministic seeds,
/// allowing for reproducible comparison between model versions.
/// Uses argmax action selection (temperature=0) for deterministic evaluation.
///
/// Args:
///     model_path: Path to ONNX model file
///     seeds: List of random seeds to use (determines piece sequence)
///     config: MCTS configuration (temperature is forced to 0 for argmax)
///     max_moves: Maximum moves per game
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (model_path, seeds, config=None, max_moves=100))]
pub fn evaluate_model(
    model_path: &str,
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_moves: u32,
) -> PyResult<EvalResult> {
    // Use provided config but force temperature=0 for argmax
    let mut config = config.unwrap_or_default();
    config.temperature = 0.0; // Argmax for deterministic evaluation

    let mut agent = MCTSAgent::new(config);

    if !agent.load_model(model_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Failed to load model from {}",
            model_path
        )));
    }

    let mut total_attack: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut total_moves: u32 = 0;
    let mut game_results: Vec<(u32, u32)> = Vec::with_capacity(seeds.len());

    for seed in &seeds {
        // Create deterministic environment with seed
        let mut env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, *seed);
        let mut game_attack: u32 = 0;
        let mut game_moves: u32 = 0;

        // Play game with MCTS (no noise, argmax via temperature=0)
        for move_idx in 0..max_moves {
            if env.game_over {
                break;
            }

            // Get action mask
            let mask = crate::nn::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                break;
            }

            // Get NN policy
            let nn = agent.get_nn().expect("Model should be loaded");
            let (policy, _) = nn
                .predict_masked(&env, move_idx as usize, &mask)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "NN prediction failed: {}",
                        e
                    ))
                })?;

            // Run MCTS search (no noise, argmax via config.temperature=0)
            let result = agent.search(&env, policy, false, move_idx as u32);

            // Execute action
            let (x, y, rot) = get_action_space()
                .index_to_placement(result.action)
                .expect("Invalid action from MCTS");
            let placements = env.get_possible_placements();
            if let Some(placement) = placements
                .iter()
                .find(|p| p.piece.x == x && p.piece.y == y && p.piece.rotation == rot)
            {
                let attack = env.execute_placement(placement);
                game_attack += attack;
                game_moves += 1;
            } else {
                break;
            }
        }

        total_attack += game_attack;
        max_attack = max_attack.max(game_attack);
        total_moves += game_moves;
        game_results.push((game_attack, game_moves));
    }

    let num_games = seeds.len() as u32;
    let avg_attack = if num_games > 0 {
        total_attack as f32 / num_games as f32
    } else {
        0.0
    };
    let avg_moves = if num_games > 0 {
        total_moves as f32 / num_games as f32
    } else {
        0.0
    };
    let attack_per_piece = if total_moves > 0 {
        total_attack as f32 / total_moves as f32
    } else {
        0.0
    };

    Ok(EvalResult {
        num_games,
        total_attack,
        max_attack,
        total_moves,
        avg_attack,
        avg_moves,
        attack_per_piece,
        game_results,
    })
}
