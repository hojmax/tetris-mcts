//! Model Evaluation
//!
//! Evaluate models on fixed seeds for consistent benchmarking.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;
use crate::mcts::{MCTSAgent, MCTSConfig};

/// A single move in a game replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayMove {
    pub action: usize,
    pub attack: u32,
}

/// A complete game replay that can be saved and replayed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameReplay {
    pub seed: u64,
    pub moves: Vec<ReplayMove>,
    pub total_attack: u32,
    pub num_moves: u32,
}

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
    /// Total lines cleared across all games
    #[pyo3(get)]
    pub total_lines: u32,
    /// Maximum lines cleared in any single game
    #[pyo3(get)]
    pub max_lines: u32,
    /// Total moves across all games
    #[pyo3(get)]
    pub total_moves: u32,
    /// Average attack per game
    #[pyo3(get)]
    pub avg_attack: f32,
    /// Average lines cleared per game
    #[pyo3(get)]
    pub avg_lines: f32,
    /// Average moves per game
    #[pyo3(get)]
    pub avg_moves: f32,
    /// Attack per piece (efficiency)
    #[pyo3(get)]
    pub attack_per_piece: f32,
    /// Lines cleared per piece (efficiency)
    #[pyo3(get)]
    pub lines_per_piece: f32,
    /// Individual game results: (attack, moves) for each seed
    #[pyo3(get)]
    pub game_results: Vec<(u32, u32)>,
}

#[pymethods]
impl EvalResult {
    fn __repr__(&self) -> String {
        format!(
            "EvalResult(games={}, avg_attack={:.1}, avg_lines={:.1}, max_attack={}, max_lines={}, attack_per_piece={:.3}, lines_per_piece={:.3})",
            self.num_games,
            self.avg_attack,
            self.avg_lines,
            self.max_attack,
            self.max_lines,
            self.attack_per_piece,
            self.lines_per_piece
        )
    }

    /// Convert to dictionary for logging.
    pub fn to_dict(&self) -> HashMap<String, f32> {
        let mut d = HashMap::new();
        d.insert("eval/num_games".to_string(), self.num_games as f32);
        d.insert("eval/total_attack".to_string(), self.total_attack as f32);
        d.insert("eval/max_attack".to_string(), self.max_attack as f32);
        d.insert("eval/total_lines".to_string(), self.total_lines as f32);
        d.insert("eval/max_lines".to_string(), self.max_lines as f32);
        d.insert("eval/avg_attack".to_string(), self.avg_attack);
        d.insert("eval/avg_lines".to_string(), self.avg_lines);
        d.insert("eval/avg_moves".to_string(), self.avg_moves);
        d.insert("eval/attack_per_piece".to_string(), self.attack_per_piece);
        d.insert("eval/lines_per_piece".to_string(), self.lines_per_piece);
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
///     output_path: Optional path to save replays as JSONL
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (model_path, seeds, config=None, max_moves=100, output_path=None))]
pub fn evaluate_model(
    model_path: &str,
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_moves: u32,
    output_path: Option<String>,
) -> PyResult<EvalResult> {
    // Use provided config but force temperature=0 for argmax
    let mut config = config.unwrap_or_default();
    config.temperature = 0.0; // Argmax for deterministic evaluation
    config.max_moves = max_moves;

    let mut agent = MCTSAgent::new(config);

    if !agent.load_model(model_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Failed to load model from {}",
            model_path
        )));
    }

    let mut replay_writer = if let Some(path) = output_path.as_deref() {
        let file = File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to create output file {}: {}",
                path, e
            ))
        })?;
        Some(BufWriter::new(file))
    } else {
        None
    };
    let save_replays = replay_writer.is_some();

    let mut total_attack: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut total_lines: u32 = 0;
    let mut max_lines: u32 = 0;
    let mut total_moves: u32 = 0;
    let mut game_results: Vec<(u32, u32)> = Vec::with_capacity(seeds.len());

    for seed in &seeds {
        // Create deterministic environment with seed
        let mut env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, *seed);
        let mut game_attack: u32 = 0;
        let mut game_lines: u32 = 0;
        let mut game_moves: u32 = 0;
        let mut replay_moves: Vec<ReplayMove> = Vec::new();

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

            // Get NN policy and value
            let nn = agent.get_nn().expect("Model should be loaded");
            let (policy, nn_value) = nn
                .predict_masked(&env, move_idx as usize, &mask, max_moves as usize)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "NN prediction failed: {}",
                        e
                    ))
                })?;

            // Run MCTS search (no noise, argmax via config.temperature=0)
            let result = agent.search(&env, policy, nn_value, false, move_idx as u32);

            // Execute action from action index directly.
            if let Some(attack) = env.execute_action_index(result.action) {
                game_attack += attack;
                if let Some(attack_result) = env.get_last_attack_result() {
                    game_lines += attack_result.lines_cleared;
                }
                game_moves += 1;
                if save_replays {
                    replay_moves.push(ReplayMove {
                        action: result.action,
                        attack,
                    });
                }
            } else {
                break;
            }
        }

        if let Some(writer) = replay_writer.as_mut() {
            let replay = GameReplay {
                seed: *seed,
                moves: replay_moves,
                total_attack: game_attack,
                num_moves: game_moves,
            };
            let json = serde_json::to_string(&replay).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize replay: {}",
                    e
                ))
            })?;
            writeln!(writer, "{}", json).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to write replay: {}",
                    e
                ))
            })?;
        }

        total_attack += game_attack;
        max_attack = max_attack.max(game_attack);
        total_lines += game_lines;
        max_lines = max_lines.max(game_lines);
        total_moves += game_moves;
        game_results.push((game_attack, game_moves));
    }

    if let Some(writer) = replay_writer.as_mut() {
        writer.flush().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to flush output: {}", e))
        })?;
    }

    let num_games = seeds.len() as u32;
    let avg_attack = if num_games > 0 {
        total_attack as f32 / num_games as f32
    } else {
        0.0
    };
    let avg_lines = if num_games > 0 {
        total_lines as f32 / num_games as f32
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
    let lines_per_piece = if total_moves > 0 {
        total_lines as f32 / total_moves as f32
    } else {
        0.0
    };

    Ok(EvalResult {
        num_games,
        total_attack,
        max_attack,
        total_lines,
        max_lines,
        total_moves,
        avg_attack,
        avg_lines,
        avg_moves,
        attack_per_piece,
        lines_per_piece,
        game_results,
    })
}
