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
#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayMove {
    pub action: usize,
    pub attack: u32,
}

/// A complete game replay that can be saved and replayed.
#[pyclass(get_all)]
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

fn create_replay_writer(output_path: Option<&str>) -> PyResult<Option<BufWriter<File>>> {
    if let Some(path) = output_path {
        let file = File::create(path).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to create output file {}: {}",
                path, error
            ))
        })?;
        Ok(Some(BufWriter::new(file)))
    } else {
        Ok(None)
    }
}

fn evaluate_agent(
    agent: &MCTSAgent,
    seeds: &[u64],
    max_placements: u32,
    add_noise: bool,
    output_path: Option<String>,
) -> PyResult<EvalResult> {
    let mut replay_writer = create_replay_writer(output_path.as_deref())?;

    let mut total_attack: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut total_lines: u32 = 0;
    let mut max_lines: u32 = 0;
    let mut total_moves: u32 = 0;
    let mut game_results: Vec<(u32, u32)> = Vec::with_capacity(seeds.len());

    for &seed in seeds {
        let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed);
        let Some((result, replay_moves)) = agent.play_game_on_env(env, max_placements, add_noise)
        else {
            continue;
        };

        let game_attack = result.total_attack;
        let game_moves = result.num_moves;
        let game_lines = result.stats.total_lines;

        if let Some(writer) = replay_writer.as_mut() {
            let replay = GameReplay {
                seed,
                moves: replay_moves,
                total_attack: game_attack,
                num_moves: game_moves,
            };
            let json = serde_json::to_string(&replay).map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize replay: {}",
                    error
                ))
            })?;
            writeln!(writer, "{}", json).map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to write replay: {}",
                    error
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
        writer.flush().map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to flush output: {}",
                error
            ))
        })?;
    }

    let num_games = game_results.len() as u32;
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

/// Evaluate a model on fixed seeds for consistent benchmarking.
///
/// Plays games using MCTS with the specified model on deterministic seeds,
/// allowing for reproducible comparison between model versions.
/// Uses the same game loop as self-play (including tree reuse) with
/// argmax action selection (temperature=0) for deterministic evaluation.
///
/// Args:
///     model_path: Path to ONNX model file
///     seeds: List of random seeds to use (determines piece sequence)
///     config: MCTS configuration (temperature is forced to 0 for argmax)
///     max_placements: Maximum placements per game (hold actions do not count)
///     output_path: Optional path to save replays as JSONL
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (model_path, seeds, config=None, max_placements=100, output_path=None))]
pub fn evaluate_model(
    model_path: &str,
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
    output_path: Option<String>,
) -> PyResult<EvalResult> {
    if max_placements == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_placements must be > 0",
        ));
    }

    let mut config = config.unwrap_or_default();
    config.temperature = 0.0;
    config.max_placements = max_placements;

    let mut agent = MCTSAgent::new(config);
    if !agent.load_model(model_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Failed to load model from {}",
            model_path
        )));
    }

    evaluate_agent(&agent, &seeds, max_placements, false, output_path)
}

/// Evaluate MCTS without a network on fixed seeds (uniform priors + zero value).
///
/// Uses the same game loop as self-play (including tree reuse) with
/// argmax action selection (temperature=0) for deterministic evaluation.
///
/// Args:
///     seeds: List of random seeds to use (determines piece sequence)
///     config: MCTS configuration (temperature is forced to 0 for argmax)
///     max_placements: Maximum placements per game (hold actions do not count)
///     add_noise: Whether to add Dirichlet root noise (matches self-play exploration)
///     output_path: Optional path to save replays as JSONL
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (seeds, config=None, max_placements=100, add_noise=false, output_path=None))]
pub fn evaluate_model_without_nn(
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
    add_noise: bool,
    output_path: Option<String>,
) -> PyResult<EvalResult> {
    if max_placements == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_placements must be > 0",
        ));
    }

    let mut config = config.unwrap_or_default();
    config.temperature = 0.0;
    config.max_placements = max_placements;

    let agent = MCTSAgent::new(config);
    evaluate_agent(&agent, &seeds, max_placements, add_noise, output_path)
}
