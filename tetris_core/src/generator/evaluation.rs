//! Model Evaluation
//!
//! Evaluate models on fixed seeds for consistent benchmarking.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::thread;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;
use crate::mcts::{MCTSAgent, MCTSConfig};

pub use super::types::{GameReplay, ReplayMove};

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

/// Per-game result collected from evaluation (attack, lines, moves).
struct GameEval {
    attack: u32,
    lines: u32,
    moves: u32,
}

fn aggregate_game_evals(evals: &[GameEval]) -> EvalResult {
    let num_games = evals.len() as u32;
    let mut total_attack: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut total_lines: u32 = 0;
    let mut max_lines: u32 = 0;
    let mut total_moves: u32 = 0;
    let mut game_results: Vec<(u32, u32)> = Vec::with_capacity(evals.len());

    for eval in evals {
        total_attack += eval.attack;
        max_attack = max_attack.max(eval.attack);
        total_lines += eval.lines;
        max_lines = max_lines.max(eval.lines);
        total_moves += eval.moves;
        game_results.push((eval.attack, eval.moves));
    }

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

    EvalResult {
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
    }
}

fn run_games_on_seeds(
    agent: &MCTSAgent,
    seeds: &[u64],
    max_placements: u32,
    add_noise: bool,
) -> Vec<GameEval> {
    let mut evals = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed);
        let Some((result, _replay_moves)) = agent.play_game_on_env(env, max_placements, add_noise)
        else {
            continue;
        };
        evals.push(GameEval {
            attack: result.total_attack,
            lines: result.stats.total_lines,
            moves: result.num_moves,
        });
    }
    evals
}

fn evaluate_agent(
    agent: &MCTSAgent,
    seeds: &[u64],
    max_placements: u32,
    add_noise: bool,
    output_path: Option<String>,
) -> PyResult<EvalResult> {
    // Replay writing path (single-threaded only)
    if let Some(path) = output_path.as_deref() {
        let mut replay_writer = create_replay_writer(Some(path))?;
        let mut evals = Vec::with_capacity(seeds.len());

        for &seed in seeds {
            let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed);
            let Some((result, replay_moves)) =
                agent.play_game_on_env(env, max_placements, add_noise)
            else {
                continue;
            };

            if let Some(writer) = replay_writer.as_mut() {
                let replay = GameReplay {
                    seed,
                    moves: replay_moves,
                    total_attack: result.total_attack,
                    num_moves: result.num_moves,
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

            evals.push(GameEval {
                attack: result.total_attack,
                lines: result.stats.total_lines,
                moves: result.num_moves,
            });
        }

        if let Some(writer) = replay_writer.as_mut() {
            writer.flush().map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to flush output: {}",
                    error
                ))
            })?;
        }

        return Ok(aggregate_game_evals(&evals));
    }

    // No replay — simple path
    let evals = run_games_on_seeds(agent, seeds, max_placements, add_noise);
    Ok(aggregate_game_evals(&evals))
}

/// Run evaluation in parallel across multiple threads.
/// Each thread creates its own MCTSAgent and loads the model independently.
/// Results are collected and merged in original seed order.
fn evaluate_parallel(
    model_path: Option<&str>,
    seeds: &[u64],
    config: &MCTSConfig,
    max_placements: u32,
    num_workers: u32,
) -> PyResult<EvalResult> {
    let num_workers = (num_workers as usize).min(seeds.len());
    let chunk_size = (seeds.len() + num_workers - 1) / num_workers;

    let seed_chunks: Vec<Vec<u64>> = seeds.chunks(chunk_size).map(|c| c.to_vec()).collect();

    let model_path_owned = model_path.map(|s| s.to_string());
    let config = config.clone();

    // Spawn worker threads
    let handles: Vec<_> = seed_chunks
        .into_iter()
        .map(|chunk_seeds| {
            let model_path = model_path_owned.clone();
            let config = config.clone();
            thread::spawn(move || -> Result<Vec<GameEval>, String> {
                let mut agent = MCTSAgent::new(config);
                if let Some(path) = model_path.as_deref() {
                    if !agent.load_model(path) {
                        return Err(format!("Failed to load model from {}", path));
                    }
                }
                Ok(run_games_on_seeds(
                    &agent,
                    &chunk_seeds,
                    max_placements,
                    false,
                ))
            })
        })
        .collect();

    // Join all threads and flatten results in seed order
    let mut all_evals: Vec<GameEval> = Vec::with_capacity(seeds.len());
    for handle in handles {
        let thread_evals = handle
            .join()
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Evaluation thread panicked")
            })?
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        all_evals.extend(thread_evals);
    }

    Ok(aggregate_game_evals(&all_evals))
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
#[pyo3(signature = (model_path, seeds, config=None, max_placements=100, output_path=None, num_workers=1))]
pub fn evaluate_model(
    model_path: &str,
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
    output_path: Option<String>,
    num_workers: u32,
) -> PyResult<EvalResult> {
    if max_placements == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_placements must be > 0",
        ));
    }

    let mut config = config.unwrap_or_default();
    config.temperature = 0.0;
    config.max_placements = max_placements;

    if num_workers > 1 && output_path.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "output_path is not supported with num_workers > 1",
        ));
    }

    if num_workers > 1 {
        return evaluate_parallel(Some(model_path), &seeds, &config, max_placements, num_workers);
    }

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
#[pyo3(signature = (seeds, config=None, max_placements=100, add_noise=false, output_path=None, num_workers=1))]
pub fn evaluate_model_without_nn(
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
    add_noise: bool,
    output_path: Option<String>,
    num_workers: u32,
) -> PyResult<EvalResult> {
    if max_placements == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_placements must be > 0",
        ));
    }

    let mut config = config.unwrap_or_default();
    config.temperature = 0.0;
    config.max_placements = max_placements;

    if num_workers > 1 && output_path.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "output_path is not supported with num_workers > 1",
        ));
    }

    if num_workers > 1 {
        return evaluate_parallel(None, &seeds, &config, max_placements, num_workers);
    }

    let agent = MCTSAgent::new(config);
    evaluate_agent(&agent, &seeds, max_placements, add_noise, output_path)
}
