//! Model Evaluation
//!
//! Evaluate models on fixed seeds for consistent benchmarking.

use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::game::env::TetrisEnv;
use crate::search::{MCTSAgent, MCTSConfig};

use crate::replay::GameReplay;

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
    /// Average tree size in nodes (mean of per-game avg tree nodes across moves)
    #[pyo3(get)]
    pub avg_tree_nodes: f32,
    /// Individual game results: (attack, moves) for each seed
    #[pyo3(get)]
    pub game_results: Vec<(u32, u32)>,
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

/// Per-game result collected from evaluation.
#[derive(Clone)]
struct GameEval {
    attack: u32,
    lines: u32,
    moves: u32,
    avg_tree_nodes: f32,
}

#[inline]
fn div_or_zero(numerator: f32, denominator: u32) -> f32 {
    if denominator > 0 {
        numerator / denominator as f32
    } else {
        0.0
    }
}

fn aggregate_game_evals(evals: &[GameEval]) -> EvalResult {
    let num_games = evals.len() as u32;
    let mut total_attack: u32 = 0;
    let mut max_attack: u32 = 0;
    let mut total_lines: u32 = 0;
    let mut max_lines: u32 = 0;
    let mut total_moves: u32 = 0;
    let mut total_avg_tree_nodes: f32 = 0.0;
    let mut game_results: Vec<(u32, u32)> = Vec::with_capacity(evals.len());

    for eval in evals {
        total_attack += eval.attack;
        max_attack = max_attack.max(eval.attack);
        total_lines += eval.lines;
        max_lines = max_lines.max(eval.lines);
        total_moves += eval.moves;
        total_avg_tree_nodes += eval.avg_tree_nodes;
        game_results.push((eval.attack, eval.moves));
    }

    let avg_attack = div_or_zero(total_attack as f32, num_games);
    let avg_lines = div_or_zero(total_lines as f32, num_games);
    let avg_moves = div_or_zero(total_moves as f32, num_games);
    let attack_per_piece = div_or_zero(total_attack as f32, total_moves);
    let lines_per_piece = div_or_zero(total_lines as f32, total_moves);
    let avg_tree_nodes = div_or_zero(total_avg_tree_nodes, num_games);

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
        avg_tree_nodes,
        game_results,
    }
}

fn evaluate_seed(
    agent: &MCTSAgent,
    seed: u64,
    max_placements: u32,
    add_noise: bool,
) -> Option<GameEval> {
    let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed);
    let (result, _replay_moves) = agent.play_game_on_env(env, max_placements, add_noise)?;
    Some(GameEval {
        attack: result.total_attack,
        lines: result.stats.total_lines,
        moves: result.num_moves,
        avg_tree_nodes: result.tree_stats.avg_total_nodes,
    })
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
                avg_tree_nodes: result.tree_stats.avg_total_nodes,
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
    let evals: Vec<GameEval> = seeds
        .iter()
        .copied()
        .filter_map(|seed| evaluate_seed(agent, seed, max_placements, add_noise))
        .collect();
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
    add_noise: bool,
    num_workers: u32,
) -> PyResult<EvalResult> {
    if seeds.is_empty() {
        return Ok(aggregate_game_evals(&[]));
    }

    let worker_count = (num_workers as usize).min(seeds.len());
    let shared_seeds = Arc::new(seeds.to_vec());
    let next_seed_index = Arc::new(AtomicUsize::new(0));
    let model_path_owned = model_path.map(str::to_string);
    let config = config.clone();

    let handles: Vec<_> = (0..worker_count)
        .map(|_| {
            let seeds = Arc::clone(&shared_seeds);
            let next_seed_index = Arc::clone(&next_seed_index);
            let model_path = model_path_owned.clone();
            let config = config.clone();
            thread::spawn(move || -> Result<Vec<(usize, Option<GameEval>)>, String> {
                let mut agent = MCTSAgent::new(config);
                if let Some(path) = model_path.as_deref() {
                    if !agent.load_model(path) {
                        return Err(format!("Failed to load model from {}", path));
                    }
                }

                let mut worker_results: Vec<(usize, Option<GameEval>)> = Vec::new();
                loop {
                    let seed_index = next_seed_index.fetch_add(1, Ordering::Relaxed);
                    if seed_index >= seeds.len() {
                        break;
                    }

                    let seed = seeds[seed_index];
                    let eval = evaluate_seed(&agent, seed, max_placements, add_noise);
                    worker_results.push((seed_index, eval));
                }
                Ok(worker_results)
            })
        })
        .collect();

    let mut results_by_seed: Vec<Option<GameEval>> = vec![None; seeds.len()];
    for handle in handles {
        let worker_results = handle
            .join()
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Evaluation thread panicked")
            })?
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyIOError, _>(error))?;
        for (seed_index, eval) in worker_results {
            results_by_seed[seed_index] = eval;
        }
    }

    let all_evals: Vec<GameEval> = results_by_seed.into_iter().flatten().collect();
    Ok(aggregate_game_evals(&all_evals))
}

/// Evaluate a model on fixed seeds for consistent benchmarking.
///
/// Plays games using MCTS with the specified model on fixed seeds,
/// allowing reproducible comparison between model versions.
/// Uses the same game loop as self-play (including tree reuse).
/// Deterministic behavior requires add_noise=false, config.visit_sampling_epsilon=0,
/// and a fixed MCTS seed.
///
/// Args:
///     model_path: Path to ONNX model file
///     seeds: List of random seeds to use (determines piece sequence)
///     config: MCTS configuration
///     max_placements: Maximum placements per game (hold actions do not count)
///     add_noise: Whether to add Dirichlet root noise
///     output_path: Optional path to save replays as JSONL
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (model_path, seeds, config=None, max_placements=100, output_path=None, num_workers=1, add_noise=false))]
pub fn evaluate_model(
    model_path: &str,
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
    output_path: Option<String>,
    num_workers: u32,
    add_noise: bool,
) -> PyResult<EvalResult> {
    if max_placements == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_placements must be > 0",
        ));
    }

    let mut config = config.unwrap_or_default();
    config.max_placements = max_placements;

    if num_workers > 1 && output_path.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "output_path is not supported with num_workers > 1",
        ));
    }

    if num_workers > 1 {
        return evaluate_parallel(
            Some(model_path),
            &seeds,
            &config,
            max_placements,
            add_noise,
            num_workers,
        );
    }

    let mut agent = MCTSAgent::new(config);
    if !agent.load_model(model_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Failed to load model from {}",
            model_path
        )));
    }

    evaluate_agent(&agent, &seeds, max_placements, add_noise, output_path)
}

/// Evaluate MCTS without a network on fixed seeds (uniform priors + zero value).
///
/// Uses the same game loop as self-play (including tree reuse).
/// Deterministic behavior requires add_noise=false, config.visit_sampling_epsilon=0,
/// and a fixed MCTS seed.
///
/// Args:
///     seeds: List of random seeds to use (determines piece sequence)
///     config: MCTS configuration
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
    config.max_placements = max_placements;

    if num_workers > 1 && output_path.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "output_path is not supported with num_workers > 1",
        ));
    }

    if num_workers > 1 {
        return evaluate_parallel(
            None,
            &seeds,
            &config,
            max_placements,
            add_noise,
            num_workers,
        );
    }

    let agent = MCTSAgent::new(config);
    evaluate_agent(&agent, &seeds, max_placements, add_noise, output_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_eval_without_nn_respects_add_noise() {
        let seeds: Vec<u64> = (0..12).collect();
        let mut config = MCTSConfig::default();
        config.num_simulations = 50;
        config.seed = Some(123);
        config.max_placements = 30;
        config.dirichlet_alpha = 0.02;
        config.dirichlet_epsilon = 0.25;
        config.reuse_tree = true;

        let single_without_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, false, None, 1)
                .expect("single-thread eval without noise should succeed");
        let single_with_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, true, None, 1)
                .expect("single-thread eval with noise should succeed");
        assert_ne!(
            single_with_noise.game_results, single_without_noise.game_results,
            "test setup should produce different trajectories when noise is enabled"
        );

        let parallel_with_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, true, None, 4)
                .expect("parallel eval with noise should succeed");
        assert_eq!(
            parallel_with_noise.game_results, single_with_noise.game_results,
            "parallel evaluation should preserve add_noise behavior"
        );

        let parallel_without_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, false, None, 4)
                .expect("parallel eval without noise should succeed");
        assert_eq!(
            parallel_without_noise.game_results, single_without_noise.game_results,
            "parallel and single-thread runs should match when noise is disabled"
        );
    }

    #[test]
    fn repeated_parallel_eval_without_nn_is_deterministic() {
        let seeds: Vec<u64> = (100..132).collect();
        let mut config = MCTSConfig::default();
        config.num_simulations = 60;
        config.seed = Some(777);
        config.max_placements = 35;
        config.reuse_tree = true;

        let first =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 35, false, None, 8)
                .expect("first parallel run should succeed");
        let second = evaluate_model_without_nn(seeds, Some(config), 35, false, None, 8)
            .expect("second parallel run should succeed");

        assert_eq!(
            first.game_results, second.game_results,
            "repeated parallel runs should preserve deterministic evaluation outputs"
        );
    }
}
