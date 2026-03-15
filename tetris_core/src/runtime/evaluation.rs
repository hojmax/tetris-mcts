//! Model Evaluation
//!
//! Evaluate models on fixed seeds for consistent benchmarking.

use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::game::env::TetrisEnv;
use crate::search::{MCTSAgent, MCTSConfig};

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
    /// Average per-game variance of `past_attack + raw_nn_value` along the chosen trajectory.
    #[pyo3(get)]
    pub avg_trajectory_predicted_total_attack_variance: f32,
    /// Average per-game std of `past_attack + raw_nn_value` along the chosen trajectory.
    #[pyo3(get)]
    pub avg_trajectory_predicted_total_attack_std: f32,
    /// Average per-game RMSE between `past_attack + raw_nn_value` and final total attack.
    #[pyo3(get)]
    pub avg_trajectory_predicted_total_attack_rmse: f32,
    /// Individual game results: (attack, moves) for each seed
    #[pyo3(get)]
    pub game_results: Vec<(u32, u32)>,
    /// Per-game variance of `past_attack + raw_nn_value` along the chosen trajectory.
    #[pyo3(get)]
    pub trajectory_predicted_total_attack_variances: Vec<f32>,
    /// Per-game std of `past_attack + raw_nn_value` along the chosen trajectory.
    #[pyo3(get)]
    pub trajectory_predicted_total_attack_stds: Vec<f32>,
    /// Per-game RMSE between `past_attack + raw_nn_value` and final total attack.
    #[pyo3(get)]
    pub trajectory_predicted_total_attack_rmses: Vec<f32>,
}

/// Per-game result collected from evaluation.
#[derive(Clone)]
struct GameEval {
    attack: u32,
    lines: u32,
    moves: u32,
    avg_tree_nodes: f32,
    trajectory_predicted_total_attack_variance: f32,
    trajectory_predicted_total_attack_std: f32,
    trajectory_predicted_total_attack_rmse: f32,
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
    let mut total_trajectory_predicted_total_attack_variance: f32 = 0.0;
    let mut total_trajectory_predicted_total_attack_std: f32 = 0.0;
    let mut total_trajectory_predicted_total_attack_rmse: f32 = 0.0;
    let mut trajectory_predicted_total_attack_variances: Vec<f32> = Vec::with_capacity(evals.len());
    let mut trajectory_predicted_total_attack_stds: Vec<f32> = Vec::with_capacity(evals.len());
    let mut trajectory_predicted_total_attack_rmses: Vec<f32> = Vec::with_capacity(evals.len());

    for eval in evals {
        total_attack += eval.attack;
        max_attack = max_attack.max(eval.attack);
        total_lines += eval.lines;
        max_lines = max_lines.max(eval.lines);
        total_moves += eval.moves;
        total_avg_tree_nodes += eval.avg_tree_nodes;
        total_trajectory_predicted_total_attack_variance +=
            eval.trajectory_predicted_total_attack_variance;
        total_trajectory_predicted_total_attack_std += eval.trajectory_predicted_total_attack_std;
        total_trajectory_predicted_total_attack_rmse += eval.trajectory_predicted_total_attack_rmse;
        game_results.push((eval.attack, eval.moves));
        trajectory_predicted_total_attack_variances
            .push(eval.trajectory_predicted_total_attack_variance);
        trajectory_predicted_total_attack_stds.push(eval.trajectory_predicted_total_attack_std);
        trajectory_predicted_total_attack_rmses.push(eval.trajectory_predicted_total_attack_rmse);
    }

    let avg_attack = div_or_zero(total_attack as f32, num_games);
    let avg_lines = div_or_zero(total_lines as f32, num_games);
    let avg_moves = div_or_zero(total_moves as f32, num_games);
    let attack_per_piece = div_or_zero(total_attack as f32, total_moves);
    let lines_per_piece = div_or_zero(total_lines as f32, total_moves);
    let avg_tree_nodes = div_or_zero(total_avg_tree_nodes, num_games);
    let avg_trajectory_predicted_total_attack_variance =
        div_or_zero(total_trajectory_predicted_total_attack_variance, num_games);
    let avg_trajectory_predicted_total_attack_std =
        div_or_zero(total_trajectory_predicted_total_attack_std, num_games);
    let avg_trajectory_predicted_total_attack_rmse =
        div_or_zero(total_trajectory_predicted_total_attack_rmse, num_games);

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
        avg_trajectory_predicted_total_attack_variance,
        avg_trajectory_predicted_total_attack_std,
        avg_trajectory_predicted_total_attack_rmse,
        game_results,
        trajectory_predicted_total_attack_variances,
        trajectory_predicted_total_attack_stds,
        trajectory_predicted_total_attack_rmses,
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
        trajectory_predicted_total_attack_variance: result
            .trajectory_predicted_total_attack_variance,
        trajectory_predicted_total_attack_std: result.trajectory_predicted_total_attack_std,
        trajectory_predicted_total_attack_rmse: result.trajectory_predicted_total_attack_rmse,
    })
}

pub(crate) fn evaluate_avg_attack_on_fixed_seeds<F>(
    agent: &MCTSAgent,
    seeds: &[u64],
    max_placements: u32,
    add_noise: bool,
    mut should_continue: F,
) -> Option<f32>
where
    F: FnMut() -> bool,
{
    if seeds.is_empty() {
        return None;
    }

    let mut total_attack: u64 = 0;
    for &seed in seeds {
        if !should_continue() {
            return None;
        }

        let eval = evaluate_seed(agent, seed, max_placements, add_noise)?;
        total_attack += eval.attack as u64;
    }

    Some(total_attack as f32 / seeds.len() as f32)
}

fn evaluate_agent(
    agent: &MCTSAgent,
    seeds: &[u64],
    max_placements: u32,
    add_noise: bool,
) -> EvalResult {
    let evals: Vec<GameEval> = seeds
        .iter()
        .copied()
        .filter_map(|seed| evaluate_seed(agent, seed, max_placements, add_noise))
        .collect();
    aggregate_game_evals(&evals)
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
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (model_path, seeds, config=None, max_placements=100, num_workers=1, add_noise=false))]
pub fn evaluate_model(
    model_path: &str,
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
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

    Ok(evaluate_agent(&agent, &seeds, max_placements, add_noise))
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
///
/// Returns:
///     EvalResult with aggregated statistics
#[pyfunction]
#[pyo3(signature = (seeds, config=None, max_placements=100, add_noise=false, num_workers=1))]
pub fn evaluate_model_without_nn(
    seeds: Vec<u64>,
    config: Option<MCTSConfig>,
    max_placements: u32,
    add_noise: bool,
    num_workers: u32,
) -> PyResult<EvalResult> {
    if max_placements == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_placements must be > 0",
        ));
    }

    let mut config = config.unwrap_or_default();
    config.max_placements = max_placements;

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
    Ok(evaluate_agent(&agent, &seeds, max_placements, add_noise))
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
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, false, 1)
                .expect("single-thread eval without noise should succeed");
        let single_with_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, true, 1)
                .expect("single-thread eval with noise should succeed");
        assert_ne!(
            single_with_noise.game_results, single_without_noise.game_results,
            "test setup should produce different trajectories when noise is enabled"
        );

        let parallel_with_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, true, 4)
                .expect("parallel eval with noise should succeed");
        assert_eq!(
            parallel_with_noise.game_results, single_with_noise.game_results,
            "parallel evaluation should preserve add_noise behavior"
        );

        let parallel_without_noise =
            evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 30, false, 4)
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

        let first = evaluate_model_without_nn(seeds.clone(), Some(config.clone()), 35, false, 8)
            .expect("first parallel run should succeed");
        let second = evaluate_model_without_nn(seeds, Some(config), 35, false, 8)
            .expect("second parallel run should succeed");

        assert_eq!(
            first.game_results, second.game_results,
            "repeated parallel runs should preserve deterministic evaluation outputs"
        );
    }

    #[test]
    fn avg_attack_helper_matches_public_eval_without_nn() {
        let seeds: Vec<u64> = (0..12).collect();
        let mut config = MCTSConfig::default();
        config.num_simulations = 40;
        config.seed = Some(321);
        config.max_placements = 30;
        config.reuse_tree = true;

        let agent = MCTSAgent::new(config.clone());
        let helper_avg =
            evaluate_avg_attack_on_fixed_seeds(&agent, &seeds, 30, false, || true).unwrap();
        let public_eval = evaluate_model_without_nn(seeds, Some(config), 30, false, 1)
            .expect("public eval should succeed");

        assert!(
            (helper_avg - public_eval.avg_attack).abs() < f32::EPSILON,
            "shared helper should match the public single-threaded eval path"
        );
    }

    #[test]
    fn avg_attack_helper_stops_when_requested() {
        let seeds: Vec<u64> = vec![1, 2, 3];
        let mut config = MCTSConfig::default();
        config.num_simulations = 20;
        config.seed = Some(123);
        config.max_placements = 20;

        let agent = MCTSAgent::new(config);
        let mut callback_calls = 0;
        let avg_attack = evaluate_avg_attack_on_fixed_seeds(&agent, &seeds, 20, false, || {
            callback_calls += 1;
            callback_calls < 2
        });

        assert!(
            avg_attack.is_none(),
            "helper should abort when the continuation callback stops"
        );
        assert_eq!(
            callback_calls, 2,
            "helper should check whether evaluation should continue before each seed"
        );
    }
}
