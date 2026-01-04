//! Background game generation and evaluation for parallel training.
//!
//! Provides:
//! - `GameGenerator`: Background thread that continuously generates self-play games
//! - `evaluate_model`: Evaluate a model on fixed seeds for consistent benchmarking

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;
use npyz::WriterBuilder;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;
use crate::mcts::{MCTSAgent, MCTSConfig, TrainingExample, NUM_ACTIONS, get_action_space};
use crate::piece::NUM_PIECE_TYPES;

/// Background game generator for parallel training.
///
/// Spawns a worker thread that continuously generates games using MCTS
/// and writes training data to disk in NPZ format.
#[pyclass]
pub struct GameGenerator {
    /// Path to ONNX model file (watched for updates)
    model_path: PathBuf,
    /// Directory to write game data files
    output_dir: PathBuf,
    /// MCTS configuration
    config: MCTSConfig,
    /// Maximum moves per game
    max_moves: u32,
    /// Whether to add Dirichlet noise
    add_noise: bool,
    /// Whether the generator is running
    running: Arc<AtomicBool>,
    /// Number of games generated since start
    games_generated: Arc<AtomicU64>,
    /// Number of examples generated since start
    examples_generated: Arc<AtomicU64>,
    /// Thread handle (for joining on stop)
    thread_handle: Option<JoinHandle<()>>,
}

#[pymethods]
impl GameGenerator {
    #[new]
    #[pyo3(signature = (model_path, output_dir, config=None, max_moves=100, add_noise=true))]
    pub fn new(
        model_path: String,
        output_dir: String,
        config: Option<MCTSConfig>,
        max_moves: u32,
        add_noise: bool,
    ) -> Self {
        GameGenerator {
            model_path: PathBuf::from(model_path),
            output_dir: PathBuf::from(output_dir),
            config: config.unwrap_or_default(),
            max_moves,
            add_noise,
            running: Arc::new(AtomicBool::new(false)),
            games_generated: Arc::new(AtomicU64::new(0)),
            examples_generated: Arc::new(AtomicU64::new(0)),
            thread_handle: None,
        }
    }

    /// Start background game generation.
    ///
    /// Spawns a worker thread that continuously generates games and writes
    /// them to the output directory. The thread watches the model file for
    /// changes and hot-swaps when updated.
    pub fn start(&mut self) -> PyResult<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Generator is already running",
            ));
        }

        // Create output directory if needed
        fs::create_dir_all(&self.output_dir).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to create output directory: {}",
                e
            ))
        })?;

        // Set running flag
        self.running.store(true, Ordering::SeqCst);

        // Clone Arc handles for the thread
        let model_path = self.model_path.clone();
        let output_dir = self.output_dir.clone();
        let config = self.config.clone();
        let max_moves = self.max_moves;
        let add_noise = self.add_noise;
        let running = Arc::clone(&self.running);
        let games_generated = Arc::clone(&self.games_generated);
        let examples_generated = Arc::clone(&self.examples_generated);

        // Spawn worker thread
        let handle = thread::spawn(move || {
            Self::worker_loop(
                model_path,
                output_dir,
                config,
                max_moves,
                add_noise,
                running,
                games_generated,
                examples_generated,
            );
        });

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Stop background game generation.
    ///
    /// Signals the worker thread to stop and waits for it to finish.
    pub fn stop(&mut self) -> PyResult<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        // Signal stop
        self.running.store(false, Ordering::SeqCst);

        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Worker thread panicked")
            })?;
        }

        Ok(())
    }

    /// Check if the generator is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the number of games generated since start.
    pub fn games_generated(&self) -> u64 {
        self.games_generated.load(Ordering::SeqCst)
    }

    /// Get the number of training examples generated since start.
    pub fn examples_generated(&self) -> u64 {
        self.examples_generated.load(Ordering::SeqCst)
    }

    /// Get statistics as a dictionary.
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("games_generated".to_string(), self.games_generated());
        stats.insert("examples_generated".to_string(), self.examples_generated());
        stats.insert("is_running".to_string(), self.is_running() as u64);
        stats
    }
}

impl GameGenerator {
    /// Worker thread main loop.
    fn worker_loop(
        model_path: PathBuf,
        output_dir: PathBuf,
        config: MCTSConfig,
        max_moves: u32,
        add_noise: bool,
        running: Arc<AtomicBool>,
        games_generated: Arc<AtomicU64>,
        examples_generated: Arc<AtomicU64>,
    ) {
        let mut agent = MCTSAgent::new(config);
        let mut current_mtime: u64 = 0;

        // Wait for initial model
        while running.load(Ordering::SeqCst) {
            if let Some(mtime) = Self::get_model_mtime(&model_path) {
                if agent.load_model(model_path.to_str().unwrap()) {
                    current_mtime = mtime;
                    eprintln!("[GameGenerator] Loaded initial model");
                    break;
                }
            }
            thread::sleep(Duration::from_millis(500));
        }

        // Main generation loop
        while running.load(Ordering::SeqCst) {
            // Check for model updates
            if let Some(mtime) = Self::get_model_mtime(&model_path) {
                if mtime > current_mtime {
                    if agent.load_model(model_path.to_str().unwrap()) {
                        current_mtime = mtime;
                        eprintln!("[GameGenerator] Reloaded updated model");
                    }
                }
            }

            // Play one game
            if let Some(result) = agent.play_game(max_moves, add_noise) {
                let num_examples = result.examples.len() as u64;

                // Write to disk
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                let filepath = output_dir.join(format!("game_{}.npz", timestamp));

                if let Err(e) = write_examples_to_npz(&filepath, &result.examples) {
                    eprintln!("[GameGenerator] Failed to write NPZ: {}", e);
                    continue;
                }

                // Update counters
                games_generated.fetch_add(1, Ordering::SeqCst);
                examples_generated.fetch_add(num_examples, Ordering::SeqCst);
            }
        }

        eprintln!("[GameGenerator] Worker thread exiting");
    }

    /// Get the modification time of a file as seconds since UNIX epoch.
    fn get_model_mtime(path: &PathBuf) -> Option<u64> {
        fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| t.duration_since(UNIX_EPOCH).unwrap().as_secs())
    }
}

impl Drop for GameGenerator {
    fn drop(&mut self) {
        // Ensure thread is stopped when generator is dropped
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

// ============================================================================
// Evaluation
// ============================================================================

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
    config.temperature = 0.0;  // Argmax for deterministic evaluation

    let mut agent = MCTSAgent::new(config);

    if !agent.load_model(model_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("Failed to load model from {}", model_path)
        ));
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
            let (policy, _) = nn.predict_masked(&env, move_idx as usize, &mask)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("NN prediction failed: {}", e)
                ))?;

            // Run MCTS search (no noise, argmax via config.temperature=0)
            let result = agent.search(&env, policy, false, move_idx as u32);

            // Execute action
            let (x, y, rot) = get_action_space().index_to_placement(result.action)
                .expect("Invalid action from MCTS");
            let placements = env.get_possible_placements();
            if let Some(placement) = placements.iter().find(|p| {
                p.piece.x == x && p.piece.y == y && p.piece.rotation == rot
            }) {
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
    let avg_attack = if num_games > 0 { total_attack as f32 / num_games as f32 } else { 0.0 };
    let avg_moves = if num_games > 0 { total_moves as f32 / num_games as f32 } else { 0.0 };
    let attack_per_piece = if total_moves > 0 { total_attack as f32 / total_moves as f32 } else { 0.0 };

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

// ============================================================================
// NPZ Writing
// ============================================================================

/// Write training examples to NPZ format (compatible with Python numpy).
///
/// Format matches `save_training_data()` in Python data.py:
/// - boards: (N, 20, 10) bool
/// - current_pieces: (N, 7) float32 one-hot
/// - hold_pieces: (N, 8) float32 one-hot
/// - hold_available: (N,) bool
/// - next_queue: (N, 5, 7) float32 one-hot
/// - move_numbers: (N,) float32 normalized
/// - policy_targets: (N, 734) float32
/// - value_targets: (N,) float32
/// - action_masks: (N, 734) bool
pub fn write_examples_to_npz(
    filepath: &PathBuf,
    examples: &[TrainingExample],
) -> Result<(), String> {
    let n = examples.len();
    if n == 0 {
        return Ok(());
    }

    // Create arrays
    let mut boards: Vec<u8> = Vec::with_capacity(n * BOARD_HEIGHT * BOARD_WIDTH);
    let mut current_pieces: Vec<f32> = vec![0.0; n * NUM_PIECE_TYPES];
    let mut hold_pieces: Vec<f32> = vec![0.0; n * (NUM_PIECE_TYPES + 1)];
    let mut hold_available: Vec<u8> = Vec::with_capacity(n);
    let mut next_queue: Vec<f32> = vec![0.0; n * 5 * NUM_PIECE_TYPES];
    let mut move_numbers: Vec<f32> = Vec::with_capacity(n);
    let mut policy_targets: Vec<f32> = Vec::with_capacity(n * NUM_ACTIONS);
    let mut value_targets: Vec<f32> = Vec::with_capacity(n);
    let mut action_masks: Vec<u8> = Vec::with_capacity(n * NUM_ACTIONS);

    for (i, ex) in examples.iter().enumerate() {
        // Board (flatten from (20, 10) to 200)
        boards.extend(ex.board.iter().copied());

        // Current piece one-hot
        current_pieces[i * NUM_PIECE_TYPES + ex.current_piece] = 1.0;

        // Hold piece one-hot (7 = empty slot)
        if ex.hold_piece < NUM_PIECE_TYPES {
            hold_pieces[i * (NUM_PIECE_TYPES + 1) + ex.hold_piece] = 1.0;
        } else {
            hold_pieces[i * (NUM_PIECE_TYPES + 1) + NUM_PIECE_TYPES] = 1.0; // Empty
        }

        // Hold available
        hold_available.push(ex.hold_available as u8);

        // Next queue one-hot (5 slots x 7 piece types)
        for (j, &piece) in ex.next_queue.iter().take(5).enumerate() {
            next_queue[i * 5 * NUM_PIECE_TYPES + j * NUM_PIECE_TYPES + piece] = 1.0;
        }

        // Move number (normalized)
        move_numbers.push(ex.move_number as f32 / 100.0);

        // Policy targets
        policy_targets.extend(ex.policy.iter().copied());

        // Value target
        value_targets.push(ex.value);

        // Action mask
        action_masks.extend(ex.action_mask.iter().map(|&b| b as u8));
    }

    // Create NPZ file (zip with npy arrays)
    let file = File::create(filepath).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    // Helper to write an npy array to the zip
    fn write_npy_to_zip<T: npyz::Serialize + npyz::AutoSerialize + Copy>(
        zip: &mut ZipWriter<File>,
        options: FileOptions,
        name: &str,
        shape: &[u64],
        data: &[T],
    ) -> Result<(), String> {
        zip.start_file(name, options)
            .map_err(|e: zip::result::ZipError| e.to_string())?;

        // Write NPY format
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(shape)
            .writer(&mut buffer)
            .begin_nd()
            .map_err(|e: std::io::Error| e.to_string())?;
        writer
            .extend(data.iter().copied())
            .map_err(|e: std::io::Error| e.to_string())?;
        writer.finish().map_err(|e: std::io::Error| e.to_string())?;

        zip.write_all(buffer.get_ref())
            .map_err(|e: std::io::Error| e.to_string())?;
        Ok(())
    }

    // Write each array
    write_npy_to_zip(
        &mut zip,
        options,
        "boards.npy",
        &[n as u64, BOARD_HEIGHT as u64, BOARD_WIDTH as u64],
        &boards,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "current_pieces.npy",
        &[n as u64, NUM_PIECE_TYPES as u64],
        &current_pieces,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "hold_pieces.npy",
        &[n as u64, (NUM_PIECE_TYPES + 1) as u64],
        &hold_pieces,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "hold_available.npy",
        &[n as u64],
        &hold_available,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "next_queue.npy",
        &[n as u64, 5, NUM_PIECE_TYPES as u64],
        &next_queue,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "move_numbers.npy",
        &[n as u64],
        &move_numbers,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "policy_targets.npy",
        &[n as u64, NUM_ACTIONS as u64],
        &policy_targets,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "value_targets.npy",
        &[n as u64],
        &value_targets,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "action_masks.npy",
        &[n as u64, NUM_ACTIONS as u64],
        &action_masks,
    )?;

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}
