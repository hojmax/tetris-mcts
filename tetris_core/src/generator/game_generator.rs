//! Background Game Generator
//!
//! Spawns a worker thread that continuously generates self-play games
//! using MCTS and writes training data to disk.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::mcts::{MCTSAgent, MCTSConfig};

use super::npz::write_examples_to_npz;

/// Background game generator for parallel training.
///
/// Spawns a worker thread that continuously generates games using MCTS
/// and writes training data to disk in NPZ format.
/// Number of games to batch before writing to disk
const GAMES_PER_FLUSH: usize = 50;

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
    /// Maximum training examples to keep (FIFO eviction)
    max_examples: usize,
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
    #[pyo3(signature = (model_path, output_dir, config=None, max_moves=100, add_noise=true, max_examples=100_000))]
    pub fn new(
        model_path: String,
        output_dir: String,
        config: Option<MCTSConfig>,
        max_moves: u32,
        add_noise: bool,
        max_examples: usize,
    ) -> Self {
        GameGenerator {
            model_path: PathBuf::from(model_path),
            output_dir: PathBuf::from(output_dir),
            config: config.unwrap_or_default(),
            max_moves,
            add_noise,
            max_examples,
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
        let max_examples = self.max_examples;
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
                max_examples,
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
        max_examples: usize,
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

        // All examples accumulate here, written to single file periodically
        let mut all_examples: Vec<crate::mcts::TrainingExample> = Vec::new();
        let mut games_since_flush = 0;
        let data_file = output_dir.join("training_data.npz");

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

                // Add to buffer
                all_examples.extend(result.examples);

                // FIFO eviction: drop oldest examples if over limit
                if all_examples.len() > max_examples {
                    let excess = all_examples.len() - max_examples;
                    all_examples.drain(..excess);
                }

                games_since_flush += 1;

                // Update counters
                games_generated.fetch_add(1, Ordering::SeqCst);
                examples_generated.fetch_add(num_examples, Ordering::SeqCst);

                // Flush to single file periodically
                if games_since_flush >= GAMES_PER_FLUSH {
                    if let Err(e) = write_examples_to_npz(&data_file, &all_examples) {
                        eprintln!("[GameGenerator] Failed to write NPZ: {}", e);
                    }
                    games_since_flush = 0;
                }
            }
        }

        // Final flush
        if !all_examples.is_empty() {
            let _ = write_examples_to_npz(&data_file, &all_examples);
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
