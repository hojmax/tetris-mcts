//! Background Game Generator
//!
//! Spawns a worker thread that continuously generates self-play games
//! using MCTS. Training data is kept in a shared in-memory buffer that
//! Python can sample from directly, avoiding disk I/O during training.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::constants::{AUX_FEATURES, BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES};
use crate::env::TetrisEnv;
use crate::mcts::GameStats;
use crate::mcts::{GameResult, GameTreeStats, MCTSAgent, MCTSConfig, TrainingExample, NUM_ACTIONS};

use super::npz::{read_examples_from_npz, write_examples_slices_to_npz};
use super::types::{GameReplay, ReplayMove};

/// Aggregate game statistics (thread-safe counters).
#[derive(Default)]
struct SharedStats {
    games_with_attack: AtomicU32,
    games_with_lines: AtomicU32,
    // Line clears
    singles: AtomicU32,
    doubles: AtomicU32,
    triples: AtomicU32,
    tetrises: AtomicU32,
    // T-spins
    tspin_minis: AtomicU32,
    tspin_singles: AtomicU32,
    tspin_doubles: AtomicU32,
    tspin_triples: AtomicU32,
    // Other
    perfect_clears: AtomicU32,
    back_to_backs: AtomicU32,
    max_combo: AtomicU32,
    total_lines: AtomicU32,
    total_attack: AtomicU32,
    holds: AtomicU32,
}

impl SharedStats {
    fn new() -> Self {
        Self::default()
    }

    /// Add stats from a completed game.
    fn add(&self, stats: &GameStats, attack: u32) {
        if attack > 0 {
            self.games_with_attack.fetch_add(1, Ordering::Relaxed);
        }
        if stats.total_lines > 0 {
            self.games_with_lines.fetch_add(1, Ordering::Relaxed);
        }
        self.singles.fetch_add(stats.singles, Ordering::Relaxed);
        self.doubles.fetch_add(stats.doubles, Ordering::Relaxed);
        self.triples.fetch_add(stats.triples, Ordering::Relaxed);
        self.tetrises.fetch_add(stats.tetrises, Ordering::Relaxed);
        self.tspin_minis
            .fetch_add(stats.tspin_minis, Ordering::Relaxed);
        self.tspin_singles
            .fetch_add(stats.tspin_singles, Ordering::Relaxed);
        self.tspin_doubles
            .fetch_add(stats.tspin_doubles, Ordering::Relaxed);
        self.tspin_triples
            .fetch_add(stats.tspin_triples, Ordering::Relaxed);
        self.perfect_clears
            .fetch_add(stats.perfect_clears, Ordering::Relaxed);
        self.back_to_backs
            .fetch_add(stats.back_to_backs, Ordering::Relaxed);
        self.total_lines
            .fetch_add(stats.total_lines, Ordering::Relaxed);
        self.total_attack.fetch_add(attack, Ordering::Relaxed);
        self.holds.fetch_add(stats.holds, Ordering::Relaxed);
        // Update max combo (atomic max)
        let mut current = self.max_combo.load(Ordering::Relaxed);
        while stats.max_combo > current {
            match self.max_combo.compare_exchange_weak(
                current,
                stats.max_combo,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Get all stats as a HashMap for Python.
    fn to_dict(&self) -> HashMap<String, u32> {
        let mut d = HashMap::new();
        d.insert(
            "games_with_attack".to_string(),
            self.games_with_attack.load(Ordering::Relaxed),
        );
        d.insert(
            "games_with_lines".to_string(),
            self.games_with_lines.load(Ordering::Relaxed),
        );
        d.insert("singles".to_string(), self.singles.load(Ordering::Relaxed));
        d.insert("doubles".to_string(), self.doubles.load(Ordering::Relaxed));
        d.insert("triples".to_string(), self.triples.load(Ordering::Relaxed));
        d.insert(
            "tetrises".to_string(),
            self.tetrises.load(Ordering::Relaxed),
        );
        d.insert(
            "tspin_minis".to_string(),
            self.tspin_minis.load(Ordering::Relaxed),
        );
        d.insert(
            "tspin_singles".to_string(),
            self.tspin_singles.load(Ordering::Relaxed),
        );
        d.insert(
            "tspin_doubles".to_string(),
            self.tspin_doubles.load(Ordering::Relaxed),
        );
        d.insert(
            "tspin_triples".to_string(),
            self.tspin_triples.load(Ordering::Relaxed),
        );
        d.insert(
            "perfect_clears".to_string(),
            self.perfect_clears.load(Ordering::Relaxed),
        );
        d.insert(
            "back_to_backs".to_string(),
            self.back_to_backs.load(Ordering::Relaxed),
        );
        d.insert(
            "max_combo".to_string(),
            self.max_combo.load(Ordering::Relaxed),
        );
        d.insert(
            "total_lines".to_string(),
            self.total_lines.load(Ordering::Relaxed),
        );
        d.insert(
            "total_attack".to_string(),
            self.total_attack.load(Ordering::Relaxed),
        );
        d.insert("holds".to_string(), self.holds.load(Ordering::Relaxed));
        d
    }
}

/// Info about the last completed game (for per-game logging).
struct LastGameInfo {
    game_number: u64,
    stats: GameStats,
    total_attack: u32,
    avg_overhang_fields: f32,
    num_moves: u32,
    avg_valid_actions: f32,
    max_valid_actions: u32,
    tree_stats: GameTreeStats,
    cache_hits: u64,
    cache_misses: u64,
    cache_size: usize,
    tree_reuse_hits: u32,
    tree_reuse_misses: u32,
    tree_reuse_carry_fraction: f32,
    traversal_total: u64,
    traversal_expansions: u64,
    traversal_terminal_ends: u64,
    traversal_horizon_ends: u64,
    traversal_expansion_fraction: f32,
    traversal_terminal_fraction: f32,
    traversal_horizon_fraction: f32,
}

#[derive(Clone)]
struct CandidateModelRequest {
    model_path: PathBuf,
    model_step: u64,
    nn_value_weight: f32,
}

struct ModelEvalEvent {
    incumbent_step: u64,
    incumbent_uses_network: bool,
    incumbent_avg_attack: f32,
    incumbent_nn_value_weight: f32,
    candidate_step: u64,
    candidate_games: u64,
    candidate_avg_attack: f32,
    candidate_attack_variance: f32,
    candidate_nn_value_weight: f32,
    promoted_nn_value_weight: f32,
    promoted_death_penalty: f32,
    promoted_overhang_penalty_weight: f32,
    promoted: bool,
    auto_promoted: bool,
    evaluation_seconds: f32,
    /// Best (max attack) game replay, if available.
    best_game_replay: Option<GameReplay>,
    /// Worst (min attack) game replay, if available.
    worst_game_replay: Option<GameReplay>,
    /// Per-game results: (seed, attack, lines, moves)
    per_game_results: Vec<(u64, u32, u32, u32)>,
}

/// Shared replay buffer for thread-safe access between generator and trainer.
struct SharedBuffer {
    /// Training examples and logical index space, updated atomically under one lock.
    state: RwLock<SharedBufferState>,
    /// Maximum buffer size
    max_size: usize,
}

struct SharedBufferState {
    examples: VecDeque<TrainingExample>,
    end_index: u64,
}

impl SharedBuffer {
    fn new(max_size: usize) -> Self {
        SharedBuffer {
            state: RwLock::new(SharedBufferState {
                examples: VecDeque::with_capacity(max_size),
                end_index: 0,
            }),
            max_size,
        }
    }

    /// Add examples to the buffer, evicting oldest if over capacity.
    fn add_examples(&self, new_examples: Vec<TrainingExample>) {
        if new_examples.is_empty() {
            return;
        }
        let added_count = new_examples.len() as u64;
        let mut state = self.state.write().unwrap();
        state.end_index = state.end_index.saturating_add(added_count);
        state.examples.extend(new_examples);
        while state.examples.len() > self.max_size {
            state.examples.pop_front();
        }
    }

    /// Get current buffer size.
    fn len(&self) -> usize {
        self.state.read().unwrap().examples.len()
    }

    /// Stream buffer contents directly to an NPZ file under read lock.
    ///
    /// Holds the read lock for the duration of the write so that the snapshot is
    /// consistent. Workers calling `add_examples` will block until the save completes
    /// (typically 10-30 seconds), which is acceptable for a save every 30 minutes.
    fn persist_to_npz(&self, filepath: &Path) -> Result<(), String> {
        let state = self.state.read().unwrap();
        let (slice_a, slice_b) = state.examples.as_slices();
        write_examples_slices_to_npz(filepath, slice_a, slice_b)
    }

    /// Return the logical one-past-end index in replay index space.
    fn window_end_index(&self) -> u64 {
        self.state.read().unwrap().end_index
    }

    /// Snapshot the current replay FIFO window atomically.
    fn logical_window_snapshot(&self) -> Option<(u64, u64, Vec<TrainingExample>)> {
        let state = self.state.read().unwrap();
        let n = state.examples.len();
        if n == 0 {
            return None;
        }
        let window_end = state.end_index;
        let window_start = window_end.saturating_sub(n as u64);
        let snapshot = state.examples.iter().cloned().collect();
        Some((window_start, window_end, snapshot))
    }

    /// Fetch a replay delta slice atomically from logical replay index space.
    fn logical_delta_slice(
        &self,
        from_index: u64,
        max_examples: usize,
    ) -> Option<(u64, u64, u64, Vec<TrainingExample>)> {
        let state = self.state.read().unwrap();
        let n = state.examples.len();
        if n == 0 {
            return None;
        }

        let window_end = state.end_index;
        let window_start = window_end.saturating_sub(n as u64);
        let slice_start = from_index.max(window_start);
        let slice_end = (slice_start + max_examples as u64).min(window_end);
        let slice_len = slice_end.saturating_sub(slice_start) as usize;
        let offset = slice_start.saturating_sub(window_start) as usize;
        let slice_examples = if slice_len == 0 {
            Vec::new()
        } else {
            state
                .examples
                .iter()
                .skip(offset)
                .take(slice_len)
                .cloned()
                .collect()
        };

        Some((window_start, window_end, slice_start, slice_examples))
    }
}

#[pyclass]
pub struct GameGenerator {
    /// Initial model path used on startup and protected from cleanup.
    bootstrap_model_path: PathBuf,
    /// Full path to training_data.npz (for periodic saves and replay preload)
    training_data_path: PathBuf,
    /// MCTS configuration
    config: MCTSConfig,
    /// Maximum placements per game (hold actions do not count)
    max_placements: u32,
    /// Whether to add Dirichlet noise
    add_noise: bool,
    /// Wall-clock interval between disk saves (for resume capability)
    save_interval_seconds: f64,
    /// Number of worker threads
    num_workers: usize,
    /// Fixed seeds used by the evaluator worker for candidate games.
    candidate_eval_seeds: Vec<u64>,
    /// Number of simulations per move before the first promoted NN model.
    non_network_num_simulations: u32,
    /// Shared replay buffer (accessed by both worker threads and Python)
    buffer: Arc<SharedBuffer>,
    /// Whether the generator is running
    running: Arc<AtomicBool>,
    /// Number of games generated since start
    games_generated: Arc<AtomicU64>,
    /// Number of examples generated since start
    examples_generated: Arc<AtomicU64>,
    /// Aggregate game statistics
    game_stats: Arc<SharedStats>,
    /// Completed game stats queue for per-game logging
    completed_games: Arc<RwLock<VecDeque<LastGameInfo>>>,
    /// Most recent pending candidate model (latest wins, older pending models are dropped).
    pending_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
    /// Candidate currently under evaluation by the evaluator worker.
    evaluating_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
    /// Queue of evaluator decisions for Python-side logging.
    model_eval_events: Arc<RwLock<VecDeque<ModelEvalEvent>>>,
    /// Shared incumbent model path used by all non-evaluator workers.
    incumbent_model_path: Arc<RwLock<PathBuf>>,
    /// Whether the incumbent currently uses NN guidance or no-network bootstrap mode.
    incumbent_uses_network: Arc<AtomicBool>,
    /// Training step associated with the incumbent model.
    incumbent_model_step: Arc<AtomicU64>,
    /// Incremented whenever a candidate is promoted and workers should reload.
    incumbent_model_version: Arc<AtomicU64>,
    /// Current nn_value_weight used by incumbent NN-guided rollouts.
    incumbent_nn_value_weight: Arc<AtomicU32>,
    /// Current death penalty for incumbent search (zeroed when nn_value_weight reaches cap).
    incumbent_death_penalty: Arc<AtomicU32>,
    /// Current overhang penalty weight for incumbent search (zeroed when nn_value_weight reaches cap).
    incumbent_overhang_penalty_weight: Arc<AtomicU32>,
    /// Cap at which nn_value_weight triggers penalty removal.
    nn_value_weight_cap: f32,
    /// Average attack from the evaluation that promoted the current incumbent.
    incumbent_eval_avg_attack: Arc<AtomicU32>,
    /// Thread handles (for joining on stop)
    thread_handles: Vec<JoinHandle<()>>,
}

#[pymethods]
impl GameGenerator {
    #[new]
    #[pyo3(signature = (model_path, training_data_path, config=None, max_placements=100, add_noise=true, max_examples=100_000, save_interval_seconds=60.0, num_workers=3, initial_model_step=0, candidate_eval_seeds=None, start_with_network=true, non_network_num_simulations=3000, initial_incumbent_eval_avg_attack=0.0, nn_value_weight_cap=1.0))]
    pub fn new(
        model_path: String,
        training_data_path: String,
        config: Option<MCTSConfig>,
        max_placements: u32,
        add_noise: bool,
        max_examples: usize,
        save_interval_seconds: f64,
        num_workers: usize,
        initial_model_step: u64,
        candidate_eval_seeds: Option<Vec<u64>>,
        start_with_network: bool,
        non_network_num_simulations: u32,
        initial_incumbent_eval_avg_attack: f32,
        nn_value_weight_cap: f32,
    ) -> PyResult<Self> {
        if max_placements == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_placements must be > 0",
            ));
        }
        if max_examples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_examples must be > 0",
            ));
        }
        if !save_interval_seconds.is_finite() || save_interval_seconds < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "save_interval_seconds must be finite and >= 0",
            ));
        }
        if num_workers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_workers must be > 0",
            ));
        }
        let candidate_eval_seeds = candidate_eval_seeds.unwrap_or_else(|| (0..50).collect());
        if candidate_eval_seeds.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "candidate_eval_seeds must not be empty",
            ));
        }
        if non_network_num_simulations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "non_network_num_simulations must be > 0",
            ));
        }

        let mut resolved_config = config.unwrap_or_default();
        resolved_config.max_placements = max_placements;
        if !resolved_config.nn_value_weight.is_finite() || resolved_config.nn_value_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "config.nn_value_weight must be finite and >= 0",
            ));
        }
        let initial_nn_value_weight_bits = resolved_config.nn_value_weight.to_bits();
        let initial_death_penalty_bits = resolved_config.death_penalty.to_bits();
        let initial_overhang_penalty_weight_bits =
            resolved_config.overhang_penalty_weight.to_bits();
        let bootstrap_model_path = PathBuf::from(model_path);

        Ok(GameGenerator {
            bootstrap_model_path: bootstrap_model_path.clone(),
            training_data_path: PathBuf::from(training_data_path),
            config: resolved_config,
            max_placements,
            add_noise,
            save_interval_seconds,
            num_workers,
            candidate_eval_seeds,
            non_network_num_simulations,
            buffer: Arc::new(SharedBuffer::new(max_examples)),
            running: Arc::new(AtomicBool::new(false)),
            games_generated: Arc::new(AtomicU64::new(0)),
            examples_generated: Arc::new(AtomicU64::new(0)),
            game_stats: Arc::new(SharedStats::new()),
            completed_games: Arc::new(RwLock::new(VecDeque::new())),
            pending_candidate: Arc::new(RwLock::new(None)),
            evaluating_candidate: Arc::new(RwLock::new(None)),
            model_eval_events: Arc::new(RwLock::new(VecDeque::new())),
            incumbent_model_path: Arc::new(RwLock::new(bootstrap_model_path)),
            incumbent_uses_network: Arc::new(AtomicBool::new(start_with_network)),
            incumbent_model_step: Arc::new(AtomicU64::new(initial_model_step)),
            incumbent_model_version: Arc::new(AtomicU64::new(0)),
            incumbent_nn_value_weight: Arc::new(AtomicU32::new(initial_nn_value_weight_bits)),
            incumbent_death_penalty: Arc::new(AtomicU32::new(initial_death_penalty_bits)),
            incumbent_overhang_penalty_weight: Arc::new(AtomicU32::new(
                initial_overhang_penalty_weight_bits,
            )),
            nn_value_weight_cap,
            incumbent_eval_avg_attack: Arc::new(AtomicU32::new(
                initial_incumbent_eval_avg_attack.to_bits(),
            )),
            thread_handles: Vec::new(),
        })
    }

    /// Start background game generation.
    ///
    /// Spawns worker threads that continuously generate games and write
    /// them to training_data_path. One worker is dedicated to candidate
    /// evaluation and promotion decisions.
    pub fn start(&mut self) -> PyResult<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Generator is already running",
            ));
        }

        // Create training data parent directory if needed.
        if let Some(parent_dir) = self.training_data_path.parent() {
            if !parent_dir.as_os_str().is_empty() {
                fs::create_dir_all(parent_dir).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to create training data directory {}: {}",
                        parent_dir.display(),
                        e
                    ))
                })?;
            }
        }

        // Load existing replay buffer snapshot if present.
        if self.training_data_path.exists() {
            let loaded_examples =
                read_examples_from_npz(&self.training_data_path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to load replay data from {}: {}",
                        self.training_data_path.display(),
                        e
                    ))
                })?;

            if !loaded_examples.is_empty() {
                let loaded_examples_count = loaded_examples.len();
                let loaded_max_game_number = loaded_examples
                    .iter()
                    .map(|example| example.game_number)
                    .max()
                    .unwrap_or(0);
                self.buffer.add_examples(loaded_examples);
                let retained_examples_count = self.buffer.len();
                let replay_end_index = self.buffer.window_end_index();
                self.games_generated
                    .store(loaded_max_game_number, Ordering::SeqCst);
                self.examples_generated
                    .store(replay_end_index, Ordering::SeqCst);
                eprintln!(
                    "[GameGenerator] Loaded {} replay examples from {} (max_game_number={})",
                    retained_examples_count,
                    self.training_data_path.display(),
                    loaded_max_game_number
                );
                if retained_examples_count < loaded_examples_count {
                    eprintln!(
                        "[GameGenerator] Dropped {} replay examples due to max_examples cap",
                        loaded_examples_count - retained_examples_count
                    );
                }
            }
        }

        // Set running flag
        self.running.store(true, Ordering::SeqCst);
        let evaluator_worker_id = self.num_workers - 1;

        // Spawn worker threads
        for worker_id in 0..self.num_workers {
            // Clone Arc handles for each thread
            let bootstrap_model_path = self.bootstrap_model_path.clone();
            let training_data_path = self.training_data_path.clone();
            let config = self.config.clone();
            let max_placements = self.max_placements;
            let add_noise = self.add_noise;
            let save_interval_seconds = self.save_interval_seconds;
            let candidate_eval_seeds = self.candidate_eval_seeds.clone();
            let non_network_num_simulations = self.non_network_num_simulations;
            let num_workers = self.num_workers;
            let is_evaluator_worker = worker_id == evaluator_worker_id;
            let buffer = Arc::clone(&self.buffer);
            let running = Arc::clone(&self.running);
            let games_generated = Arc::clone(&self.games_generated);
            let examples_generated = Arc::clone(&self.examples_generated);
            let game_stats = Arc::clone(&self.game_stats);
            let completed_games = Arc::clone(&self.completed_games);
            let pending_candidate = Arc::clone(&self.pending_candidate);
            let evaluating_candidate = Arc::clone(&self.evaluating_candidate);
            let model_eval_events = Arc::clone(&self.model_eval_events);
            let incumbent_model_path = Arc::clone(&self.incumbent_model_path);
            let incumbent_uses_network = Arc::clone(&self.incumbent_uses_network);
            let incumbent_model_step = Arc::clone(&self.incumbent_model_step);
            let incumbent_model_version = Arc::clone(&self.incumbent_model_version);
            let incumbent_nn_value_weight = Arc::clone(&self.incumbent_nn_value_weight);
            let incumbent_death_penalty = Arc::clone(&self.incumbent_death_penalty);
            let incumbent_overhang_penalty_weight =
                Arc::clone(&self.incumbent_overhang_penalty_weight);
            let nn_value_weight_cap = self.nn_value_weight_cap;
            let incumbent_eval_avg_attack = Arc::clone(&self.incumbent_eval_avg_attack);

            let handle = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    num_workers,
                    is_evaluator_worker,
                    bootstrap_model_path,
                    training_data_path,
                    config,
                    max_placements,
                    add_noise,
                    save_interval_seconds,
                    candidate_eval_seeds,
                    non_network_num_simulations,
                    buffer,
                    running,
                    games_generated,
                    examples_generated,
                    game_stats,
                    completed_games,
                    pending_candidate,
                    evaluating_candidate,
                    model_eval_events,
                    incumbent_model_path,
                    incumbent_uses_network,
                    incumbent_model_step,
                    incumbent_model_version,
                    incumbent_nn_value_weight,
                    incumbent_death_penalty,
                    incumbent_overhang_penalty_weight,
                    nn_value_weight_cap,
                    incumbent_eval_avg_attack,
                );
            });

            self.thread_handles.push(handle);
        }

        Ok(())
    }

    /// Stop background game generation.
    ///
    /// Signals the worker threads to stop and waits for them to finish.
    pub fn stop(&mut self) -> PyResult<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        // Signal stop
        self.running.store(false, Ordering::SeqCst);

        // Wait for all threads to finish
        for handle in self.thread_handles.drain(..) {
            handle.join().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Worker thread panicked")
            })?;
        }

        self.cleanup_queued_candidate_artifacts();

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

    pub fn incumbent_model_step(&self) -> u64 {
        self.incumbent_model_step.load(Ordering::SeqCst)
    }

    pub fn incumbent_model_path(&self) -> String {
        self.incumbent_model_path
            .read()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }

    pub fn incumbent_uses_network(&self) -> bool {
        self.incumbent_uses_network.load(Ordering::SeqCst)
    }

    pub fn incumbent_eval_avg_attack(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_eval_avg_attack)
    }

    pub fn incumbent_nn_value_weight(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_nn_value_weight)
    }

    pub fn incumbent_death_penalty(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_death_penalty)
    }

    pub fn incumbent_overhang_penalty_weight(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_overhang_penalty_weight)
    }

    /// Queue a candidate model for evaluator-worker gating.
    ///
    /// If another candidate is already pending, it is dropped in favor of this one.
    /// Returns True when the candidate is queued, False when ignored as stale.
    pub fn queue_candidate_model(
        &self,
        model_path: String,
        model_step: u64,
        nn_value_weight: f32,
    ) -> PyResult<bool> {
        let candidate_path = PathBuf::from(model_path);
        if !nn_value_weight.is_finite() || nn_value_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "nn_value_weight must be finite and >= 0",
            ));
        }
        if !candidate_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Candidate model does not exist: {}",
                candidate_path.display()
            )));
        }

        let incumbent_step = self.incumbent_model_step.load(Ordering::SeqCst);
        let incumbent_path = self.incumbent_model_path.read().unwrap().clone();
        if model_step <= incumbent_step || candidate_path == incumbent_path {
            Self::remove_model_artifacts_if_safe(
                &candidate_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
            return Ok(false);
        }

        let request = CandidateModelRequest {
            model_path: candidate_path.clone(),
            model_step,
            nn_value_weight,
        };
        let replaced = {
            let mut pending = self.pending_candidate.write().unwrap();
            pending.replace(request)
        };
        if let Some(old_request) = replaced {
            if old_request.model_path != candidate_path {
                let evaluating_path = self
                    .evaluating_candidate
                    .read()
                    .unwrap()
                    .as_ref()
                    .map(|r| r.model_path.clone());
                Self::remove_model_artifacts_if_safe(
                    &old_request.model_path,
                    &self.bootstrap_model_path,
                    &incumbent_path,
                    evaluating_path.as_deref(),
                );
            }
        }
        Ok(true)
    }

    /// Get statistics as a dictionary.
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("games_generated".to_string(), self.games_generated());
        stats.insert("examples_generated".to_string(), self.examples_generated());
        stats.insert("is_running".to_string(), self.is_running() as u64);
        stats.insert("buffer_size".to_string(), self.buffer_size() as u64);
        stats.insert(
            "incumbent_model_step".to_string(),
            self.incumbent_model_step.load(Ordering::SeqCst),
        );
        stats.insert(
            "incumbent_uses_network".to_string(),
            self.incumbent_uses_network() as u64,
        );
        stats.insert(
            "incumbent_eval_avg_attack".to_string(),
            Self::load_atomic_f32(&self.incumbent_eval_avg_attack).to_bits() as u64,
        );
        stats
    }

    /// Get aggregate game statistics (line clears, T-spins, etc.)
    pub fn get_game_stats(&self) -> HashMap<String, u32> {
        self.game_stats.to_dict()
    }

    /// Drain all completed game stats in generation order.
    pub fn drain_completed_game_stats(&self) -> Vec<(u64, HashMap<String, f32>)> {
        let mut queue = self.completed_games.write().unwrap();
        let mut drained = Vec::with_capacity(queue.len());
        while let Some(info) = queue.pop_front() {
            let mut d = HashMap::new();
            d.insert("singles".to_string(), info.stats.singles as f32);
            d.insert("doubles".to_string(), info.stats.doubles as f32);
            d.insert("triples".to_string(), info.stats.triples as f32);
            d.insert("tetrises".to_string(), info.stats.tetrises as f32);
            d.insert("tspin_minis".to_string(), info.stats.tspin_minis as f32);
            d.insert("tspin_singles".to_string(), info.stats.tspin_singles as f32);
            d.insert("tspin_doubles".to_string(), info.stats.tspin_doubles as f32);
            d.insert("tspin_triples".to_string(), info.stats.tspin_triples as f32);
            d.insert(
                "perfect_clears".to_string(),
                info.stats.perfect_clears as f32,
            );
            d.insert("back_to_backs".to_string(), info.stats.back_to_backs as f32);
            d.insert("max_combo".to_string(), info.stats.max_combo as f32);
            d.insert("total_lines".to_string(), info.stats.total_lines as f32);
            d.insert("holds".to_string(), info.stats.holds as f32);
            d.insert("total_attack".to_string(), info.total_attack as f32);
            d.insert("avg_overhang".to_string(), info.avg_overhang_fields);
            d.insert("episode_length".to_string(), info.num_moves as f32);
            d.insert("avg_valid_actions".to_string(), info.avg_valid_actions);
            d.insert(
                "max_valid_actions".to_string(),
                info.max_valid_actions as f32,
            );
            // Tree statistics
            d.insert(
                "tree_avg_branching_factor".to_string(),
                info.tree_stats.avg_branching_factor,
            );
            d.insert("tree_avg_leaves".to_string(), info.tree_stats.avg_leaves);
            d.insert(
                "tree_avg_total_nodes".to_string(),
                info.tree_stats.avg_total_nodes,
            );
            d.insert(
                "tree_avg_max_depth".to_string(),
                info.tree_stats.avg_max_depth,
            );
            d.insert(
                "tree_max_attack".to_string(),
                info.tree_stats.max_tree_attack as f32,
            );
            // Board embedding cache statistics
            let total_lookups = info.cache_hits + info.cache_misses;
            let hit_rate = if total_lookups > 0 {
                info.cache_hits as f32 / total_lookups as f32
            } else {
                0.0
            };
            d.insert("cache_hit_rate".to_string(), hit_rate);
            d.insert("cache_hits".to_string(), info.cache_hits as f32);
            d.insert("cache_misses".to_string(), info.cache_misses as f32);
            d.insert("cache_size".to_string(), info.cache_size as f32);
            // Tree reuse statistics
            let tree_reuse_total = info.tree_reuse_hits + info.tree_reuse_misses;
            let tree_reuse_rate = if tree_reuse_total > 0 {
                info.tree_reuse_hits as f32 / tree_reuse_total as f32
            } else {
                0.0
            };
            d.insert("tree_reuse_rate".to_string(), tree_reuse_rate);
            d.insert("tree_reuse_hits".to_string(), info.tree_reuse_hits as f32);
            d.insert(
                "tree_reuse_misses".to_string(),
                info.tree_reuse_misses as f32,
            );
            d.insert(
                "tree_reuse_carry_fraction".to_string(),
                info.tree_reuse_carry_fraction,
            );
            // Traversal outcome statistics
            d.insert("traversal_total".to_string(), info.traversal_total as f32);
            d.insert(
                "traversal_expansions".to_string(),
                info.traversal_expansions as f32,
            );
            d.insert(
                "traversal_terminal_ends".to_string(),
                info.traversal_terminal_ends as f32,
            );
            d.insert(
                "traversal_horizon_ends".to_string(),
                info.traversal_horizon_ends as f32,
            );
            d.insert(
                "traversal_expansion_fraction".to_string(),
                info.traversal_expansion_fraction,
            );
            d.insert(
                "traversal_terminal_fraction".to_string(),
                info.traversal_terminal_fraction,
            );
            d.insert(
                "traversal_horizon_fraction".to_string(),
                info.traversal_horizon_fraction,
            );
            drained.push((info.game_number, d));
        }
        drained
    }

    /// Drain evaluator decision events in generation order.
    pub fn drain_model_eval_events(&self, py: Python<'_>) -> Vec<HashMap<String, PyObject>> {
        let mut queue = self.model_eval_events.write().unwrap();
        let mut drained = Vec::with_capacity(queue.len());
        while let Some(event) = queue.pop_front() {
            let mut d: HashMap<String, PyObject> = HashMap::new();
            d.insert(
                "incumbent_step".into(),
                (event.incumbent_step as f64).into_py(py),
            );
            d.insert(
                "incumbent_uses_network".into(),
                (if event.incumbent_uses_network {
                    1.0
                } else {
                    0.0_f64
                })
                .into_py(py),
            );
            d.insert(
                "incumbent_avg_attack".into(),
                (event.incumbent_avg_attack as f64).into_py(py),
            );
            d.insert(
                "incumbent_nn_value_weight".into(),
                (event.incumbent_nn_value_weight as f64).into_py(py),
            );
            d.insert(
                "candidate_step".into(),
                (event.candidate_step as f64).into_py(py),
            );
            d.insert(
                "candidate_games".into(),
                (event.candidate_games as f64).into_py(py),
            );
            d.insert(
                "candidate_avg_attack".into(),
                (event.candidate_avg_attack as f64).into_py(py),
            );
            d.insert(
                "candidate_attack_variance".into(),
                (event.candidate_attack_variance as f64).into_py(py),
            );
            d.insert(
                "candidate_nn_value_weight".into(),
                (event.candidate_nn_value_weight as f64).into_py(py),
            );
            d.insert(
                "promoted_nn_value_weight".into(),
                (event.promoted_nn_value_weight as f64).into_py(py),
            );
            d.insert(
                "promoted_death_penalty".into(),
                (event.promoted_death_penalty as f64).into_py(py),
            );
            d.insert(
                "promoted_overhang_penalty_weight".into(),
                (event.promoted_overhang_penalty_weight as f64).into_py(py),
            );
            d.insert(
                "promoted".into(),
                (if event.promoted { 1.0 } else { 0.0_f64 }).into_py(py),
            );
            d.insert(
                "auto_promoted".into(),
                (if event.auto_promoted { 1.0 } else { 0.0_f64 }).into_py(py),
            );
            d.insert(
                "evaluation_seconds".into(),
                (event.evaluation_seconds as f64).into_py(py),
            );
            d.insert(
                "best_game_replay".into(),
                event.best_game_replay.into_py(py),
            );
            d.insert(
                "worst_game_replay".into(),
                event.worst_game_replay.into_py(py),
            );
            d.insert(
                "per_game_results".into(),
                event.per_game_results.into_py(py),
            );
            drained.push(d);
        }
        drained
    }

    /// Get the current number of examples in the replay buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Snapshot the full replay buffer as numpy arrays plus the logical start index.
    ///
    /// The returned `start_index` is the global index (exclusive-end counter based)
    /// of the first retained replay example in the current FIFO window.
    ///
    /// Returns None if the buffer is empty.
    pub fn replay_buffer_snapshot<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<
        Option<(
            u64,
            &'py PyArray2<f32>,
            &'py PyArray2<f32>,
            &'py PyArray2<f32>,
            &'py PyArray1<f32>,
            &'py PyArray1<f32>,
            &'py PyArray2<f32>,
        )>,
    > {
        let Some((start_index, _end_index, snapshot_examples)) =
            self.buffer.logical_window_snapshot()
        else {
            return Ok(None);
        };

        let arrays = Self::examples_to_numpy(py, &snapshot_examples)?;
        Ok(Some((
            start_index,
            arrays.0,
            arrays.1,
            arrays.2,
            arrays.3,
            arrays.4,
            arrays.5,
        )))
    }

    /// Fetch a delta slice from the replay buffer's logical index space.
    ///
    /// Returns:
    /// - `window_start_index`: global index of current FIFO window start
    /// - `window_end_index`: global index one-past current FIFO window end
    /// - `slice_start_index`: global index of first returned example
    /// - tensor arrays for up to `max_examples` examples in `[slice_start_index, window_end_index)`
    ///
    /// Returns None if the buffer is empty.
    #[pyo3(signature = (from_index, max_examples))]
    pub fn replay_buffer_delta<'py>(
        &self,
        py: Python<'py>,
        from_index: u64,
        max_examples: usize,
    ) -> PyResult<
        Option<(
            u64,
            u64,
            u64,
            &'py PyArray2<f32>,
            &'py PyArray2<f32>,
            &'py PyArray2<f32>,
            &'py PyArray1<f32>,
            &'py PyArray1<f32>,
            &'py PyArray2<f32>,
        )>,
    > {
        if max_examples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_examples must be > 0",
            ));
        }

        let Some((window_start, window_end, slice_start, slice_examples)) =
            self.buffer.logical_delta_slice(from_index, max_examples)
        else {
            return Ok(None);
        };

        let arrays = Self::examples_to_numpy(py, &slice_examples)?;
        Ok(Some((
            window_start,
            window_end,
            slice_start,
            arrays.0,
            arrays.1,
            arrays.2,
            arrays.3,
            arrays.4,
            arrays.5,
        )))
    }

    /// Sample a batch of training data from the replay buffer.
    ///
    /// Returns a tuple of numpy arrays:
    /// (
    ///   boards,
    ///   aux_features,
    ///   policy_targets,
    ///   value_targets,
    ///   overhang_fields,
    ///   action_masks,
    /// )
    ///
    /// Returns None if the buffer is empty.
    #[pyo3(signature = (batch_size))]
    pub fn sample_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
    ) -> PyResult<
        Option<(
            &'py PyArray2<f32>,
            &'py PyArray2<f32>,
            &'py PyArray2<f32>,
            &'py PyArray1<f32>,
            &'py PyArray1<f32>,
            &'py PyArray2<f32>,
        )>,
    > {
        // Snapshot sampled examples under lock, then release lock before heavy work.
        let sampled_examples: Vec<TrainingExample> = {
            let state = self.buffer.state.read().unwrap();
            let n = state.examples.len();
            if n == 0 {
                return Ok(None);
            }

            let actual_batch = batch_size.min(n);
            let mut rng = thread_rng();
            (0..actual_batch)
                .map(|_| {
                    let idx = rng.gen_range(0..n);
                    state.examples[idx].clone()
                })
                .collect()
        };
        let arrays = Self::examples_to_numpy(py, &sampled_examples)?;
        Ok(Some((
            arrays.0, arrays.1, arrays.2, arrays.3, arrays.4, arrays.5,
        )))
    }
}

impl GameGenerator {
    fn load_atomic_f32(value: &AtomicU32) -> f32 {
        f32::from_bits(value.load(Ordering::SeqCst))
    }

    fn store_atomic_f32(target: &AtomicU32, value: f32) {
        target.store(value.to_bits(), Ordering::SeqCst);
    }

    fn examples_to_numpy<'py>(
        py: Python<'py>,
        examples: &[TrainingExample],
    ) -> PyResult<(
        &'py PyArray2<f32>,
        &'py PyArray2<f32>,
        &'py PyArray2<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray2<f32>,
    )> {
        let batch_size = examples.len();
        let board_height = BOARD_HEIGHT;
        let board_width = BOARD_WIDTH;
        let num_actions = NUM_ACTIONS;
        let aux_features_size = AUX_FEATURES;

        let mut boards = vec![0.0f32; batch_size * board_height * board_width];
        let mut aux = vec![0.0f32; batch_size * aux_features_size];
        let mut policies = vec![0.0f32; batch_size * num_actions];
        let mut values = vec![0.0f32; batch_size];
        let mut overhangs = vec![0.0f32; batch_size];
        let mut masks = vec![0.0f32; batch_size * num_actions];

        for (i, ex) in examples.iter().enumerate() {
            for (j, &val) in ex.board.iter().enumerate() {
                boards[i * board_height * board_width + j] = val as f32;
            }

            let aux_offset = i * aux_features_size;
            let aux_slice = &mut aux[aux_offset..aux_offset + aux_features_size];
            let hold_piece = if ex.hold_piece < NUM_PIECE_TYPES {
                Some(ex.hold_piece)
            } else {
                None
            };
            crate::nn::encode_aux_features(
                aux_slice,
                ex.current_piece,
                hold_piece,
                ex.hold_available,
                &ex.next_queue,
                ex.placement_count,
                ex.combo,
                ex.back_to_back,
                &ex.next_hidden_piece_probs,
                &ex.column_heights,
                ex.max_column_height,
                &ex.row_fill_counts,
                ex.total_blocks,
                ex.bumpiness,
                ex.holes,
                ex.overhang_fields,
            )
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to encode aux features for sampled example {}: {}",
                    i, error
                ))
            })?;

            for (j, &val) in ex.policy.iter().enumerate() {
                policies[i * num_actions + j] = val;
            }
            values[i] = ex.value;
            overhangs[i] = ex.overhang_fields;
            for (j, &val) in ex.action_mask.iter().enumerate() {
                masks[i * num_actions + j] = if val { 1.0 } else { 0.0 };
            }
        }

        let boards_arr = PyArray1::from_vec(py, boards)
            .reshape([batch_size, board_height * board_width])
            .unwrap();
        let aux_arr = PyArray1::from_vec(py, aux)
            .reshape([batch_size, aux_features_size])
            .unwrap();
        let policies_arr = PyArray1::from_vec(py, policies)
            .reshape([batch_size, num_actions])
            .unwrap();
        let values_arr = PyArray1::from_vec(py, values);
        let overhangs_arr = PyArray1::from_vec(py, overhangs);
        let masks_arr = PyArray1::from_vec(py, masks)
            .reshape([batch_size, num_actions])
            .unwrap();

        Ok((
            boards_arr,
            aux_arr,
            policies_arr,
            values_arr,
            overhangs_arr,
            masks_arr,
        ))
    }

    fn persist_snapshot_if_due(
        training_data_path: &Path,
        buffer: &Arc<SharedBuffer>,
        save_interval_seconds: f64,
        next_snapshot_deadline: &mut Option<Instant>,
    ) {
        if save_interval_seconds <= 0.0 {
            return;
        }
        let interval = Duration::from_secs_f64(save_interval_seconds);
        let now = Instant::now();
        let deadline = *next_snapshot_deadline.get_or_insert_with(|| now + interval);
        if now < deadline {
            return;
        }

        Self::persist_buffer_snapshot(training_data_path, buffer);

        let mut next_deadline = deadline + interval;
        while next_deadline <= now {
            next_deadline += interval;
        }
        *next_snapshot_deadline = Some(next_deadline);
    }

    /// Worker thread main loop.
    fn worker_loop(
        worker_id: usize,
        num_workers: usize,
        is_evaluator_worker: bool,
        bootstrap_model_path: PathBuf,
        training_data_path: PathBuf,
        config: MCTSConfig,
        max_placements: u32,
        add_noise: bool,
        save_interval_seconds: f64,
        candidate_eval_seeds: Vec<u64>,
        non_network_num_simulations: u32,
        buffer: Arc<SharedBuffer>,
        running: Arc<AtomicBool>,
        games_generated: Arc<AtomicU64>,
        examples_generated: Arc<AtomicU64>,
        game_stats: Arc<SharedStats>,
        completed_games: Arc<RwLock<VecDeque<LastGameInfo>>>,
        pending_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
        evaluating_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
        model_eval_events: Arc<RwLock<VecDeque<ModelEvalEvent>>>,
        incumbent_model_path: Arc<RwLock<PathBuf>>,
        incumbent_uses_network: Arc<AtomicBool>,
        incumbent_model_step: Arc<AtomicU64>,
        incumbent_model_version: Arc<AtomicU64>,
        incumbent_nn_value_weight: Arc<AtomicU32>,
        incumbent_death_penalty: Arc<AtomicU32>,
        incumbent_overhang_penalty_weight: Arc<AtomicU32>,
        nn_value_weight_cap: f32,
        incumbent_eval_avg_attack: Arc<AtomicU32>,
    ) {
        let mut agent = MCTSAgent::new(config.clone());
        let mut loaded_model_version = u64::MAX;
        let mut loaded_model_step = 0u64;
        let mut loaded_with_network = !incumbent_uses_network.load(Ordering::SeqCst);

        while running.load(Ordering::SeqCst)
            && !Self::sync_incumbent_agent_if_needed(
                &config,
                non_network_num_simulations,
                &mut agent,
                &incumbent_model_path,
                &incumbent_uses_network,
                &incumbent_model_step,
                &incumbent_model_version,
                &incumbent_nn_value_weight,
                &incumbent_death_penalty,
                &incumbent_overhang_penalty_weight,
                &mut loaded_model_version,
                &mut loaded_model_step,
                &mut loaded_with_network,
                worker_id,
                num_workers,
            )
        {
            thread::sleep(Duration::from_millis(500));
        }

        // Only worker 0 handles disk saves to avoid race conditions
        let is_save_worker = worker_id == 0;
        let mut next_snapshot_deadline = if save_interval_seconds > 0.0 {
            Some(Instant::now() + Duration::from_secs_f64(save_interval_seconds))
        } else {
            None
        };

        // Main generation loop
        while running.load(Ordering::SeqCst) {
            if !Self::sync_incumbent_agent_if_needed(
                &config,
                non_network_num_simulations,
                &mut agent,
                &incumbent_model_path,
                &incumbent_uses_network,
                &incumbent_model_step,
                &incumbent_model_version,
                &incumbent_nn_value_weight,
                &incumbent_death_penalty,
                &incumbent_overhang_penalty_weight,
                &mut loaded_model_version,
                &mut loaded_model_step,
                &mut loaded_with_network,
                worker_id,
                num_workers,
            ) {
                thread::sleep(Duration::from_millis(200));
                continue;
            }

            if is_evaluator_worker {
                let maybe_candidate = {
                    let mut pending = pending_candidate.write().unwrap();
                    pending.take()
                };
                if let Some(candidate) = maybe_candidate {
                    let current_incumbent_step = incumbent_model_step.load(Ordering::SeqCst);
                    if candidate.model_step <= current_incumbent_step {
                        let incumbent_path = incumbent_model_path.read().unwrap().clone();
                        Self::remove_model_artifacts_if_safe(
                            &candidate.model_path,
                            &bootstrap_model_path,
                            &incumbent_path,
                            None,
                        );
                        continue;
                    }

                    {
                        let mut evaluating = evaluating_candidate.write().unwrap();
                        *evaluating = Some(candidate.clone());
                    }

                    let _committed_games = Self::run_candidate_evaluation(
                        worker_id,
                        candidate,
                        &config,
                        &running,
                        max_placements,
                        &candidate_eval_seeds,
                        add_noise,
                        non_network_num_simulations,
                        &buffer,
                        &games_generated,
                        &examples_generated,
                        &game_stats,
                        &completed_games,
                        &model_eval_events,
                        &bootstrap_model_path,
                        &incumbent_model_path,
                        &incumbent_uses_network,
                        &incumbent_model_step,
                        &incumbent_model_version,
                        &incumbent_nn_value_weight,
                        &incumbent_death_penalty,
                        &incumbent_overhang_penalty_weight,
                        nn_value_weight_cap,
                        &incumbent_eval_avg_attack,
                    );

                    {
                        let mut evaluating = evaluating_candidate.write().unwrap();
                        *evaluating = None;
                    }
                    loaded_model_version = u64::MAX;
                    loaded_with_network = !incumbent_uses_network.load(Ordering::SeqCst);

                    if is_save_worker {
                        Self::persist_snapshot_if_due(
                            &training_data_path,
                            &buffer,
                            save_interval_seconds,
                            &mut next_snapshot_deadline,
                        );
                    }
                    continue;
                }
            }

            // Play one game
            if let Some(result) = agent.play_game(max_placements, add_noise) {
                Self::commit_game_result(
                    result,
                    &buffer,
                    &games_generated,
                    &examples_generated,
                    &game_stats,
                    &completed_games,
                );

                // Periodically save to disk for resume capability based on
                // wall-clock time.
                if is_save_worker {
                    Self::persist_snapshot_if_due(
                        &training_data_path,
                        &buffer,
                        save_interval_seconds,
                        &mut next_snapshot_deadline,
                    );
                }
            }
        }

        // Final save on shutdown (only worker 0)
        if is_save_worker && buffer.len() > 0 {
            let n = buffer.len();
            let _ = buffer.persist_to_npz(&training_data_path);
            eprintln!("[GameGenerator] Saved {} examples to disk", n);
        }

        eprintln!("[GameGenerator] Worker {} exiting", worker_id);
    }

    fn sync_incumbent_agent_if_needed(
        config: &MCTSConfig,
        non_network_num_simulations: u32,
        agent: &mut MCTSAgent,
        incumbent_model_path: &Arc<RwLock<PathBuf>>,
        incumbent_uses_network: &Arc<AtomicBool>,
        incumbent_model_step: &Arc<AtomicU64>,
        incumbent_model_version: &Arc<AtomicU64>,
        incumbent_nn_value_weight: &Arc<AtomicU32>,
        incumbent_death_penalty: &Arc<AtomicU32>,
        incumbent_overhang_penalty_weight: &Arc<AtomicU32>,
        loaded_model_version: &mut u64,
        loaded_model_step: &mut u64,
        loaded_with_network: &mut bool,
        worker_id: usize,
        num_workers: usize,
    ) -> bool {
        let target_version = incumbent_model_version.load(Ordering::SeqCst);
        let target_uses_network = incumbent_uses_network.load(Ordering::SeqCst);
        if *loaded_model_version == target_version && *loaded_with_network == target_uses_network {
            return true;
        }

        let model_path = if target_uses_network {
            Some(incumbent_model_path.read().unwrap().clone())
        } else {
            None
        };
        let target_nn_value_weight = Self::load_atomic_f32(incumbent_nn_value_weight);
        let target_death_penalty = Self::load_atomic_f32(incumbent_death_penalty);
        let target_overhang_penalty_weight =
            Self::load_atomic_f32(incumbent_overhang_penalty_weight);
        let Some(new_agent) = Self::create_rollout_agent(
            config,
            target_uses_network,
            non_network_num_simulations,
            target_nn_value_weight,
            target_death_penalty,
            target_overhang_penalty_weight,
            model_path.as_deref(),
            worker_id,
            "incumbent",
        ) else {
            return false;
        };

        *loaded_model_step = incumbent_model_step.load(Ordering::SeqCst);
        if worker_id == 0 {
            if target_uses_network {
                eprintln!(
                    "[GameGenerator] Loaded incumbent NN model step {} ({} workers, sims={}, nn_value_weight={:.6}, death_penalty={:.3}, overhang_penalty_weight={:.3})",
                    *loaded_model_step, num_workers, config.num_simulations, target_nn_value_weight, target_death_penalty, target_overhang_penalty_weight
                );
            } else {
                eprintln!(
                    "[GameGenerator] Using no-network incumbent at step {} ({} workers, sims={})",
                    *loaded_model_step, num_workers, non_network_num_simulations
                );
            }
        }

        *agent = new_agent;
        *loaded_model_version = target_version;
        *loaded_with_network = target_uses_network;
        true
    }

    fn run_candidate_evaluation(
        worker_id: usize,
        candidate: CandidateModelRequest,
        config: &MCTSConfig,
        running: &Arc<AtomicBool>,
        max_placements: u32,
        candidate_eval_seeds: &[u64],
        add_noise: bool,
        non_network_num_simulations: u32,
        buffer: &Arc<SharedBuffer>,
        games_generated: &Arc<AtomicU64>,
        examples_generated: &Arc<AtomicU64>,
        game_stats: &Arc<SharedStats>,
        completed_games: &Arc<RwLock<VecDeque<LastGameInfo>>>,
        model_eval_events: &Arc<RwLock<VecDeque<ModelEvalEvent>>>,
        bootstrap_model_path: &Path,
        incumbent_model_path: &Arc<RwLock<PathBuf>>,
        incumbent_uses_network: &Arc<AtomicBool>,
        incumbent_model_step: &Arc<AtomicU64>,
        incumbent_model_version: &Arc<AtomicU64>,
        incumbent_nn_value_weight: &Arc<AtomicU32>,
        incumbent_death_penalty: &Arc<AtomicU32>,
        incumbent_overhang_penalty_weight: &Arc<AtomicU32>,
        nn_value_weight_cap: f32,
        incumbent_eval_avg_attack: &Arc<AtomicU32>,
    ) -> usize {
        // Read current penalty values for candidate evaluation
        let candidate_death_penalty = Self::load_atomic_f32(incumbent_death_penalty);
        let candidate_overhang_penalty_weight =
            Self::load_atomic_f32(incumbent_overhang_penalty_weight);
        let Some(candidate_agent) = Self::create_rollout_agent(
            config,
            true,
            non_network_num_simulations,
            candidate.nn_value_weight,
            candidate_death_penalty,
            candidate_overhang_penalty_weight,
            Some(&candidate.model_path),
            worker_id,
            "candidate",
        ) else {
            eprintln!(
                "[GameGenerator] Worker {} failed to load candidate step {} from {}",
                worker_id,
                candidate.model_step,
                candidate.model_path.display()
            );
            let incumbent_path = incumbent_model_path.read().unwrap().clone();
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                bootstrap_model_path,
                &incumbent_path,
                None,
            );
            return 0;
        };

        let eval_start = Instant::now();

        // Play candidate games on fixed seeds for consistent benchmarking.
        struct CandidateGameResult {
            seed: u64,
            game_result: GameResult,
            replay_moves: Vec<ReplayMove>,
        }

        let mut candidate_results: Vec<CandidateGameResult> =
            Vec::with_capacity(candidate_eval_seeds.len());
        for &seed in candidate_eval_seeds {
            if !running.load(Ordering::SeqCst) {
                let incumbent_path = incumbent_model_path.read().unwrap().clone();
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    bootstrap_model_path,
                    &incumbent_path,
                    None,
                );
                return 0;
            }
            let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed);
            if let Some((result, replay)) =
                candidate_agent.play_game_on_env(env, max_placements, add_noise)
            {
                candidate_results.push(CandidateGameResult {
                    seed,
                    game_result: result,
                    replay_moves: replay,
                });
            }
        }

        if candidate_results.is_empty() {
            let incumbent_path = incumbent_model_path.read().unwrap().clone();
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                bootstrap_model_path,
                &incumbent_path,
                None,
            );
            return 0;
        }

        // Build per-game results and find best/worst for replay serialization
        let per_game_results: Vec<(u64, u32, u32, u32)> = candidate_results
            .iter()
            .map(|r| {
                (
                    r.seed,
                    r.game_result.total_attack,
                    r.game_result.stats.total_lines,
                    r.game_result.num_moves,
                )
            })
            .collect();

        let best_idx = candidate_results
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.game_result.total_attack)
            .map(|(i, _)| i);
        let worst_idx = candidate_results
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| r.game_result.total_attack)
            .map(|(i, _)| i);

        let build_replay = |idx: usize| -> GameReplay {
            let r = &candidate_results[idx];
            GameReplay {
                seed: r.seed,
                moves: r.replay_moves.clone(),
                total_attack: r.game_result.total_attack,
                num_moves: r.game_result.num_moves,
            }
        };

        let best_game_replay = best_idx.map(build_replay);
        let worst_game_replay = worst_idx.and_then(|i| {
            // Avoid duplicating if best and worst are the same game
            if Some(i) == best_idx {
                None
            } else {
                Some(build_replay(i))
            }
        });

        let candidate_games = candidate_results.len() as u64;
        let candidate_total_attack: u64 = candidate_results
            .iter()
            .map(|r| r.game_result.total_attack as u64)
            .sum();
        let candidate_avg_attack = candidate_total_attack as f32 / candidate_games as f32;
        let candidate_attack_variance = candidate_results
            .iter()
            .map(|r| {
                let diff = r.game_result.total_attack as f32 - candidate_avg_attack;
                diff * diff
            })
            .sum::<f32>()
            / candidate_games as f32;

        let incumbent_avg_attack = Self::load_atomic_f32(incumbent_eval_avg_attack);
        let incumbent_nn_value_weight_before = Self::load_atomic_f32(incumbent_nn_value_weight);
        let incumbent_uses_network_before = incumbent_uses_network.load(Ordering::SeqCst);
        let incumbent_step_before = incumbent_model_step.load(Ordering::SeqCst);
        let auto_promoted = incumbent_avg_attack == 0.0 && !incumbent_uses_network_before;
        let promoted = auto_promoted || candidate_avg_attack > incumbent_avg_attack;
        let promoted_nn_value_weight = if promoted {
            candidate.nn_value_weight
        } else {
            incumbent_nn_value_weight_before
        };

        let committed_games = if promoted {
            let previous_incumbent_path = incumbent_model_path.read().unwrap().clone();
            {
                let mut incumbent_path = incumbent_model_path.write().unwrap();
                *incumbent_path = candidate.model_path.clone();
            }
            incumbent_uses_network.store(true, Ordering::SeqCst);
            incumbent_model_step.store(candidate.model_step, Ordering::SeqCst);
            Self::store_atomic_f32(incumbent_nn_value_weight, candidate.nn_value_weight);
            if candidate.nn_value_weight >= nn_value_weight_cap {
                Self::store_atomic_f32(incumbent_death_penalty, 0.0);
                Self::store_atomic_f32(incumbent_overhang_penalty_weight, 0.0);
                eprintln!(
                    "[GameGenerator] nn_value_weight reached cap ({:.6}), disabling death_penalty and overhang_penalty_weight",
                    nn_value_weight_cap
                );
            }
            incumbent_model_version.fetch_add(1, Ordering::SeqCst);
            Self::store_atomic_f32(incumbent_eval_avg_attack, candidate_avg_attack);

            let mut committed = 0usize;
            for r in candidate_results {
                Self::commit_game_result(
                    r.game_result,
                    buffer,
                    games_generated,
                    examples_generated,
                    game_stats,
                    completed_games,
                );
                committed += 1;
            }

            Self::remove_model_artifacts_if_safe(
                &previous_incumbent_path,
                bootstrap_model_path,
                &candidate.model_path,
                None,
            );

            eprintln!(
                "[GameGenerator] Promoted candidate step {} (avg_attack {:.3} > incumbent {:.3}, games={}, nn_value_weight {:.6} -> {:.6})",
                candidate.model_step,
                candidate_avg_attack,
                incumbent_avg_attack,
                candidate_games,
                incumbent_nn_value_weight_before,
                candidate.nn_value_weight
            );
            committed
        } else {
            let incumbent_path = incumbent_model_path.read().unwrap().clone();
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                bootstrap_model_path,
                &incumbent_path,
                None,
            );

            eprintln!(
                "[GameGenerator] Rejected candidate step {} (avg_attack {:.3} <= incumbent {:.3}, games={}, candidate_nn_value_weight {:.6}, incumbent_nn_value_weight {:.6})",
                candidate.model_step,
                candidate_avg_attack,
                incumbent_avg_attack,
                candidate_games,
                candidate.nn_value_weight,
                incumbent_nn_value_weight_before
            );
            0
        };

        let evaluation_seconds = eval_start.elapsed().as_secs_f32();
        let promoted_death_penalty = Self::load_atomic_f32(incumbent_death_penalty);
        let promoted_overhang_penalty_weight =
            Self::load_atomic_f32(incumbent_overhang_penalty_weight);

        model_eval_events
            .write()
            .unwrap()
            .push_back(ModelEvalEvent {
                incumbent_step: incumbent_step_before,
                incumbent_uses_network: incumbent_uses_network_before,
                incumbent_avg_attack,
                incumbent_nn_value_weight: incumbent_nn_value_weight_before,
                candidate_step: candidate.model_step,
                candidate_games,
                candidate_avg_attack,
                candidate_attack_variance,
                candidate_nn_value_weight: candidate.nn_value_weight,
                promoted_nn_value_weight,
                promoted_death_penalty,
                promoted_overhang_penalty_weight,
                promoted,
                auto_promoted,
                evaluation_seconds,
                best_game_replay,
                worst_game_replay,
                per_game_results,
            });

        committed_games
    }

    fn build_rollout_config(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        nn_value_weight: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
    ) -> MCTSConfig {
        let mut rollout_config = base_config.clone();
        rollout_config.num_simulations = if uses_network {
            base_config.num_simulations
        } else {
            non_network_num_simulations
        };
        rollout_config.nn_value_weight = nn_value_weight;
        rollout_config.death_penalty = death_penalty;
        rollout_config.overhang_penalty_weight = overhang_penalty_weight;
        if !uses_network {
            rollout_config.q_scale = None;
        }
        rollout_config
    }

    fn create_rollout_agent(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        nn_value_weight: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
        model_path: Option<&Path>,
        worker_id: usize,
        role: &str,
    ) -> Option<MCTSAgent> {
        // Keep rollout behavior consistent across training and evaluator code paths,
        // including visit_sampling_epsilon and all other shared MCTS settings.
        let rollout_config = Self::build_rollout_config(
            base_config,
            uses_network,
            non_network_num_simulations,
            nn_value_weight,
            death_penalty,
            overhang_penalty_weight,
        );
        let mut agent = MCTSAgent::new(rollout_config);
        if uses_network {
            let model_path = model_path.expect("network rollout requires model path");
            let path_str = model_path
                .to_str()
                .expect("Model path contains invalid UTF-8");
            if !agent.load_model(path_str) {
                eprintln!(
                    "[GameGenerator] Worker {} failed to load {} model {}",
                    worker_id,
                    role,
                    model_path.display()
                );
                return None;
            }
        }
        Some(agent)
    }

    fn commit_game_result(
        result: GameResult,
        buffer: &Arc<SharedBuffer>,
        games_generated: &Arc<AtomicU64>,
        examples_generated: &Arc<AtomicU64>,
        game_stats: &Arc<SharedStats>,
        completed_games: &Arc<RwLock<VecDeque<LastGameInfo>>>,
    ) {
        let GameResult {
            mut examples,
            total_attack,
            num_moves,
            avg_valid_actions,
            max_valid_actions,
            stats,
            tree_stats,
            avg_overhang_fields,
            cache_hits,
            cache_misses,
            cache_size,
            tree_reuse_hits,
            tree_reuse_misses,
            tree_reuse_carry_fraction,
            traversal_total,
            traversal_expansions,
            traversal_terminal_ends,
            traversal_horizon_ends,
            traversal_expansion_fraction,
            traversal_terminal_fraction,
            traversal_horizon_fraction,
            ..
        } = result;

        let game_number = games_generated.fetch_add(1, Ordering::SeqCst) + 1;
        for example in &mut examples {
            example.game_number = game_number;
            example.game_total_attack = total_attack;
        }

        let num_examples = examples.len() as u64;
        buffer.add_examples(examples);

        examples_generated.fetch_add(num_examples, Ordering::SeqCst);
        game_stats.add(&stats, total_attack);

        completed_games.write().unwrap().push_back(LastGameInfo {
            game_number,
            stats,
            total_attack,
            avg_overhang_fields,
            num_moves,
            avg_valid_actions,
            max_valid_actions,
            tree_stats,
            cache_hits,
            cache_misses,
            cache_size,
            tree_reuse_hits,
            tree_reuse_misses,
            tree_reuse_carry_fraction,
            traversal_total,
            traversal_expansions,
            traversal_terminal_ends,
            traversal_horizon_ends,
            traversal_expansion_fraction,
            traversal_terminal_fraction,
            traversal_horizon_fraction,
        });
    }

    fn persist_buffer_snapshot(training_data_path: &Path, buffer: &Arc<SharedBuffer>) {
        if let Err(error) = buffer.persist_to_npz(training_data_path) {
            eprintln!("[GameGenerator] Failed to write NPZ: {}", error);
        }
    }

    fn model_artifact_paths(model_path: &Path) -> [PathBuf; 7] {
        let base_path = model_path.with_extension("");
        [
            model_path.to_path_buf(),
            model_path.with_extension("onnx.data"),
            base_path.with_extension("conv.onnx"),
            base_path.with_extension("conv.onnx.data"),
            base_path.with_extension("heads.onnx"),
            base_path.with_extension("heads.onnx.data"),
            base_path.with_extension("fc.bin"),
        ]
    }

    fn remove_model_artifacts(model_path: &Path) {
        for artifact_path in Self::model_artifact_paths(model_path) {
            if let Err(error) = fs::remove_file(&artifact_path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    eprintln!(
                        "[GameGenerator] Failed to remove model artifact {}: {}",
                        artifact_path.display(),
                        error
                    );
                }
            }
        }
    }

    fn remove_model_artifacts_if_safe(
        model_path: &Path,
        bootstrap_model_path: &Path,
        incumbent_model_path: &Path,
        evaluating_model_path: Option<&Path>,
    ) {
        if model_path == bootstrap_model_path {
            return;
        }
        if model_path == incumbent_model_path {
            return;
        }
        if let Some(path) = evaluating_model_path {
            if model_path == path {
                return;
            }
        }
        Self::remove_model_artifacts(model_path);
    }

    fn cleanup_queued_candidate_artifacts(&self) {
        let incumbent_path = self.incumbent_model_path.read().unwrap().clone();

        let pending = self.pending_candidate.write().unwrap().take();
        if let Some(candidate) = pending {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
        }

        let evaluating = self.evaluating_candidate.write().unwrap().take();
        if let Some(candidate) = evaluating {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
        }
    }
}

impl Drop for GameGenerator {
    fn drop(&mut self) {
        // Ensure all threads are stopped when generator is dropped
        self.running.store(false, Ordering::SeqCst);
        for handle in self.thread_handles.drain(..) {
            let _ = handle.join();
        }
        self.cleanup_queued_candidate_artifacts();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, ROW_FILL_FEATURE_ROWS};
    use crate::generator::test_utils;
    use crate::mcts::NUM_ACTIONS;
    use std::fs;

    fn make_example(move_number: u32) -> TrainingExample {
        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[0] = 1.0;
        let mut action_mask = vec![false; NUM_ACTIONS];
        action_mask[0] = true;

        TrainingExample {
            board: vec![0; BOARD_HEIGHT * BOARD_WIDTH],
            current_piece: 0,
            hold_piece: 7,
            hold_available: true,
            next_queue: vec![0, 1, 2, 3, 4],
            move_number,
            placement_count: move_number as f32 / 100.0,
            combo: 0.0,
            back_to_back: false,
            next_hidden_piece_probs: vec![1.0 / NUM_PIECE_TYPES as f32; NUM_PIECE_TYPES],
            column_heights: vec![0.0; BOARD_WIDTH],
            max_column_height: 0.0,
            row_fill_counts: vec![0.0; ROW_FILL_FEATURE_ROWS],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy,
            value: move_number as f32,
            action_mask,
            overhang_fields: crate::mcts::normalize_overhang_fields(move_number),
            game_number: 0,
            game_total_attack: 0,
        }
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        test_utils::unique_temp_path("tetris_generator", name)
    }

    #[test]
    fn test_shared_buffer_fifo_eviction() {
        let buffer = SharedBuffer::new(3);
        buffer.add_examples(vec![make_example(1), make_example(2)]);
        buffer.add_examples(vec![make_example(3), make_example(4)]);

        let (_, _, kept) = buffer.logical_window_snapshot().unwrap();
        assert_eq!(kept.len(), 3);
        let move_numbers: Vec<u32> = kept.iter().map(|e| e.move_number).collect();
        assert_eq!(move_numbers, vec![2, 3, 4]);
    }

    #[test]
    fn test_shared_buffer_logical_indices_follow_fifo_window() {
        let buffer = SharedBuffer::new(3);
        assert!(buffer.logical_window_snapshot().is_none());

        buffer.add_examples(vec![make_example(1), make_example(2)]);
        let (window_start, window_end, snapshot) = buffer
            .logical_window_snapshot()
            .expect("snapshot should exist after adding examples");
        assert_eq!(window_start, 0);
        assert_eq!(window_end, 2);
        let move_numbers: Vec<u32> = snapshot.iter().map(|e| e.move_number).collect();
        assert_eq!(move_numbers, vec![1, 2]);

        buffer.add_examples(vec![make_example(3), make_example(4)]);
        let (window_start, window_end, snapshot) = buffer
            .logical_window_snapshot()
            .expect("snapshot should exist after eviction");
        assert_eq!(window_start, 1);
        assert_eq!(window_end, 4);
        let move_numbers: Vec<u32> = snapshot.iter().map(|e| e.move_number).collect();
        assert_eq!(move_numbers, vec![2, 3, 4]);

        let (window_start, window_end, slice_start, slice) = buffer
            .logical_delta_slice(2, 10)
            .expect("delta should exist");
        assert_eq!(window_start, 1);
        assert_eq!(window_end, 4);
        assert_eq!(slice_start, 2);
        let move_numbers: Vec<u32> = slice.iter().map(|e| e.move_number).collect();
        assert_eq!(move_numbers, vec![3, 4]);

        let (window_start, window_end, slice_start, slice) = buffer
            .logical_delta_slice(0, 2)
            .expect("delta should clamp to window start");
        assert_eq!(window_start, 1);
        assert_eq!(window_end, 4);
        assert_eq!(slice_start, 1);
        let move_numbers: Vec<u32> = slice.iter().map(|e| e.move_number).collect();
        assert_eq!(move_numbers, vec![2, 3]);
    }

    #[test]
    fn test_shared_stats_accumulates_and_tracks_max_combo() {
        let stats = SharedStats::new();
        let game_a = GameStats {
            singles: 1,
            doubles: 0,
            triples: 0,
            tetrises: 0,
            tspin_minis: 0,
            tspin_singles: 0,
            tspin_doubles: 0,
            tspin_triples: 0,
            perfect_clears: 0,
            back_to_backs: 1,
            max_combo: 2,
            total_lines: 1,
            holds: 3,
        };
        let game_b = GameStats {
            singles: 0,
            doubles: 1,
            triples: 0,
            tetrises: 1,
            tspin_minis: 0,
            tspin_singles: 1,
            tspin_doubles: 0,
            tspin_triples: 0,
            perfect_clears: 1,
            back_to_backs: 0,
            max_combo: 5,
            total_lines: 6,
            holds: 4,
        };

        stats.add(&game_a, 0);
        stats.add(&game_b, 7);
        let d = stats.to_dict();

        assert_eq!(d["games_with_attack"], 1);
        assert_eq!(d["games_with_lines"], 2);
        assert_eq!(d["singles"], 1);
        assert_eq!(d["doubles"], 1);
        assert_eq!(d["tetrises"], 1);
        assert_eq!(d["tspin_singles"], 1);
        assert_eq!(d["perfect_clears"], 1);
        assert_eq!(d["back_to_backs"], 1);
        assert_eq!(d["max_combo"], 5);
        assert_eq!(d["total_lines"], 7);
        assert_eq!(d["total_attack"], 7);
        assert_eq!(d["holds"], 7);
    }

    #[test]
    fn test_build_rollout_config_keeps_sampling_settings() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 123;
        config.visit_sampling_epsilon = 0.42;
        config.temperature = 1.5;
        config.dirichlet_alpha = 0.02;
        config.dirichlet_epsilon = 0.3;
        config.nn_value_weight = 0.123;
        config.death_penalty = 5.0;
        config.overhang_penalty_weight = 3.0;
        config.q_scale = Some(7.5);

        let network_config =
            GameGenerator::build_rollout_config(&config, true, 999, 0.123, 5.0, 3.0);
        assert_eq!(network_config.num_simulations, 123);
        assert_eq!(network_config.visit_sampling_epsilon, 0.42);
        assert_eq!(network_config.temperature, 1.5);
        assert_eq!(network_config.dirichlet_alpha, 0.02);
        assert_eq!(network_config.dirichlet_epsilon, 0.3);
        assert_eq!(network_config.nn_value_weight, 0.123);
        assert_eq!(network_config.death_penalty, 5.0);
        assert_eq!(network_config.overhang_penalty_weight, 3.0);
        assert_eq!(network_config.q_scale, Some(7.5));

        let bootstrap_config =
            GameGenerator::build_rollout_config(&config, false, 999, 0.456, 5.0, 3.0);
        assert_eq!(bootstrap_config.num_simulations, 999);
        assert_eq!(bootstrap_config.visit_sampling_epsilon, 0.42);
        assert_eq!(bootstrap_config.temperature, 1.5);
        assert_eq!(bootstrap_config.dirichlet_alpha, 0.02);
        assert_eq!(bootstrap_config.dirichlet_epsilon, 0.3);
        assert_eq!(bootstrap_config.nn_value_weight, 0.456);
        assert_eq!(bootstrap_config.death_penalty, 5.0);
        assert_eq!(bootstrap_config.overhang_penalty_weight, 3.0);
        assert_eq!(bootstrap_config.q_scale, None);

        // Verify penalties are overridden when passed as 0
        let zeroed_config = GameGenerator::build_rollout_config(&config, true, 999, 1.0, 0.0, 0.0);
        assert_eq!(zeroed_config.death_penalty, 0.0);
        assert_eq!(zeroed_config.overhang_penalty_weight, 0.0);
    }

    #[test]
    fn test_model_artifact_paths_cover_all_split_outputs() {
        let onnx_path = unique_temp_path("candidate").with_extension("onnx");
        let artifacts = GameGenerator::model_artifact_paths(&onnx_path);
        assert_eq!(artifacts[0], onnx_path);
        assert_eq!(artifacts[1], onnx_path.with_extension("onnx.data"));
        assert_eq!(artifacts[2], onnx_path.with_extension("conv.onnx"));
        assert_eq!(artifacts[3], onnx_path.with_extension("conv.onnx.data"));
        assert_eq!(artifacts[4], onnx_path.with_extension("heads.onnx"));
        assert_eq!(artifacts[5], onnx_path.with_extension("heads.onnx.data"));
        assert_eq!(artifacts[6], onnx_path.with_extension("fc.bin"));
    }

    #[test]
    fn test_remove_model_artifacts_if_safe_preserves_bootstrap_and_incumbent() {
        let model_path = unique_temp_path("cleanup").with_extension("onnx");
        let artifacts = GameGenerator::model_artifact_paths(&model_path);
        for artifact in &artifacts {
            fs::write(artifact, b"model").expect("temp file write should succeed");
        }

        GameGenerator::remove_model_artifacts_if_safe(
            &model_path,
            &model_path,
            &PathBuf::from("different.onnx"),
            None,
        );
        for artifact in &artifacts {
            assert!(
                artifact.exists(),
                "bootstrap artifacts should not be removed"
            );
        }

        GameGenerator::remove_model_artifacts_if_safe(
            &model_path,
            &PathBuf::from("bootstrap.onnx"),
            &model_path,
            None,
        );
        for artifact in &artifacts {
            assert!(
                artifact.exists(),
                "incumbent artifacts should not be removed"
            );
        }

        GameGenerator::remove_model_artifacts_if_safe(
            &model_path,
            &PathBuf::from("bootstrap.onnx"),
            &PathBuf::from("incumbent.onnx"),
            None,
        );
        for artifact in &artifacts {
            assert!(!artifact.exists(), "candidate artifacts should be removed");
        }
    }

    #[test]
    fn test_cleanup_queued_candidate_artifacts_removes_pending_and_evaluating() {
        let bootstrap_path = unique_temp_path("bootstrap").with_extension("onnx");
        let training_data_path = unique_temp_path("training_data");
        let generator = GameGenerator::new(
            bootstrap_path.to_string_lossy().to_string(),
            training_data_path.to_string_lossy().to_string(),
            None,
            100,
            true,
            16,
            1.0,
            1,
            0,
            Some(vec![0]),
            true,
            10,
            0.0,
            1.0,
        )
        .expect("generator should construct");

        let pending_path = unique_temp_path("pending").with_extension("onnx");
        for artifact in GameGenerator::model_artifact_paths(&pending_path) {
            fs::write(&artifact, b"pending").expect("pending artifact write should succeed");
        }
        let evaluating_path = unique_temp_path("evaluating").with_extension("onnx");
        for artifact in GameGenerator::model_artifact_paths(&evaluating_path) {
            fs::write(&artifact, b"evaluating").expect("evaluating artifact write should succeed");
        }

        {
            let mut pending = generator.pending_candidate.write().unwrap();
            *pending = Some(CandidateModelRequest {
                model_path: pending_path.clone(),
                model_step: 1,
                nn_value_weight: 0.1,
            });
        }
        {
            let mut evaluating = generator.evaluating_candidate.write().unwrap();
            *evaluating = Some(CandidateModelRequest {
                model_path: evaluating_path.clone(),
                model_step: 2,
                nn_value_weight: 0.2,
            });
        }

        generator.cleanup_queued_candidate_artifacts();

        for artifact in GameGenerator::model_artifact_paths(&pending_path) {
            assert!(!artifact.exists(), "pending artifacts should be removed");
        }
        for artifact in GameGenerator::model_artifact_paths(&evaluating_path) {
            assert!(!artifact.exists(), "evaluating artifacts should be removed");
        }
        assert!(generator.pending_candidate.read().unwrap().is_none());
        assert!(generator.evaluating_candidate.read().unwrap().is_none());
    }
}
