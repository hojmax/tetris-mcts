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
use std::time::Duration;

use crate::mcts::GameStats;
use crate::mcts::{GameResult, GameTreeStats, MCTSAgent, MCTSConfig, TrainingExample, NUM_ACTIONS};

use super::npz::{read_examples_from_npz, write_examples_to_npz};

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
    avg_moves: f32,
    max_moves: u32,
    tree_stats: GameTreeStats,
    cache_hits: u64,
    cache_misses: u64,
    cache_size: usize,
}

#[derive(Clone)]
struct CandidateModelRequest {
    model_path: PathBuf,
    model_step: u64,
}

struct ModelEvalEvent {
    incumbent_step: u64,
    incumbent_uses_network: bool,
    incumbent_games: u64,
    incumbent_avg_attack: f32,
    candidate_step: u64,
    candidate_games: u64,
    candidate_avg_attack: f32,
    promoted: bool,
    auto_promoted: bool,
}

/// Shared replay buffer for thread-safe access between generator and trainer.
struct SharedBuffer {
    /// Training examples (circular buffer with FIFO eviction)
    examples: RwLock<VecDeque<TrainingExample>>,
    /// Maximum buffer size
    max_size: usize,
}

impl SharedBuffer {
    fn new(max_size: usize) -> Self {
        SharedBuffer {
            examples: RwLock::new(VecDeque::with_capacity(max_size)),
            max_size,
        }
    }

    /// Add examples to the buffer, evicting oldest if over capacity.
    fn add_examples(&self, new_examples: Vec<TrainingExample>) {
        let mut examples = self.examples.write().unwrap();
        examples.extend(new_examples);
        while examples.len() > self.max_size {
            examples.pop_front();
        }
    }

    /// Get current buffer size.
    fn len(&self) -> usize {
        self.examples.read().unwrap().len()
    }

    /// Get a copy of all examples (for disk saves).
    fn get_all(&self) -> Vec<TrainingExample> {
        self.examples.read().unwrap().iter().cloned().collect()
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
    /// Maximum moves per game
    max_moves: u32,
    /// Whether to add Dirichlet noise
    add_noise: bool,
    /// Number of games between disk saves (for resume capability)
    games_per_save: usize,
    /// Number of worker threads
    num_workers: usize,
    /// Number of games the evaluator worker plays per candidate model.
    candidate_eval_games: usize,
    /// Whether to use Dirichlet noise while evaluating candidate models.
    candidate_eval_add_noise: bool,
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
    /// Lifetime game count for the currently deployed incumbent model.
    incumbent_lifetime_games: Arc<AtomicU64>,
    /// Lifetime total attack for the currently deployed incumbent model.
    incumbent_lifetime_attack: Arc<AtomicU64>,
    /// Thread handles (for joining on stop)
    thread_handles: Vec<JoinHandle<()>>,
}

#[pymethods]
impl GameGenerator {
    #[new]
    #[pyo3(signature = (model_path, training_data_path, config=None, max_moves=100, add_noise=true, max_examples=100_000, games_per_save=100, num_workers=3, initial_model_step=0, candidate_eval_games=30, candidate_eval_add_noise=false, start_with_network=true, non_network_num_simulations=3000))]
    pub fn new(
        model_path: String,
        training_data_path: String,
        config: Option<MCTSConfig>,
        max_moves: u32,
        add_noise: bool,
        max_examples: usize,
        games_per_save: usize,
        num_workers: usize,
        initial_model_step: u64,
        candidate_eval_games: usize,
        candidate_eval_add_noise: bool,
        start_with_network: bool,
        non_network_num_simulations: u32,
    ) -> PyResult<Self> {
        if max_moves == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_moves must be > 0",
            ));
        }
        if max_examples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_examples must be > 0",
            ));
        }
        if num_workers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_workers must be > 0",
            ));
        }
        if candidate_eval_games == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "candidate_eval_games must be > 0",
            ));
        }
        if non_network_num_simulations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "non_network_num_simulations must be > 0",
            ));
        }

        let mut resolved_config = config.unwrap_or_default();
        resolved_config.max_moves = max_moves;
        let bootstrap_model_path = PathBuf::from(model_path);

        Ok(GameGenerator {
            bootstrap_model_path: bootstrap_model_path.clone(),
            training_data_path: PathBuf::from(training_data_path),
            config: resolved_config,
            max_moves,
            add_noise,
            games_per_save,
            num_workers,
            candidate_eval_games,
            candidate_eval_add_noise,
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
            incumbent_lifetime_games: Arc::new(AtomicU64::new(0)),
            incumbent_lifetime_attack: Arc::new(AtomicU64::new(0)),
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
            let loaded_examples = read_examples_from_npz(&self.training_data_path, self.max_moves)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to load replay data from {}: {}",
                        self.training_data_path.display(),
                        e
                    ))
                })?;

            if !loaded_examples.is_empty() {
                let loaded_examples_count = loaded_examples.len();
                self.buffer.add_examples(loaded_examples);
                let retained_examples_count = self.buffer.len();
                self.examples_generated
                    .store(retained_examples_count as u64, Ordering::SeqCst);
                eprintln!(
                    "[GameGenerator] Loaded {} replay examples from {}",
                    retained_examples_count,
                    self.training_data_path.display()
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
            let max_moves = self.max_moves;
            let add_noise = self.add_noise;
            let games_per_save = self.games_per_save;
            let candidate_eval_games = self.candidate_eval_games;
            let candidate_eval_add_noise = self.candidate_eval_add_noise;
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
            let incumbent_lifetime_games = Arc::clone(&self.incumbent_lifetime_games);
            let incumbent_lifetime_attack = Arc::clone(&self.incumbent_lifetime_attack);

            let handle = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    num_workers,
                    is_evaluator_worker,
                    bootstrap_model_path,
                    training_data_path,
                    config,
                    max_moves,
                    add_noise,
                    games_per_save,
                    candidate_eval_games,
                    candidate_eval_add_noise,
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
                    incumbent_lifetime_games,
                    incumbent_lifetime_attack,
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

    pub fn incumbent_uses_network(&self) -> bool {
        self.incumbent_uses_network.load(Ordering::SeqCst)
    }

    pub fn incumbent_lifetime_games(&self) -> u64 {
        self.incumbent_lifetime_games.load(Ordering::SeqCst)
    }

    pub fn incumbent_lifetime_avg_attack(&self) -> f32 {
        let games = self.incumbent_lifetime_games();
        if games == 0 {
            return 0.0;
        }
        self.incumbent_lifetime_attack.load(Ordering::SeqCst) as f32 / games as f32
    }

    /// Queue a candidate model for evaluator-worker gating.
    ///
    /// If another candidate is already pending, it is dropped in favor of this one.
    /// Returns True when the candidate is queued, False when ignored as stale.
    pub fn queue_candidate_model(&self, model_path: String, model_step: u64) -> PyResult<bool> {
        let candidate_path = PathBuf::from(model_path);
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
                    evaluating_path.as_ref(),
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
            "incumbent_lifetime_games".to_string(),
            self.incumbent_lifetime_games.load(Ordering::SeqCst),
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
            d.insert("avg_valid_actions".to_string(), info.avg_moves);
            d.insert("max_valid_actions".to_string(), info.max_moves as f32);
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
            drained.push((info.game_number, d));
        }
        drained
    }

    /// Drain evaluator decision events in generation order.
    pub fn drain_model_eval_events(&self) -> Vec<HashMap<String, f64>> {
        let mut queue = self.model_eval_events.write().unwrap();
        let mut drained = Vec::with_capacity(queue.len());
        while let Some(event) = queue.pop_front() {
            let mut d = HashMap::new();
            d.insert("incumbent_step".to_string(), event.incumbent_step as f64);
            d.insert(
                "incumbent_uses_network".to_string(),
                if event.incumbent_uses_network {
                    1.0
                } else {
                    0.0
                },
            );
            d.insert("incumbent_games".to_string(), event.incumbent_games as f64);
            d.insert(
                "incumbent_avg_attack".to_string(),
                event.incumbent_avg_attack as f64,
            );
            d.insert("candidate_step".to_string(), event.candidate_step as f64);
            d.insert("candidate_games".to_string(), event.candidate_games as f64);
            d.insert(
                "candidate_avg_attack".to_string(),
                event.candidate_avg_attack as f64,
            );
            d.insert(
                "promoted".to_string(),
                if event.promoted { 1.0 } else { 0.0 },
            );
            d.insert(
                "auto_promoted".to_string(),
                if event.auto_promoted { 1.0 } else { 0.0 },
            );
            drained.push(d);
        }
        drained
    }

    /// Get the current number of examples in the replay buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Sample a batch of training data from the replay buffer.
    ///
    /// Returns a tuple of numpy arrays:
    /// (boards, aux_features, policy_targets, value_targets, overhang_fields, action_masks)
    ///
    /// Returns None if the buffer is empty.
    #[pyo3(signature = (batch_size, max_moves))]
    pub fn sample_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        max_moves: u32,
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
        if max_moves == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_moves must be > 0",
            ));
        }

        // Snapshot sampled examples under lock, then release lock before heavy work.
        let sampled_examples: Vec<TrainingExample> = {
            let examples = self.buffer.examples.read().unwrap();
            let n = examples.len();
            if n == 0 {
                return Ok(None);
            }

            let actual_batch = batch_size.min(n);
            let mut rng = thread_rng();
            (0..actual_batch)
                .map(|_| {
                    let idx = rng.gen_range(0..n);
                    examples[idx].clone()
                })
                .collect()
        };
        let actual_batch = sampled_examples.len();

        // Allocate output arrays
        let board_height = 20usize;
        let board_width = 10usize;
        let num_actions = NUM_ACTIONS;
        let aux_features_size = 52usize;

        let mut boards = vec![0.0f32; actual_batch * board_height * board_width];
        let mut aux = vec![0.0f32; actual_batch * aux_features_size];
        let mut policies = vec![0.0f32; actual_batch * num_actions];
        let mut values = vec![0.0f32; actual_batch];
        let mut overhangs = vec![0.0f32; actual_batch];
        let mut masks = vec![0.0f32; actual_batch * num_actions];
        let move_norm_denominator = max_moves as f32;

        for (i, ex) in sampled_examples.iter().enumerate() {
            // Copy board (already flat u8, convert to f32)
            for (j, &val) in ex.board.iter().enumerate() {
                boards[i * board_height * board_width + j] = val as f32;
            }

            // Build aux features (same encoding as Python)
            let aux_offset = i * aux_features_size;
            // Current piece one-hot (7)
            aux[aux_offset + ex.current_piece] = 1.0;
            // Hold piece one-hot (8) - 7 means empty
            if ex.hold_piece < 7 {
                aux[aux_offset + 7 + ex.hold_piece] = 1.0;
            } else {
                aux[aux_offset + 7 + 7] = 1.0; // Empty slot
            }
            // Hold available (1)
            aux[aux_offset + 15] = if ex.hold_available { 1.0 } else { 0.0 };
            // Next queue one-hot (5 * 7 = 35)
            for (j, &piece) in ex.next_queue.iter().take(5).enumerate() {
                aux[aux_offset + 16 + j * 7 + piece] = 1.0;
            }
            // Move number normalized (1)
            aux[aux_offset + 51] = ex.move_number as f32 / move_norm_denominator;

            // Copy policy
            for (j, &val) in ex.policy.iter().enumerate() {
                policies[i * num_actions + j] = val;
            }

            // Copy value
            values[i] = ex.value;

            // Copy overhang fields
            overhangs[i] = ex.overhang_fields as f32;

            // Copy mask
            for (j, &val) in ex.action_mask.iter().enumerate() {
                masks[i * num_actions + j] = if val { 1.0 } else { 0.0 };
            }
        }

        // Create numpy arrays
        let boards_arr = PyArray1::from_vec(py, boards)
            .reshape([actual_batch, board_height * board_width])
            .unwrap();
        let aux_arr = PyArray1::from_vec(py, aux)
            .reshape([actual_batch, aux_features_size])
            .unwrap();
        let policies_arr = PyArray1::from_vec(py, policies)
            .reshape([actual_batch, num_actions])
            .unwrap();
        let values_arr = PyArray1::from_vec(py, values);
        let overhangs_arr = PyArray1::from_vec(py, overhangs);
        let masks_arr = PyArray1::from_vec(py, masks)
            .reshape([actual_batch, num_actions])
            .unwrap();

        Ok(Some((
            boards_arr,
            aux_arr,
            policies_arr,
            values_arr,
            overhangs_arr,
            masks_arr,
        )))
    }
}

impl GameGenerator {
    /// Worker thread main loop.
    fn worker_loop(
        worker_id: usize,
        num_workers: usize,
        is_evaluator_worker: bool,
        bootstrap_model_path: PathBuf,
        training_data_path: PathBuf,
        config: MCTSConfig,
        max_moves: u32,
        add_noise: bool,
        games_per_save: usize,
        candidate_eval_games: usize,
        candidate_eval_add_noise: bool,
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
        incumbent_lifetime_games: Arc<AtomicU64>,
        incumbent_lifetime_attack: Arc<AtomicU64>,
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
        let mut local_games_count: usize = 0;

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

                    let committed_games = Self::run_candidate_evaluation(
                        worker_id,
                        candidate,
                        &config,
                        &running,
                        max_moves,
                        candidate_eval_games,
                        candidate_eval_add_noise,
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
                        &incumbent_lifetime_games,
                        &incumbent_lifetime_attack,
                    );

                    {
                        let mut evaluating = evaluating_candidate.write().unwrap();
                        *evaluating = None;
                    }

                    local_games_count += committed_games;
                    loaded_model_version = u64::MAX;
                    loaded_with_network = !incumbent_uses_network.load(Ordering::SeqCst);

                    if is_save_worker && games_per_save > 0 && local_games_count >= games_per_save {
                        Self::persist_buffer_snapshot(&training_data_path, &buffer, max_moves);
                        local_games_count = 0;
                    }
                    continue;
                }
            }

            // Play one game
            if let Some(result) = agent.play_game(max_moves, add_noise) {
                let count_toward_incumbent =
                    loaded_model_version == incumbent_model_version.load(Ordering::SeqCst);
                Self::commit_game_result(
                    result,
                    &buffer,
                    &games_generated,
                    &examples_generated,
                    &game_stats,
                    &completed_games,
                    &incumbent_lifetime_games,
                    &incumbent_lifetime_attack,
                    count_toward_incumbent,
                );
                local_games_count += 1;

                // Periodically save to disk for resume capability (only worker 0)
                if is_save_worker && games_per_save > 0 && local_games_count >= games_per_save {
                    Self::persist_buffer_snapshot(&training_data_path, &buffer, max_moves);
                    local_games_count = 0;
                }
            }
        }

        // Final save on shutdown (only worker 0)
        if is_save_worker {
            let all_examples = buffer.get_all();
            if !all_examples.is_empty() {
                let _ = write_examples_to_npz(&training_data_path, &all_examples, max_moves);
                eprintln!(
                    "[GameGenerator] Saved {} examples to disk",
                    all_examples.len()
                );
            }
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
        let Some(new_agent) = Self::create_rollout_agent(
            config,
            target_uses_network,
            non_network_num_simulations,
            model_path.as_ref(),
            worker_id,
            "incumbent",
        ) else {
            return false;
        };

        *loaded_model_step = incumbent_model_step.load(Ordering::SeqCst);
        if worker_id == 0 {
            if target_uses_network {
                eprintln!(
                    "[GameGenerator] Loaded incumbent NN model step {} ({} workers, sims={})",
                    *loaded_model_step, num_workers, config.num_simulations
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
        max_moves: u32,
        candidate_eval_games: usize,
        candidate_eval_add_noise: bool,
        non_network_num_simulations: u32,
        buffer: &Arc<SharedBuffer>,
        games_generated: &Arc<AtomicU64>,
        examples_generated: &Arc<AtomicU64>,
        game_stats: &Arc<SharedStats>,
        completed_games: &Arc<RwLock<VecDeque<LastGameInfo>>>,
        model_eval_events: &Arc<RwLock<VecDeque<ModelEvalEvent>>>,
        bootstrap_model_path: &PathBuf,
        incumbent_model_path: &Arc<RwLock<PathBuf>>,
        incumbent_uses_network: &Arc<AtomicBool>,
        incumbent_model_step: &Arc<AtomicU64>,
        incumbent_model_version: &Arc<AtomicU64>,
        incumbent_lifetime_games: &Arc<AtomicU64>,
        incumbent_lifetime_attack: &Arc<AtomicU64>,
    ) -> usize {
        let Some(candidate_agent) = Self::create_rollout_agent(
            config,
            true,
            non_network_num_simulations,
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

        let mut candidate_results: Vec<GameResult> = Vec::with_capacity(candidate_eval_games);
        for _ in 0..candidate_eval_games {
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
            if let Some(result) = candidate_agent.play_game(max_moves, candidate_eval_add_noise) {
                candidate_results.push(result);
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

        let candidate_games = candidate_results.len() as u64;
        let candidate_total_attack: u64 = candidate_results
            .iter()
            .map(|result| result.total_attack as u64)
            .sum();
        let candidate_avg_attack = candidate_total_attack as f32 / candidate_games as f32;

        let incumbent_games = incumbent_lifetime_games.load(Ordering::SeqCst);
        let incumbent_total_attack = incumbent_lifetime_attack.load(Ordering::SeqCst);
        let incumbent_avg_attack = if incumbent_games > 0 {
            incumbent_total_attack as f32 / incumbent_games as f32
        } else {
            0.0
        };
        let incumbent_uses_network_before = incumbent_uses_network.load(Ordering::SeqCst);
        let incumbent_step_before = incumbent_model_step.load(Ordering::SeqCst);
        let auto_promoted = incumbent_games == 0;
        let promoted = auto_promoted || candidate_avg_attack > incumbent_avg_attack;

        let committed_games = if promoted {
            let previous_incumbent_path = incumbent_model_path.read().unwrap().clone();
            {
                let mut incumbent_path = incumbent_model_path.write().unwrap();
                *incumbent_path = candidate.model_path.clone();
            }
            incumbent_uses_network.store(true, Ordering::SeqCst);
            incumbent_model_step.store(candidate.model_step, Ordering::SeqCst);
            incumbent_model_version.fetch_add(1, Ordering::SeqCst);
            incumbent_lifetime_games.store(0, Ordering::SeqCst);
            incumbent_lifetime_attack.store(0, Ordering::SeqCst);

            let mut committed = 0usize;
            for result in candidate_results {
                Self::commit_game_result(
                    result,
                    buffer,
                    games_generated,
                    examples_generated,
                    game_stats,
                    completed_games,
                    incumbent_lifetime_games,
                    incumbent_lifetime_attack,
                    true,
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
                "[GameGenerator] Promoted candidate step {} (avg_attack {:.3} > incumbent {:.3}, games={})",
                candidate.model_step, candidate_avg_attack, incumbent_avg_attack, candidate_games
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
                "[GameGenerator] Rejected candidate step {} (avg_attack {:.3} <= incumbent {:.3}, games={})",
                candidate.model_step, candidate_avg_attack, incumbent_avg_attack, candidate_games
            );
            0
        };

        model_eval_events
            .write()
            .unwrap()
            .push_back(ModelEvalEvent {
                incumbent_step: incumbent_step_before,
                incumbent_uses_network: incumbent_uses_network_before,
                incumbent_games,
                incumbent_avg_attack,
                candidate_step: candidate.model_step,
                candidate_games,
                candidate_avg_attack,
                promoted,
                auto_promoted,
            });

        committed_games
    }

    fn build_rollout_config(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
    ) -> MCTSConfig {
        let mut rollout_config = base_config.clone();
        rollout_config.num_simulations = if uses_network {
            base_config.num_simulations
        } else {
            non_network_num_simulations
        };
        rollout_config
    }

    fn create_rollout_agent(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        model_path: Option<&PathBuf>,
        worker_id: usize,
        role: &str,
    ) -> Option<MCTSAgent> {
        // Keep rollout behavior consistent across training and evaluator code paths,
        // including visit_sampling_epsilon and all other shared MCTS settings.
        let rollout_config =
            Self::build_rollout_config(base_config, uses_network, non_network_num_simulations);
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
        incumbent_lifetime_games: &Arc<AtomicU64>,
        incumbent_lifetime_attack: &Arc<AtomicU64>,
        count_toward_incumbent: bool,
    ) {
        let GameResult {
            examples,
            total_attack,
            num_moves,
            avg_moves,
            max_moves,
            stats,
            tree_stats,
            avg_overhang_fields,
            cache_hits,
            cache_misses,
            cache_size,
            ..
        } = result;

        let num_examples = examples.len() as u64;
        buffer.add_examples(examples);

        let game_number = games_generated.fetch_add(1, Ordering::SeqCst) + 1;
        examples_generated.fetch_add(num_examples, Ordering::SeqCst);
        game_stats.add(&stats, total_attack);

        completed_games.write().unwrap().push_back(LastGameInfo {
            game_number,
            stats,
            total_attack,
            avg_overhang_fields,
            num_moves,
            avg_moves,
            max_moves,
            tree_stats,
            cache_hits,
            cache_misses,
            cache_size,
        });

        if count_toward_incumbent {
            incumbent_lifetime_games.fetch_add(1, Ordering::SeqCst);
            incumbent_lifetime_attack.fetch_add(total_attack as u64, Ordering::SeqCst);
        }
    }

    fn persist_buffer_snapshot(
        training_data_path: &PathBuf,
        buffer: &Arc<SharedBuffer>,
        max_moves: u32,
    ) {
        let all_examples = buffer.get_all();
        if let Err(error) = write_examples_to_npz(training_data_path, &all_examples, max_moves) {
            eprintln!("[GameGenerator] Failed to write NPZ: {}", error);
        }
    }

    fn model_artifact_paths(model_path: &Path) -> [PathBuf; 4] {
        let base_path = model_path.with_extension("");
        [
            model_path.to_path_buf(),
            base_path.with_extension("conv.onnx"),
            base_path.with_extension("heads.onnx"),
            base_path.with_extension("fc.bin"),
        ]
    }

    fn remove_model_artifacts(model_path: &PathBuf) {
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
        model_path: &PathBuf,
        bootstrap_model_path: &PathBuf,
        incumbent_model_path: &PathBuf,
        evaluating_model_path: Option<&PathBuf>,
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
}

impl Drop for GameGenerator {
    fn drop(&mut self) {
        // Ensure all threads are stopped when generator is dropped
        self.running.store(false, Ordering::SeqCst);
        for handle in self.thread_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
    use crate::mcts::NUM_ACTIONS;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
            policy,
            value: move_number as f32,
            action_mask,
            overhang_fields: move_number,
        }
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tetris_generator_{}_{}_{}.npz",
            name,
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn test_shared_buffer_fifo_eviction() {
        let buffer = SharedBuffer::new(3);
        buffer.add_examples(vec![make_example(1), make_example(2)]);
        buffer.add_examples(vec![make_example(3), make_example(4)]);

        let kept = buffer.get_all();
        assert_eq!(kept.len(), 3);
        let move_numbers: Vec<u32> = kept.iter().map(|e| e.move_number).collect();
        assert_eq!(move_numbers, vec![2, 3, 4]);
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

        let network_config = GameGenerator::build_rollout_config(&config, true, 999);
        assert_eq!(network_config.num_simulations, 123);
        assert_eq!(network_config.visit_sampling_epsilon, 0.42);
        assert_eq!(network_config.temperature, 1.5);
        assert_eq!(network_config.dirichlet_alpha, 0.02);
        assert_eq!(network_config.dirichlet_epsilon, 0.3);

        let bootstrap_config = GameGenerator::build_rollout_config(&config, false, 999);
        assert_eq!(bootstrap_config.num_simulations, 999);
        assert_eq!(bootstrap_config.visit_sampling_epsilon, 0.42);
        assert_eq!(bootstrap_config.temperature, 1.5);
        assert_eq!(bootstrap_config.dirichlet_alpha, 0.02);
        assert_eq!(bootstrap_config.dirichlet_epsilon, 0.3);
    }

    #[test]
    fn test_model_artifact_paths_cover_all_split_outputs() {
        let onnx_path = unique_temp_path("candidate").with_extension("onnx");
        let artifacts = GameGenerator::model_artifact_paths(&onnx_path);
        assert_eq!(artifacts[0], onnx_path);
        assert_eq!(artifacts[1], onnx_path.with_extension("conv.onnx"));
        assert_eq!(artifacts[2], onnx_path.with_extension("heads.onnx"));
        assert_eq!(artifacts[3], onnx_path.with_extension("fc.bin"));
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
}
