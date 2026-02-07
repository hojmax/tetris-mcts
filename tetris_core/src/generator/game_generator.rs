//! Background Game Generator
//!
//! Spawns a worker thread that continuously generates self-play games
//! using MCTS. Training data is kept in a shared in-memory buffer that
//! Python can sample from directly, avoiding disk I/O during training.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, UNIX_EPOCH};

use crate::mcts::GameStats;
use crate::mcts::{MCTSAgent, MCTSConfig, TrainingExample};

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
        d
    }
}

/// Info about the last completed game (for per-game logging).
struct LastGameInfo {
    game_number: u64,
    stats: GameStats,
    total_attack: u32,
    num_moves: u32,
}

/// Shared replay buffer for thread-safe access between generator and trainer.
struct SharedBuffer {
    /// Training examples (circular buffer with FIFO eviction)
    examples: RwLock<Vec<TrainingExample>>,
    /// Maximum buffer size
    max_size: usize,
}

impl SharedBuffer {
    fn new(max_size: usize) -> Self {
        SharedBuffer {
            examples: RwLock::new(Vec::with_capacity(max_size)),
            max_size,
        }
    }

    /// Add examples to the buffer, evicting oldest if over capacity.
    fn add_examples(&self, new_examples: Vec<TrainingExample>) {
        let mut examples = self.examples.write().unwrap();
        examples.extend(new_examples);

        // FIFO eviction
        if examples.len() > self.max_size {
            let excess = examples.len() - self.max_size;
            examples.drain(..excess);
        }
    }

    /// Get current buffer size.
    fn len(&self) -> usize {
        self.examples.read().unwrap().len()
    }

    /// Get a copy of all examples (for disk saves).
    fn get_all(&self) -> Vec<TrainingExample> {
        self.examples.read().unwrap().clone()
    }
}

#[pyclass]
pub struct GameGenerator {
    /// Path to ONNX model file (watched for updates)
    model_path: PathBuf,
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
    /// Last completed game's stats (for per-game logging)
    last_game: Arc<RwLock<Option<LastGameInfo>>>,
    /// Thread handles (for joining on stop)
    thread_handles: Vec<JoinHandle<()>>,
}

#[pymethods]
impl GameGenerator {
    #[new]
    #[pyo3(signature = (model_path, training_data_path, config=None, max_moves=100, add_noise=true, max_examples=100_000, games_per_save=100, num_workers=3))]
    pub fn new(
        model_path: String,
        training_data_path: String,
        config: Option<MCTSConfig>,
        max_moves: u32,
        add_noise: bool,
        max_examples: usize,
        games_per_save: usize,
        num_workers: usize,
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

        let mut resolved_config = config.unwrap_or_default();
        resolved_config.max_moves = max_moves;

        Ok(GameGenerator {
            model_path: PathBuf::from(model_path),
            training_data_path: PathBuf::from(training_data_path),
            config: resolved_config,
            max_moves,
            add_noise,
            games_per_save,
            num_workers,
            buffer: Arc::new(SharedBuffer::new(max_examples)),
            running: Arc::new(AtomicBool::new(false)),
            games_generated: Arc::new(AtomicU64::new(0)),
            examples_generated: Arc::new(AtomicU64::new(0)),
            game_stats: Arc::new(SharedStats::new()),
            last_game: Arc::new(RwLock::new(None)),
            thread_handles: Vec::new(),
        })
    }

    /// Start background game generation.
    ///
    /// Spawns worker threads that continuously generate games and write
    /// them to training_data_path. The threads watch the model file for
    /// changes and hot-swap when updated.
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

        // Spawn worker threads
        for worker_id in 0..self.num_workers {
            // Clone Arc handles for each thread
            let model_path = self.model_path.clone();
            let training_data_path = self.training_data_path.clone();
            let config = self.config.clone();
            let max_moves = self.max_moves;
            let add_noise = self.add_noise;
            let games_per_save = self.games_per_save;
            let num_workers = self.num_workers;
            let buffer = Arc::clone(&self.buffer);
            let running = Arc::clone(&self.running);
            let games_generated = Arc::clone(&self.games_generated);
            let examples_generated = Arc::clone(&self.examples_generated);
            let game_stats = Arc::clone(&self.game_stats);
            let last_game = Arc::clone(&self.last_game);

            let handle = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    num_workers,
                    model_path,
                    training_data_path,
                    config,
                    max_moves,
                    add_noise,
                    games_per_save,
                    buffer,
                    running,
                    games_generated,
                    examples_generated,
                    game_stats,
                    last_game,
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

    /// Get statistics as a dictionary.
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("games_generated".to_string(), self.games_generated());
        stats.insert("examples_generated".to_string(), self.examples_generated());
        stats.insert("is_running".to_string(), self.is_running() as u64);
        stats.insert("buffer_size".to_string(), self.buffer_size() as u64);
        stats
    }

    /// Get aggregate game statistics (line clears, T-spins, etc.)
    pub fn get_game_stats(&self) -> HashMap<String, u32> {
        self.game_stats.to_dict()
    }

    /// Get the last completed game's stats for per-game logging.
    /// Returns (game_number, stats_dict) or None if no games completed yet.
    pub fn get_last_game_stats(&self) -> Option<(u64, HashMap<String, u32>)> {
        let guard = self.last_game.read().unwrap();
        guard.as_ref().map(|info| {
            let mut d = HashMap::new();
            d.insert("singles".to_string(), info.stats.singles);
            d.insert("doubles".to_string(), info.stats.doubles);
            d.insert("triples".to_string(), info.stats.triples);
            d.insert("tetrises".to_string(), info.stats.tetrises);
            d.insert("tspin_minis".to_string(), info.stats.tspin_minis);
            d.insert("tspin_singles".to_string(), info.stats.tspin_singles);
            d.insert("tspin_doubles".to_string(), info.stats.tspin_doubles);
            d.insert("tspin_triples".to_string(), info.stats.tspin_triples);
            d.insert("perfect_clears".to_string(), info.stats.perfect_clears);
            d.insert("back_to_backs".to_string(), info.stats.back_to_backs);
            d.insert("max_combo".to_string(), info.stats.max_combo);
            d.insert("total_lines".to_string(), info.stats.total_lines);
            d.insert("total_attack".to_string(), info.total_attack);
            d.insert("episode_length".to_string(), info.num_moves);
            (info.game_number, d)
        })
    }

    /// Get the current number of examples in the replay buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Sample a batch of training data from the replay buffer.
    ///
    /// Returns a tuple of numpy arrays:
    /// (boards, aux_features, policy_targets, value_targets, action_masks)
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
            &'py PyArray2<f32>,
        )>,
    > {
        if max_moves == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_moves must be > 0",
            ));
        }

        let examples = self.buffer.examples.read().unwrap();
        let n = examples.len();
        if n == 0 {
            return Ok(None);
        }

        let actual_batch = batch_size.min(n);
        let mut rng = thread_rng();

        // Sample random indices
        let indices: Vec<usize> = (0..actual_batch).map(|_| rng.gen_range(0..n)).collect();

        // Allocate output arrays
        let board_height = 20usize;
        let board_width = 10usize;
        let num_actions = 734usize;
        let aux_features_size = 52usize;

        let mut boards = vec![0.0f32; actual_batch * board_height * board_width];
        let mut aux = vec![0.0f32; actual_batch * aux_features_size];
        let mut policies = vec![0.0f32; actual_batch * num_actions];
        let mut values = vec![0.0f32; actual_batch];
        let mut masks = vec![0.0f32; actual_batch * num_actions];
        let move_norm_denominator = max_moves as f32;

        for (i, &idx) in indices.iter().enumerate() {
            let ex = &examples[idx];

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
        let masks_arr = PyArray1::from_vec(py, masks)
            .reshape([actual_batch, num_actions])
            .unwrap();

        Ok(Some((
            boards_arr,
            aux_arr,
            policies_arr,
            values_arr,
            masks_arr,
        )))
    }
}

impl GameGenerator {
    /// Worker thread main loop.
    fn worker_loop(
        worker_id: usize,
        num_workers: usize,
        model_path: PathBuf,
        training_data_path: PathBuf,
        config: MCTSConfig,
        max_moves: u32,
        add_noise: bool,
        games_per_save: usize,
        buffer: Arc<SharedBuffer>,
        running: Arc<AtomicBool>,
        games_generated: Arc<AtomicU64>,
        examples_generated: Arc<AtomicU64>,
        game_stats: Arc<SharedStats>,
        last_game: Arc<RwLock<Option<LastGameInfo>>>,
    ) {
        let mut agent = MCTSAgent::new(config);
        let mut current_mtime: u64 = 0;

        // Wait for initial model
        while running.load(Ordering::SeqCst) {
            if let Some(mtime) = Self::get_model_mtime(&model_path) {
                let path_str = model_path
                    .to_str()
                    .expect("Model path contains invalid UTF-8");
                if agent.load_model(path_str) {
                    current_mtime = mtime;
                    if worker_id == 0 {
                        eprintln!(
                            "[GameGenerator] Loaded initial model ({} workers)",
                            num_workers
                        );
                    }
                    break;
                }
            }
            thread::sleep(Duration::from_millis(500));
        }

        // Only worker 0 handles disk saves to avoid race conditions
        let is_save_worker = worker_id == 0;
        let mut local_games_count: usize = 0;

        // Main generation loop
        while running.load(Ordering::SeqCst) {
            // Check for model updates
            if let Some(mtime) = Self::get_model_mtime(&model_path) {
                if mtime > current_mtime {
                    let path_str = model_path
                        .to_str()
                        .expect("Model path contains invalid UTF-8");
                    if agent.load_model(path_str) {
                        current_mtime = mtime;
                        if worker_id == 0 {
                            eprintln!("[GameGenerator] Reloaded updated model");
                        }
                    }
                }
            }

            // Play one game
            if let Some(result) = agent.play_game(max_moves, add_noise) {
                let num_examples = result.examples.len() as u64;

                // Add to shared buffer (thread-safe, handles FIFO eviction)
                buffer.add_examples(result.examples);

                local_games_count += 1;

                // Update counters
                games_generated.fetch_add(1, Ordering::SeqCst);
                examples_generated.fetch_add(num_examples, Ordering::SeqCst);

                // Accumulate game stats
                game_stats.add(&result.stats, result.total_attack);

                // Store last game info for per-game logging
                let game_number = games_generated.load(Ordering::SeqCst);
                *last_game.write().unwrap() = Some(LastGameInfo {
                    game_number,
                    stats: result.stats,
                    total_attack: result.total_attack,
                    num_moves: result.num_moves,
                });

                // Periodically save to disk for resume capability (only worker 0)
                if is_save_worker && games_per_save > 0 && local_games_count >= games_per_save {
                    let all_examples = buffer.get_all();
                    if let Err(e) =
                        write_examples_to_npz(&training_data_path, &all_examples, max_moves)
                    {
                        eprintln!("[GameGenerator] Failed to write NPZ: {}", e);
                    }
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

    /// Get the modification time of a file as seconds since UNIX epoch.
    fn get_model_mtime(path: &PathBuf) -> Option<u64> {
        match fs::metadata(path) {
            Ok(meta) => match meta.modified() {
                Ok(time) => Some(time.duration_since(UNIX_EPOCH).unwrap().as_secs()),
                Err(e) => {
                    eprintln!(
                        "[GameGenerator] Failed to get modification time for {:?}: {}",
                        path, e
                    );
                    None
                }
            },
            Err(e) => {
                // Only log if it's not a "file not found" error (which is expected during startup)
                if e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!(
                        "[GameGenerator] Failed to access model file {:?}: {}",
                        path, e
                    );
                }
                None
            }
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
    }

    #[test]
    fn test_get_model_mtime_for_missing_and_existing_file() {
        let missing = unique_temp_path("missing");
        assert_eq!(GameGenerator::get_model_mtime(&missing), None);

        let existing = unique_temp_path("existing");
        fs::write(&existing, b"model").expect("temp file write should succeed");
        let mtime = GameGenerator::get_model_mtime(&existing);
        fs::remove_file(&existing).expect("temp file cleanup should succeed");
        assert!(mtime.is_some());
    }
}
