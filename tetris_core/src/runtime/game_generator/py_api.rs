use super::*;

#[pymethods]
impl GameGenerator {
    #[new]
    #[pyo3(signature = (model_path, training_data_path, config=None, max_placements=100, add_noise=true, max_examples=100_000, save_interval_seconds=60.0, num_workers=3, initial_model_step=0, candidate_eval_seeds=None, start_with_network=true, non_network_num_simulations=4000, bootstrap_use_min_max_q_normalization=true, initial_incumbent_eval_avg_attack=0.0, nn_value_weight_cap=1.0))]
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
        bootstrap_use_min_max_q_normalization: bool,
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
        let candidate_eval_seeds = candidate_eval_seeds.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "candidate_eval_seeds must be provided explicitly",
            )
        })?;
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
            bootstrap_use_min_max_q_normalization,
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
            let bootstrap_use_min_max_q_normalization = self.bootstrap_use_min_max_q_normalization;
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
                    bootstrap_use_min_max_q_normalization,
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
