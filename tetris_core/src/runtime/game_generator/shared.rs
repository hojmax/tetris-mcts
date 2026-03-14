use super::*;

/// Info about the last completed game (for per-game logging).
pub(super) struct LastGameInfo {
    pub(super) game_number: u64,
    pub(super) stats: GameStats,
    pub(super) total_attack: u32,
    pub(super) avg_overhang_fields: f32,
    pub(super) num_moves: u32,
    pub(super) avg_valid_actions: f32,
    pub(super) max_valid_actions: u32,
    pub(super) tree_stats: GameTreeStats,
    pub(super) cache_hits: u64,
    pub(super) cache_misses: u64,
    pub(super) cache_size: usize,
    pub(super) tree_reuse_hits: u32,
    pub(super) tree_reuse_misses: u32,
    pub(super) tree_reuse_carry_fraction: f32,
    pub(super) traversal_total: u64,
    pub(super) traversal_expansions: u64,
    pub(super) traversal_terminal_ends: u64,
    pub(super) traversal_horizon_ends: u64,
    pub(super) traversal_expansion_fraction: f32,
    pub(super) traversal_terminal_fraction: f32,
    pub(super) traversal_horizon_fraction: f32,
}

impl LastGameInfo {
    fn ratio_or_zero(numerator: u64, denominator: u64) -> f32 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f32 / denominator as f32
        }
    }

    pub(super) fn to_dict(&self) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        metrics.insert("singles".to_string(), self.stats.singles as f32);
        metrics.insert("doubles".to_string(), self.stats.doubles as f32);
        metrics.insert("triples".to_string(), self.stats.triples as f32);
        metrics.insert("tetrises".to_string(), self.stats.tetrises as f32);
        metrics.insert("tspin_minis".to_string(), self.stats.tspin_minis as f32);
        metrics.insert("tspin_singles".to_string(), self.stats.tspin_singles as f32);
        metrics.insert("tspin_doubles".to_string(), self.stats.tspin_doubles as f32);
        metrics.insert("tspin_triples".to_string(), self.stats.tspin_triples as f32);
        metrics.insert(
            "perfect_clears".to_string(),
            self.stats.perfect_clears as f32,
        );
        metrics.insert("back_to_backs".to_string(), self.stats.back_to_backs as f32);
        metrics.insert("max_combo".to_string(), self.stats.max_combo as f32);
        metrics.insert("total_lines".to_string(), self.stats.total_lines as f32);
        metrics.insert("holds".to_string(), self.stats.holds as f32);
        metrics.insert("total_attack".to_string(), self.total_attack as f32);
        metrics.insert("avg_overhang".to_string(), self.avg_overhang_fields);
        metrics.insert("episode_length".to_string(), self.num_moves as f32);
        metrics.insert("avg_valid_actions".to_string(), self.avg_valid_actions);
        metrics.insert(
            "max_valid_actions".to_string(),
            self.max_valid_actions as f32,
        );
        metrics.insert(
            "tree_avg_branching_factor".to_string(),
            self.tree_stats.avg_branching_factor,
        );
        metrics.insert("tree_avg_leaves".to_string(), self.tree_stats.avg_leaves);
        metrics.insert(
            "tree_avg_total_nodes".to_string(),
            self.tree_stats.avg_total_nodes,
        );
        metrics.insert(
            "tree_avg_max_depth".to_string(),
            self.tree_stats.avg_max_depth,
        );
        metrics.insert(
            "tree_max_attack".to_string(),
            self.tree_stats.max_tree_attack as f32,
        );

        let cache_lookups = self.cache_hits + self.cache_misses;
        metrics.insert(
            "cache_hit_rate".to_string(),
            Self::ratio_or_zero(self.cache_hits, cache_lookups),
        );
        metrics.insert("cache_hits".to_string(), self.cache_hits as f32);
        metrics.insert("cache_misses".to_string(), self.cache_misses as f32);
        metrics.insert("cache_size".to_string(), self.cache_size as f32);

        let tree_reuse_total = (self.tree_reuse_hits + self.tree_reuse_misses) as u64;
        metrics.insert(
            "tree_reuse_rate".to_string(),
            Self::ratio_or_zero(self.tree_reuse_hits as u64, tree_reuse_total),
        );
        metrics.insert("tree_reuse_hits".to_string(), self.tree_reuse_hits as f32);
        metrics.insert(
            "tree_reuse_misses".to_string(),
            self.tree_reuse_misses as f32,
        );
        metrics.insert(
            "tree_reuse_carry_fraction".to_string(),
            self.tree_reuse_carry_fraction,
        );

        metrics.insert("traversal_total".to_string(), self.traversal_total as f32);
        metrics.insert(
            "traversal_expansions".to_string(),
            self.traversal_expansions as f32,
        );
        metrics.insert(
            "traversal_terminal_ends".to_string(),
            self.traversal_terminal_ends as f32,
        );
        metrics.insert(
            "traversal_horizon_ends".to_string(),
            self.traversal_horizon_ends as f32,
        );
        metrics.insert(
            "traversal_expansion_fraction".to_string(),
            self.traversal_expansion_fraction,
        );
        metrics.insert(
            "traversal_terminal_fraction".to_string(),
            self.traversal_terminal_fraction,
        );
        metrics.insert(
            "traversal_horizon_fraction".to_string(),
            self.traversal_horizon_fraction,
        );
        metrics
    }
}

#[derive(Clone)]
pub(super) struct CandidateModelRequest {
    pub(super) model_path: PathBuf,
    pub(super) model_step: u64,
    pub(super) nn_value_weight: f32,
}

pub(super) struct ModelEvalEvent {
    pub(super) incumbent_step: u64,
    pub(super) incumbent_uses_network: bool,
    pub(super) incumbent_avg_attack: f32,
    pub(super) incumbent_nn_value_weight: f32,
    pub(super) candidate_step: u64,
    pub(super) candidate_games: u64,
    pub(super) candidate_avg_attack: f32,
    pub(super) candidate_attack_variance: f32,
    pub(super) candidate_nn_value_weight: f32,
    pub(super) promoted_nn_value_weight: f32,
    pub(super) promoted_death_penalty: f32,
    pub(super) promoted_overhang_penalty_weight: f32,
    pub(super) promoted: bool,
    pub(super) auto_promoted: bool,
    pub(super) evaluation_seconds: f32,
    /// Best (max attack) game replay, if available.
    pub(super) best_game_replay: Option<GameReplay>,
    /// Worst (min attack) game replay, if available.
    pub(super) worst_game_replay: Option<GameReplay>,
    /// Per-game results: (seed, attack, lines, moves)
    pub(super) per_game_results: Vec<(u64, u32, u32, u32)>,
}

#[derive(Clone)]
pub(super) struct IncumbentState {
    pub(super) model_path: Arc<RwLock<PathBuf>>,
    pub(super) uses_network: Arc<AtomicBool>,
    pub(super) model_step: Arc<AtomicU64>,
    pub(super) model_version: Arc<AtomicU64>,
    pub(super) nn_value_weight: Arc<AtomicU32>,
    pub(super) death_penalty: Arc<AtomicU32>,
    pub(super) overhang_penalty_weight: Arc<AtomicU32>,
    pub(super) eval_avg_attack: Arc<AtomicU32>,
}

#[derive(Clone)]
pub(super) struct WorkerSharedState {
    pub(super) buffer: Arc<SharedBuffer>,
    pub(super) running: Arc<AtomicBool>,
    pub(super) games_generated: Arc<AtomicU64>,
    pub(super) examples_generated: Arc<AtomicU64>,
    pub(super) completed_games: Arc<RwLock<VecDeque<LastGameInfo>>>,
    pub(super) pending_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
    pub(super) evaluating_candidate: Arc<RwLock<Option<CandidateModelRequest>>>,
    pub(super) model_eval_events: Arc<RwLock<VecDeque<ModelEvalEvent>>>,
    pub(super) incumbent: IncumbentState,
}

#[derive(Clone)]
pub(super) struct WorkerSettings {
    pub(super) bootstrap_model_path: PathBuf,
    pub(super) training_data_path: PathBuf,
    pub(super) config: MCTSConfig,
    pub(super) max_placements: u32,
    pub(super) add_noise: bool,
    pub(super) save_interval_seconds: f64,
    pub(super) num_workers: usize,
    pub(super) candidate_eval_seeds: Arc<[u64]>,
    pub(super) non_network_num_simulations: u32,
    pub(super) bootstrap_use_min_max_q_normalization: bool,
    pub(super) nn_value_weight_cap: f32,
}

#[derive(Clone)]
pub(super) struct WorkerContext {
    pub(super) worker_id: usize,
    pub(super) is_evaluator_worker: bool,
    pub(super) settings: WorkerSettings,
    pub(super) shared: WorkerSharedState,
}

/// Shared replay buffer for thread-safe access between generator and trainer.
pub(super) struct SharedBuffer {
    /// Training examples and logical index space, updated atomically under one lock.
    pub(super) state: RwLock<SharedBufferState>,
    /// Maximum buffer size
    pub(super) max_size: usize,
}

pub(super) struct SharedBufferState {
    pub(super) examples: VecDeque<TrainingExample>,
    pub(super) end_index: u64,
}

impl SharedBuffer {
    pub(super) fn new(max_size: usize) -> Self {
        SharedBuffer {
            state: RwLock::new(SharedBufferState {
                examples: VecDeque::with_capacity(max_size),
                end_index: 0,
            }),
            max_size,
        }
    }

    /// Add examples to the buffer, evicting oldest if over capacity.
    pub(super) fn add_examples(&self, new_examples: Vec<TrainingExample>) {
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
    pub(super) fn len(&self) -> usize {
        self.state.read().unwrap().examples.len()
    }

    /// Stream buffer contents directly to an NPZ file under read lock.
    ///
    /// Holds the read lock for the duration of the write so that the snapshot is
    /// consistent. Workers calling `add_examples` will block until the save completes
    /// (typically 10-30 seconds), which is acceptable for a save every 30 minutes.
    pub(super) fn persist_to_npz(&self, filepath: &Path) -> Result<(), String> {
        let state = self.state.read().unwrap();
        let (slice_a, slice_b) = state.examples.as_slices();
        let tmp_extension = filepath
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}.tmp"))
            .unwrap_or_else(|| "tmp".to_string());
        let tmp_path = filepath.with_extension(tmp_extension);
        if let Err(error) = fs::remove_file(&tmp_path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                return Err(format!(
                    "Failed to remove stale temp snapshot {}: {}",
                    tmp_path.display(),
                    error
                ));
            }
        }

        if let Err(error) = write_examples_slices_to_npz(&tmp_path, slice_a, slice_b) {
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }

        if let Err(rename_error) = fs::rename(&tmp_path, filepath) {
            if filepath.exists() {
                if let Err(remove_error) = fs::remove_file(filepath) {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(format!(
                        "Failed to replace snapshot {} (rename error: {}; remove old file error: {})",
                        filepath.display(),
                        rename_error,
                        remove_error
                    ));
                }
                if let Err(second_rename_error) = fs::rename(&tmp_path, filepath) {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(format!(
                        "Failed to move temp snapshot {} into place: {}",
                        tmp_path.display(),
                        second_rename_error
                    ));
                }
            } else {
                let _ = fs::remove_file(&tmp_path);
                return Err(format!(
                    "Failed to move temp snapshot {} into place: {}",
                    tmp_path.display(),
                    rename_error
                ));
            }
        }

        Ok(())
    }

    /// Return the logical one-past-end index in replay index space.
    pub(super) fn window_end_index(&self) -> u64 {
        self.state.read().unwrap().end_index
    }

    /// Snapshot the current replay FIFO window atomically.
    pub(super) fn logical_window_snapshot(&self) -> Option<(u64, u64, Vec<TrainingExample>)> {
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
    pub(super) fn logical_delta_slice(
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
