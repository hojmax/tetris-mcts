use super::*;

/// Aggregate game statistics (thread-safe counters).
#[derive(Default)]
pub(super) struct SharedStats {
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
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Add stats from a completed game.
    pub(super) fn add(&self, stats: &GameStats, attack: u32) {
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
    pub(super) fn to_dict(&self) -> HashMap<String, u32> {
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
