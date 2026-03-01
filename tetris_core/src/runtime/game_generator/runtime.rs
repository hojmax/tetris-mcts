use super::*;

const CANDIDATE_EVAL_MCTS_SEED: u64 = 0;
const GAME_COMMIT_BATCH_SIZE: usize = 4;

impl GameGenerator {
    pub(super) fn load_atomic_f32(value: &AtomicU32) -> f32 {
        f32::from_bits(value.load(Ordering::SeqCst))
    }

    pub(super) fn store_atomic_f32(target: &AtomicU32, value: f32) {
        target.store(value.to_bits(), Ordering::SeqCst);
    }

    pub(super) fn examples_to_numpy<'py>(
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
            crate::inference::encode_aux_features(
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

    pub(super) fn persist_snapshot_if_due(
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
    pub(super) fn worker_loop(
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
        bootstrap_use_min_max_q_normalization: bool,
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
        let mut pending_results: Vec<GameResult> = Vec::with_capacity(GAME_COMMIT_BATCH_SIZE);

        while running.load(Ordering::SeqCst)
            && !Self::sync_incumbent_agent_if_needed(
                &config,
                non_network_num_simulations,
                bootstrap_use_min_max_q_normalization,
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
                bootstrap_use_min_max_q_normalization,
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
                    if !pending_results.is_empty() {
                        let to_commit = std::mem::take(&mut pending_results);
                        Self::commit_game_results_batch(
                            to_commit,
                            &buffer,
                            &games_generated,
                            &examples_generated,
                            &game_stats,
                            &completed_games,
                        );
                    }

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
                        non_network_num_simulations,
                        bootstrap_use_min_max_q_normalization,
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
                pending_results.push(result);
                if pending_results.len() >= GAME_COMMIT_BATCH_SIZE {
                    let to_commit = std::mem::take(&mut pending_results);
                    Self::commit_game_results_batch(
                        to_commit,
                        &buffer,
                        &games_generated,
                        &examples_generated,
                        &game_stats,
                        &completed_games,
                    );
                }

                // Periodically save to disk for resume capability based on
                // wall-clock time.
                if is_save_worker {
                    if !pending_results.is_empty() {
                        let to_commit = std::mem::take(&mut pending_results);
                        Self::commit_game_results_batch(
                            to_commit,
                            &buffer,
                            &games_generated,
                            &examples_generated,
                            &game_stats,
                            &completed_games,
                        );
                    }
                    Self::persist_snapshot_if_due(
                        &training_data_path,
                        &buffer,
                        save_interval_seconds,
                        &mut next_snapshot_deadline,
                    );
                }
            }
        }

        if !pending_results.is_empty() {
            let to_commit = std::mem::take(&mut pending_results);
            Self::commit_game_results_batch(
                to_commit,
                &buffer,
                &games_generated,
                &examples_generated,
                &game_stats,
                &completed_games,
            );
        }

        // Final save on shutdown (only worker 0)
        if is_save_worker && buffer.len() > 0 {
            let n = buffer.len();
            let _ = buffer.persist_to_npz(&training_data_path);
            eprintln!("[GameGenerator] Saved {} examples to disk", n);
        }

        eprintln!("[GameGenerator] Worker {} exiting", worker_id);
    }

    pub(super) fn sync_incumbent_agent_if_needed(
        config: &MCTSConfig,
        non_network_num_simulations: u32,
        bootstrap_use_min_max_q_normalization: bool,
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
            bootstrap_use_min_max_q_normalization,
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

    pub(super) fn run_candidate_evaluation(
        worker_id: usize,
        candidate: CandidateModelRequest,
        config: &MCTSConfig,
        running: &Arc<AtomicBool>,
        max_placements: u32,
        candidate_eval_seeds: &[u64],
        non_network_num_simulations: u32,
        bootstrap_use_min_max_q_normalization: bool,
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
        let eval_config = Self::build_candidate_eval_config(config);
        let incumbent_uses_network_before = incumbent_uses_network.load(Ordering::SeqCst);
        let incumbent_step_before = incumbent_model_step.load(Ordering::SeqCst);
        let incumbent_nn_value_weight_before = Self::load_atomic_f32(incumbent_nn_value_weight);
        let previous_incumbent_avg_attack = Self::load_atomic_f32(incumbent_eval_avg_attack);
        let incumbent_path_before = incumbent_model_path.read().unwrap().clone();

        // Read current penalty values for candidate evaluation
        let candidate_death_penalty = Self::load_atomic_f32(incumbent_death_penalty);
        let candidate_overhang_penalty_weight =
            Self::load_atomic_f32(incumbent_overhang_penalty_weight);
        let Some(candidate_agent) = Self::create_rollout_agent(
            &eval_config,
            true,
            non_network_num_simulations,
            bootstrap_use_min_max_q_normalization,
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
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                bootstrap_model_path,
                &incumbent_path_before,
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
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return 0;
            }

            // Deterministic fixed-seed evaluation:
            // same seed, no root noise, deterministic MCTS seed in eval_config.
            let candidate_outcome = candidate_agent.play_game_on_env(
                TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed),
                max_placements,
                false,
            );
            if let Some((candidate_result, replay)) = candidate_outcome {
                candidate_results.push(CandidateGameResult {
                    seed,
                    game_result: candidate_result,
                    replay_moves: replay,
                });
            } else {
                // Keep incumbent baseline and candidate sample size comparable:
                // a failed rollout invalidates this gate decision.
                eprintln!(
                    "[GameGenerator] Candidate eval failed on seed {}; rejecting candidate step {}",
                    seed, candidate.model_step
                );
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return 0;
            }
        }

        if candidate_results.is_empty() {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                bootstrap_model_path,
                &incumbent_path_before,
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
        let incumbent_avg_attack = previous_incumbent_avg_attack;

        let auto_promoted = previous_incumbent_avg_attack == 0.0 && !incumbent_uses_network_before;
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

            let committed = Self::commit_game_results_batch(
                candidate_results
                    .into_iter()
                    .map(|r| r.game_result)
                    .collect(),
                buffer,
                games_generated,
                examples_generated,
                game_stats,
                completed_games,
            );

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
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                bootstrap_model_path,
                &incumbent_path_before,
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

    pub(super) fn build_candidate_eval_config(base_config: &MCTSConfig) -> MCTSConfig {
        let mut eval_config = base_config.clone();
        eval_config.seed = Some(CANDIDATE_EVAL_MCTS_SEED);
        eval_config.visit_sampling_epsilon = 0.0;
        eval_config
    }

    pub(super) fn build_rollout_config(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        bootstrap_use_min_max_q_normalization: bool,
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
        if !uses_network && bootstrap_use_min_max_q_normalization {
            rollout_config.q_scale = None;
        }
        rollout_config
    }

    pub(super) fn create_rollout_agent(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        bootstrap_use_min_max_q_normalization: bool,
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
            bootstrap_use_min_max_q_normalization,
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

    pub(super) fn commit_game_results_batch(
        results: Vec<GameResult>,
        buffer: &Arc<SharedBuffer>,
        games_generated: &Arc<AtomicU64>,
        examples_generated: &Arc<AtomicU64>,
        game_stats: &Arc<SharedStats>,
        completed_games: &Arc<RwLock<VecDeque<LastGameInfo>>>,
    ) -> usize {
        if results.is_empty() {
            return 0;
        }

        let mut all_examples: Vec<TrainingExample> = Vec::new();
        let mut total_examples = 0u64;
        let mut completed_infos: Vec<LastGameInfo> = Vec::with_capacity(results.len());

        for result in results {
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

            total_examples += examples.len() as u64;
            all_examples.extend(examples);

            game_stats.add(&stats, total_attack);
            completed_infos.push(LastGameInfo {
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

        if !all_examples.is_empty() {
            buffer.add_examples(all_examples);
        }
        if total_examples > 0 {
            examples_generated.fetch_add(total_examples, Ordering::SeqCst);
        }
        let committed_games = completed_infos.len();
        if committed_games > 0 {
            completed_games.write().unwrap().extend(completed_infos);
        }

        committed_games
    }

    pub(super) fn persist_buffer_snapshot(training_data_path: &Path, buffer: &Arc<SharedBuffer>) {
        if let Err(error) = buffer.persist_to_npz(training_data_path) {
            eprintln!("[GameGenerator] Failed to write NPZ: {}", error);
        }
    }

    pub(super) fn model_artifact_paths(model_path: &Path) -> [PathBuf; 7] {
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

    pub(super) fn remove_model_artifacts(model_path: &Path) {
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

    pub(super) fn remove_model_artifacts_if_safe(
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

    pub(super) fn cleanup_queued_candidate_artifacts(&self) {
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
