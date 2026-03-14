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

    pub(super) fn effective_search_penalties(
        nn_value_weight: f32,
        nn_value_weight_cap: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
    ) -> (f32, f32) {
        if nn_value_weight >= nn_value_weight_cap {
            (0.0, 0.0)
        } else {
            (death_penalty, overhang_penalty_weight)
        }
    }

    pub(super) fn needs_incumbent_eval_rebaseline(
        incumbent_uses_network: bool,
        current_penalties: (f32, f32),
        candidate_penalties: (f32, f32),
    ) -> bool {
        incumbent_uses_network && current_penalties != candidate_penalties
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

        let mut boards = vec![0.0f32; batch_size * BOARD_HEIGHT * BOARD_WIDTH];
        let mut aux = vec![0.0f32; batch_size * AUX_FEATURES];
        let mut policies = vec![0.0f32; batch_size * NUM_ACTIONS];
        let mut values = vec![0.0f32; batch_size];
        let mut overhangs = vec![0.0f32; batch_size];
        let mut masks = vec![0.0f32; batch_size * NUM_ACTIONS];

        for (i, ex) in examples.iter().enumerate() {
            for (j, &val) in ex.board.iter().enumerate() {
                boards[i * BOARD_HEIGHT * BOARD_WIDTH + j] = val as f32;
            }

            let aux_offset = i * AUX_FEATURES;
            let aux_slice = &mut aux[aux_offset..aux_offset + AUX_FEATURES];
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
                policies[i * NUM_ACTIONS + j] = val;
            }
            values[i] = ex.value;
            overhangs[i] = ex.overhang_fields;
            for (j, &val) in ex.action_mask.iter().enumerate() {
                masks[i * NUM_ACTIONS + j] = if val { 1.0 } else { 0.0 };
            }
        }

        let boards_arr = PyArray1::from_vec(py, boards)
            .reshape([batch_size, BOARD_HEIGHT * BOARD_WIDTH])
            .unwrap();
        let aux_arr = PyArray1::from_vec(py, aux)
            .reshape([batch_size, AUX_FEATURES])
            .unwrap();
        let policies_arr = PyArray1::from_vec(py, policies)
            .reshape([batch_size, NUM_ACTIONS])
            .unwrap();
        let values_arr = PyArray1::from_vec(py, values);
        let overhangs_arr = PyArray1::from_vec(py, overhangs);
        let masks_arr = PyArray1::from_vec(py, masks)
            .reshape([batch_size, NUM_ACTIONS])
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
        buffer: &SharedBuffer,
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
    pub(super) fn worker_loop(worker_context: WorkerContext) {
        let WorkerContext {
            worker_id,
            is_evaluator_worker,
            settings,
            shared,
        } = worker_context;
        let mut agent = MCTSAgent::new(settings.config.clone());
        let mut loaded_model_version = u64::MAX;
        let mut loaded_model_step = 0u64;
        let mut loaded_with_network = !shared.incumbent.uses_network.load(Ordering::SeqCst);
        let mut pending_results: Vec<GameResult> = Vec::with_capacity(GAME_COMMIT_BATCH_SIZE);

        while shared.running.load(Ordering::SeqCst)
            && !Self::sync_incumbent_agent_if_needed(
                &settings,
                &shared.incumbent,
                &mut agent,
                &mut loaded_model_version,
                &mut loaded_model_step,
                &mut loaded_with_network,
                worker_id,
            )
        {
            thread::sleep(Duration::from_millis(500));
        }

        // Only worker 0 handles disk saves to avoid race conditions
        let is_save_worker = worker_id == 0;
        let mut next_snapshot_deadline = if settings.save_interval_seconds > 0.0 {
            Some(Instant::now() + Duration::from_secs_f64(settings.save_interval_seconds))
        } else {
            None
        };

        // Main generation loop
        while shared.running.load(Ordering::SeqCst) {
            if !Self::sync_incumbent_agent_if_needed(
                &settings,
                &shared.incumbent,
                &mut agent,
                &mut loaded_model_version,
                &mut loaded_model_step,
                &mut loaded_with_network,
                worker_id,
            ) {
                thread::sleep(Duration::from_millis(200));
                continue;
            }

            if is_evaluator_worker {
                let maybe_candidate = {
                    let mut pending = shared.pending_candidate.write().unwrap();
                    pending.take()
                };
                if let Some(candidate) = maybe_candidate {
                    if !pending_results.is_empty() {
                        let to_commit = std::mem::take(&mut pending_results);
                        Self::commit_game_results_batch(to_commit, &shared);
                    }

                    let current_incumbent_step = shared.incumbent.model_step.load(Ordering::SeqCst);
                    if candidate.model_step <= current_incumbent_step {
                        let incumbent_path = shared.incumbent.model_path.read().unwrap().clone();
                        Self::remove_model_artifacts_if_safe(
                            &candidate.model_path,
                            &settings.bootstrap_model_path,
                            &incumbent_path,
                            None,
                        );
                        continue;
                    }

                    {
                        let mut evaluating = shared.evaluating_candidate.write().unwrap();
                        *evaluating = Some(candidate.clone());
                    }

                    Self::run_candidate_evaluation(worker_id, candidate, &settings, &shared);

                    {
                        let mut evaluating = shared.evaluating_candidate.write().unwrap();
                        *evaluating = None;
                    }
                    loaded_model_version = u64::MAX;
                    loaded_with_network = !shared.incumbent.uses_network.load(Ordering::SeqCst);

                    if is_save_worker {
                        Self::persist_snapshot_if_due(
                            &settings.training_data_path,
                            shared.buffer.as_ref(),
                            settings.save_interval_seconds,
                            &mut next_snapshot_deadline,
                        );
                    }
                    continue;
                }
            }

            // Play one game
            if let Some(result) = agent.play_game(settings.max_placements, settings.add_noise) {
                pending_results.push(result);
                if pending_results.len() >= GAME_COMMIT_BATCH_SIZE {
                    let to_commit = std::mem::take(&mut pending_results);
                    Self::commit_game_results_batch(to_commit, &shared);
                }

                // Periodically save to disk for resume capability based on
                // wall-clock time.
                if is_save_worker {
                    if !pending_results.is_empty() {
                        let to_commit = std::mem::take(&mut pending_results);
                        Self::commit_game_results_batch(to_commit, &shared);
                    }
                    Self::persist_snapshot_if_due(
                        &settings.training_data_path,
                        shared.buffer.as_ref(),
                        settings.save_interval_seconds,
                        &mut next_snapshot_deadline,
                    );
                }
            }
        }

        if !pending_results.is_empty() {
            let to_commit = std::mem::take(&mut pending_results);
            Self::commit_game_results_batch(to_commit, &shared);
        }

        // Final save on shutdown (only worker 0)
        if is_save_worker && shared.buffer.len() > 0 {
            let n = shared.buffer.len();
            let _ = shared.buffer.persist_to_npz(&settings.training_data_path);
            eprintln!("[GameGenerator] Saved {} examples to disk", n);
        }

        eprintln!("[GameGenerator] Worker {} exiting", worker_id);
    }

    pub(super) fn sync_incumbent_agent_if_needed(
        settings: &WorkerSettings,
        incumbent: &IncumbentState,
        agent: &mut MCTSAgent,
        loaded_model_version: &mut u64,
        loaded_model_step: &mut u64,
        loaded_with_network: &mut bool,
        worker_id: usize,
    ) -> bool {
        let target_version = incumbent.model_version.load(Ordering::SeqCst);
        let target_uses_network = incumbent.uses_network.load(Ordering::SeqCst);
        if *loaded_model_version == target_version && *loaded_with_network == target_uses_network {
            return true;
        }

        let model_path = if target_uses_network {
            Some(incumbent.model_path.read().unwrap().clone())
        } else {
            None
        };
        let target_nn_value_weight = Self::load_atomic_f32(&incumbent.nn_value_weight);
        let target_death_penalty = Self::load_atomic_f32(&incumbent.death_penalty);
        let target_overhang_penalty_weight =
            Self::load_atomic_f32(&incumbent.overhang_penalty_weight);
        let Some(new_agent) = Self::create_rollout_agent(
            &settings.config,
            target_uses_network,
            settings.non_network_num_simulations,
            settings.bootstrap_use_min_max_q_normalization,
            target_nn_value_weight,
            target_death_penalty,
            target_overhang_penalty_weight,
            model_path.as_deref(),
            worker_id,
            "incumbent",
        ) else {
            return false;
        };

        *loaded_model_step = incumbent.model_step.load(Ordering::SeqCst);
        if worker_id == 0 {
            if target_uses_network {
                eprintln!(
                    "[GameGenerator] Loaded incumbent NN model step {} ({} workers, sims={}, nn_value_weight={:.6}, death_penalty={:.3}, overhang_penalty_weight={:.3})",
                    *loaded_model_step,
                    settings.num_workers,
                    settings.config.num_simulations,
                    target_nn_value_weight,
                    target_death_penalty,
                    target_overhang_penalty_weight
                );
            } else {
                eprintln!(
                    "[GameGenerator] Using no-network incumbent at step {} ({} workers, sims={})",
                    *loaded_model_step, settings.num_workers, settings.non_network_num_simulations
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
        settings: &WorkerSettings,
        shared: &WorkerSharedState,
    ) {
        let eval_config = Self::build_candidate_eval_config(&settings.config);
        let incumbent_uses_network_before = shared.incumbent.uses_network.load(Ordering::SeqCst);
        let incumbent_step_before = shared.incumbent.model_step.load(Ordering::SeqCst);
        let incumbent_nn_value_weight_before =
            Self::load_atomic_f32(&shared.incumbent.nn_value_weight);
        let previous_incumbent_avg_attack =
            Self::load_atomic_f32(&shared.incumbent.eval_avg_attack);
        let incumbent_path_before = shared.incumbent.model_path.read().unwrap().clone();

        let current_incumbent_death_penalty =
            Self::load_atomic_f32(&shared.incumbent.death_penalty);
        let current_incumbent_overhang_penalty_weight =
            Self::load_atomic_f32(&shared.incumbent.overhang_penalty_weight);
        let (candidate_death_penalty, candidate_overhang_penalty_weight) =
            Self::effective_search_penalties(
                candidate.nn_value_weight,
                settings.nn_value_weight_cap,
                current_incumbent_death_penalty,
                current_incumbent_overhang_penalty_weight,
            );
        let Some(candidate_agent) = Self::create_rollout_agent(
            &eval_config,
            true,
            settings.non_network_num_simulations,
            settings.bootstrap_use_min_max_q_normalization,
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
                &settings.bootstrap_model_path,
                &incumbent_path_before,
                None,
            );
            return;
        };

        let eval_start = Instant::now();

        // Play candidate games on fixed seeds for consistent benchmarking.
        struct CandidateGameResult {
            seed: u64,
            game_result: GameResult,
            replay_moves: Vec<ReplayMove>,
        }

        let mut candidate_results: Vec<CandidateGameResult> =
            Vec::with_capacity(settings.candidate_eval_seeds.len());
        for &seed in settings.candidate_eval_seeds.iter() {
            if !shared.running.load(Ordering::SeqCst) {
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            }

            // Deterministic fixed-seed evaluation:
            // same seed, no root noise, deterministic MCTS seed in eval_config.
            let candidate_outcome = candidate_agent.play_game_on_env(
                TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed),
                settings.max_placements,
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
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            }
        }

        if candidate_results.is_empty() {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &settings.bootstrap_model_path,
                &incumbent_path_before,
                None,
            );
            return;
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
        let mut incumbent_avg_attack = previous_incumbent_avg_attack;
        if Self::needs_incumbent_eval_rebaseline(
            incumbent_uses_network_before,
            (
                current_incumbent_death_penalty,
                current_incumbent_overhang_penalty_weight,
            ),
            (candidate_death_penalty, candidate_overhang_penalty_weight),
        ) {
            let Some(incumbent_agent) = Self::create_rollout_agent(
                &eval_config,
                true,
                settings.non_network_num_simulations,
                settings.bootstrap_use_min_max_q_normalization,
                incumbent_nn_value_weight_before,
                candidate_death_penalty,
                candidate_overhang_penalty_weight,
                Some(&incumbent_path_before),
                worker_id,
                "incumbent-baseline",
            ) else {
                eprintln!(
                    "[GameGenerator] Worker {} failed to load incumbent baseline from {} for adjusted evaluation",
                    worker_id,
                    incumbent_path_before.display()
                );
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            };
            let Some(recomputed_incumbent_avg_attack) =
                crate::runtime::evaluation::evaluate_avg_attack_on_fixed_seeds(
                    &incumbent_agent,
                    &settings.candidate_eval_seeds,
                    settings.max_placements,
                    false,
                    || shared.running.load(Ordering::SeqCst),
                )
            else {
                eprintln!(
                    "[GameGenerator] Failed to recompute incumbent baseline for adjusted penalties; rejecting candidate step {}",
                    candidate.model_step
                );
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            };
            incumbent_avg_attack = recomputed_incumbent_avg_attack;
        }

        let auto_promoted = previous_incumbent_avg_attack == 0.0 && !incumbent_uses_network_before;
        let promoted = auto_promoted || candidate_avg_attack > incumbent_avg_attack;
        let promoted_nn_value_weight = if promoted {
            candidate.nn_value_weight
        } else {
            incumbent_nn_value_weight_before
        };

        if promoted {
            let previous_incumbent_path = shared.incumbent.model_path.read().unwrap().clone();
            {
                let mut incumbent_path = shared.incumbent.model_path.write().unwrap();
                *incumbent_path = candidate.model_path.clone();
            }
            shared.incumbent.uses_network.store(true, Ordering::SeqCst);
            shared
                .incumbent
                .model_step
                .store(candidate.model_step, Ordering::SeqCst);
            Self::store_atomic_f32(&shared.incumbent.nn_value_weight, candidate.nn_value_weight);
            Self::store_atomic_f32(&shared.incumbent.death_penalty, candidate_death_penalty);
            Self::store_atomic_f32(
                &shared.incumbent.overhang_penalty_weight,
                candidate_overhang_penalty_weight,
            );
            if (candidate_death_penalty, candidate_overhang_penalty_weight) == (0.0, 0.0)
                && (
                    current_incumbent_death_penalty,
                    current_incumbent_overhang_penalty_weight,
                ) != (0.0, 0.0)
            {
                eprintln!(
                    "[GameGenerator] nn_value_weight reached cap ({:.6}), disabling death_penalty and overhang_penalty_weight",
                    settings.nn_value_weight_cap
                );
            }
            shared
                .incumbent
                .model_version
                .fetch_add(1, Ordering::SeqCst);
            Self::store_atomic_f32(&shared.incumbent.eval_avg_attack, candidate_avg_attack);

            Self::remove_model_artifacts_if_safe(
                &previous_incumbent_path,
                &settings.bootstrap_model_path,
                &candidate.model_path,
                None,
            );

            eprintln!(
                "[GameGenerator] Promoted candidate step {} (avg_attack {:.3} > incumbent {:.3}, games={}, nn_value_weight {:.6} -> {:.6}, eval trajectories discarded from replay)",
                candidate.model_step,
                candidate_avg_attack,
                incumbent_avg_attack,
                candidate_games,
                incumbent_nn_value_weight_before,
                candidate.nn_value_weight
            );
        } else {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &settings.bootstrap_model_path,
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
        }

        let evaluation_seconds = eval_start.elapsed().as_secs_f32();
        let promoted_death_penalty = Self::load_atomic_f32(&shared.incumbent.death_penalty);
        let promoted_overhang_penalty_weight =
            Self::load_atomic_f32(&shared.incumbent.overhang_penalty_weight);

        shared
            .model_eval_events
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
        shared: &WorkerSharedState,
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

            let game_number = shared.games_generated.fetch_add(1, Ordering::SeqCst) + 1;
            for example in &mut examples {
                example.game_number = game_number;
                example.game_total_attack = total_attack;
            }

            total_examples += examples.len() as u64;
            all_examples.extend(examples);

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
            shared.buffer.add_examples(all_examples);
        }
        if total_examples > 0 {
            shared
                .examples_generated
                .fetch_add(total_examples, Ordering::SeqCst);
        }
        let committed_games = completed_infos.len();
        if committed_games > 0 {
            shared
                .completed_games
                .write()
                .unwrap()
                .extend(completed_infos);
        }

        committed_games
    }

    pub(super) fn persist_buffer_snapshot(training_data_path: &Path, buffer: &SharedBuffer) {
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
