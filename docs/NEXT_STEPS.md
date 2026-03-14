# Next Steps

- [ ] Maybe I can simplify code now and drop all the bootstrapping logic, and just pretrain the network on the replay buffer from earlier experiment as warm start. But kind of wack to have unreproducable training run. Like does the method actually work without bootstrapping now? Could we get rid of the penalties and stuff? Once everything is fixed we should try, is kind of ugly. Why would this be necessary. Could of course just be compute multiplier.
- [ ] Look at game that ends really early, and understand exactly why that happened. I would never expect it to look like higher reward to screw up the game board.
- [ ] New Metric: Is past attack + future attack roughly stable? Maybe we log per game variance in this estimate.
- [ ] I don't want random move sampling in the the last ~10 moves.
- [ ] Why is the GPU going cold repeatedly like 50% of the time?
- [ ] Why don't the final moves look full optimal? Like just clean out the board in the last 5 moves? Surely that is highest attack possible? Instead it does random nonsense.
- [ ] Why don't we always have 50 piece episodes??????
- [ ] One thing we could do is to make the amount of resources allocated to evaluating candidates configurable, and then spend alot on it every time we have upgrade, and then less and less each time they fail to upgrade. So we evaluate frequently when we are seeing big changes, and rarely when we are seeing small changes.
- [ ] Hmm the test set overfitting is kind of a problem. Maybe we should just never include those trajectories in the replay buffer? Yeah probably the right move. That way we have the low variance model estimate, whilst avoiding the issue of later models being better on the test set simply because they have been trained on it... Yeah drop the adding the samples to the test set.

- [ ] Less sharpening of temperature
- [ ] Try lower cpuct
- [ ] Higher exploration
- [ ] We are overfitting on the test evals.
- [ ] Go back to hyper params like v37 and then scale up and run on vast.ai
- [ ] What if you subtracted the root node value from all the child values before tanh? Then you immediately see if it is worse or better than expected.
- [ ] look at mcts tree across multiple moves, not just root

- [ ] Try MSE loss
- [ ] I screwed up the loose penalty I think
- [ ] Make network smaller and faster
- [ ] The tetris tspin stuff is still weird.
  - Investigate with make viz
- [ ] Fix the t-spin logic
- [ ] Are we actually using the whole action space? I guess we should log a bit about whether some actiosn are always masked out.
- [ ] Speed up data generation
- [ ] Multi machine RL environment generation?
- [ ] Replay buffer should grow gradually?

- [ ] Adding in alpha downweighting of value loss?
- [ ] Do offline learning experiments. Things to validate:
  - Learning rate and scheduler
  - Model size
  - Batch size
  - Policy / Value loss weighting
  - Weight decay
  - Optimizer
  - Architecture choices
  - Huber loss vs. MSE loss
- [ ] predict "n-step bootstrapped return" instead of "cummulative reward"

# Confusions

- [ ] How come the model is not even that much better even when doing tree reuse
- [ ] Why does MCTS sometimes have games with like 1 in attack? And sometimes 37? What is the difference between these games? Look at replay buffer and inspect.

# Deep Review

Done: ✅
In progress: 🟨

## Rust Code

### tetris_core/src/scoring.rs ✅

### tetris_core/src/lib.rs ✅

### tetris_core/src/piece.rs ✅

### tetris_core/src/nn.rs ✅

### tetris_core/src/moves.rs ✅

### tetris_core/src/kicks.rs ✅

### tetris_core/src/constants.rs ✅

### tetris_core/src/env/board.rs ✅

### tetris_core/src/env/clearing.rs ✅

### tetris_core/src/mcts/utils.rs ✅

### tetris_core/src/mcts/search.rs ✅

### tetris_core/src/mcts/results.rs ✅

### tetris_core/src/inference/mod.rs ✅

### tetris_core/src/replay/mod.rs ✅

### tetris_core/src/replay/types.rs ✅

### tetris_core/src/replay/npz.rs ✅

### tetris_core/src/runtime/evaluation.rs ✅

### tetris_core/src/runtime/mod.rs ✅

### tetris_core/src/runtime/game_generator/py_api.rs ✅

Why is there so much duplicate code in tetris_core/src/inference/mod.rs?

## Prompts To Run

> Why does this exist:
    ```

        fn evaluate_avg_attack_on_fixed_seeds(
            agent: &MCTSAgent,
            running: &Arc<AtomicBool>,
            max_placements: u32,
            candidate_eval_seeds: &[u64],
        ) -> Option<f32> {
            let mut total_attack: u64 = 0;
            let mut completed_games: u64 = 0;

            for &seed in candidate_eval_seeds {
                if !running.load(Ordering::SeqCst) {
                    return None;
                }
                let (result, _) = agent.play_game_on_env(
                    TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed),
                    max_placements,
                    false,
                )?;
                total_attack += result.total_attack as u64;
                completed_games += 1;
            }

            if completed_games == 0 {
                None
            } else {
                Some(total_attack as f32 / completed_games as f32)
            }
        }
    ``` in tetris_core/src/runtime/game_generator/runtime.rs, like could we not just call something from evaluate.rs instead of rewriting this type of function. Also why are we defining local constants that just map to global constants?:
    ```

    let batch_size = examples.len();
    let board_height = BOARD_HEIGHT;
    let board_width = BOARD_WIDTH;
    let num_actions = NUM_ACTIONS;
    let aux_features_size = AUX_FEATURES;

    ```

> Hmm the test set overfitting is kind of a problem. Maybe we should just never include those trajectories in the replay buffer? Yeah probably the right move. That way we have the low variance model estimate, whilst avoiding the issue of later models being better on the test set simply because they have been trained on it... Yeah drop the adding the samples to the test set. Maybe lets then lower the amount of games evalauted on to like 20. Like I think noise is less of an issue than one might think, since models should be getting better over time a priori expecation. So probably not that bad if some go through on noise.

> I think actually we should be saving the full search tree from training for the worst env, so I can look at that exact one later with make viz. So like I think we already have a struct for a search tree to save, but if not imrpove and make so we can save and load from python. And then make a script so I can load one and look at it, with all the info like the network predictions and all that. Probably we can compress a bit by avoiding to save the environments themsleves, but just the initial seed and the actions on the nodes, then we can just reconstruct env etc. We just have to make sure that the randomization is matching actually, so that the seeded random numbers in python is not different than the ones in rust.


> Why is CANDIDATE_EVAL_MCTS_SEED hardcoded? tetris_core/src/runtime/game_generator/runtime.rs

> In tetris_core/src/runtime/game_generator/py_api.rs, do we really need all these checks?:
    ```

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
    ```
    I think its fine not to do checks like that since it is not an public api, but just something I use. Also this looks very ugly:
    ```

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
    ```
    In general seems like the codebase have many places with huge blocks of passing in many many params. Usually there are better ways of doing this. Like another example of this:
    ```

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

    ```
    A remdey might be to pass (nested) objects into functions instead of these insane defintions:
    ```

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
    ```
    So if subfunctions only need a subset, than maybe we can logically group these such that we could just pass that subgroup into that function with the nesting. Or maybe just the full object everywhere sometimes.


> Is the slicing algorithm for loading up the gpu and efficient way of doing it? Any simpler way of ensuring the training data is on the gpu? Also why is my gpu not running out of memory, is like 2M training points not a crap ton of data. Or maybe not?


> When I do ctrl + c twice, I think this does not actually stop the wandb run properly and upload the current replay buffer and model, but just kills everything. I just want to stop the workers and wrap up, not just stop workers and die.

> Is this ever used?:
    ```

        /// Get statistics as a typed Python dictionary.
        pub fn get_stats(&self, py: Python<'_>) -> HashMap<String, PyObject> {
            let mut stats = HashMap::new();
            stats.insert(
                "games_generated".to_string(),
                self.games_generated().into_py(py),
            );
            stats.insert(
                "examples_generated".to_string(),
                self.examples_generated().into_py(py),
            );
            stats.insert("is_running".to_string(), self.is_running().into_py(py));
            stats.insert("buffer_size".to_string(), self.buffer_size().into_py(py));
            stats.insert(
                "incumbent_model_step".to_string(),
                self.incumbent_model_step.load(Ordering::SeqCst).into_py(py),
            );
            stats.insert(
                "incumbent_uses_network".to_string(),
                self.incumbent_uses_network().into_py(py),
            );
            stats.insert(
                "incumbent_eval_avg_attack".to_string(),
                Self::load_atomic_f32(&self.incumbent_eval_avg_attack).into_py(py),
            );
            stats
        }

    ```
    I also think in general it seems like the rust code is often exposing things that are not really used, like string representations or dict representations or stuff. Get rid of that.


> In tetris_core/src/search/utils.rs why do we need:

    ```

        let mut overhang_fields: u32 = 0;
        let mut holes: u32 = 0;
        for x in 0..env.width {
            let mut seen_filled = false;
            for y in 0..env.height {
                let idx = y * env.width + x;
                let cell = board[idx];
                if cell != 0 {
                    seen_filled = true;
                    continue;
                }
                if seen_filled {
                    overhang_fields += 1;
                    if !reachable[idx] {
                        holes += 1;
                    }
                }
            }
        }
    ```

    Is holes not just 200 - blocks - visit_count. I.e. holes are what is left when you account for all the air you can visit and the placed blocks? But actually thinking of it, since we anyway need the loop for overhang, then kind of makes sense to just do the holes in there?

> is this ever used?:

    ```

        /// Convert to dictionary for logging.
        pub fn to_dict(&self) -> HashMap<String, f32> {
            let mut d = HashMap::new();
            d.insert("eval/num_games".to_string(), self.num_games as f32);
            d.insert("eval/total_attack".to_string(), self.total_attack as f32);
            d.insert("eval/max_attack".to_string(), self.max_attack as f32);
            d.insert("eval/total_lines".to_string(), self.total_lines as f32);
            d.insert("eval/max_lines".to_string(), self.max_lines as f32);
            d.insert("eval/avg_attack".to_string(), self.avg_attack);
            d.insert("eval/avg_lines".to_string(), self.avg_lines);
            d.insert("eval/avg_moves".to_string(), self.avg_moves);
            d.insert("eval/attack_per_piece".to_string(), self.attack_per_piece);
            d.insert("eval/lines_per_piece".to_string(), self.lines_per_piece);
            d.insert("eval/avg_tree_nodes".to_string(), self.avg_tree_nodes);
            d
        }
    ```
    tetris_core/src/runtime/evaluation.rs

> Is the replay storage format the most efficient?

> Why is evaluate_agent writing to disk instead of being in memory?

> tetris_core/src/replay/npz.rs seems really repetitive. Is there not a library one could use for all this? Or some way of condensing it?

> I want a new metric that considers how much variance there is in the value estimate along the chosen trajectory. Like if the model was a perfect predictor, then the past cummulative attack + predicted future, should be fully constant throughout the trajectory. So I want to for each game, get the variance of this for the chosen actions, so variance over past cum + network prediction for each step of played game and return that. Or is variance like the best metric here, or what would you think would be useful to log here for this?

> Should we try to compress the policy targets for saving to disk? Like only saving the floats for the valid actions, rest is just zeros anyway right?

> Am I correctly currently taking the average over 100 steps when logging the step info to wanddb, like I do for games, or am I logging every single GPU step?

> I still cant see a full tree with reuse in the make viz. It just loads for ever please fix.

> Is there a cleaner way than what I am doing right now for syncing replay buffer to GPU?

> Why is tetris_core/src/replay/mod.rs its own file?

> Clean up dead code please around the repo

> How could I improve my model architecture?

> Why is there so much duplicate code in tetris_core/src/inference/mod.rs? And why are there so many different functions in mod.rs? Seems like a lot of code specialized for a single use case that could have been generalized?

> Why is placement count passed in here?:

    ```

    /// Encode a TetrisEnv state into neural network input tensors
    #[cfg(test)]
    fn encode_state(
        env: &TetrisEnv,
        placement_count: usize,
        max_placements: usize,
    ) -> TractResult<(Tensor, Tensor)> {
        let (board_tensor, aux_tensor) = encode_state_features(env, placement_count, max_placements)?;
        let board =
            tract_ndarray::Array4::from_shape_vec((1, 1, BOARD_HEIGHT, BOARD_WIDTH), board_tensor)?
                .into_tensor();
        let aux = tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor)?.into_tensor();
        Ok((board, aux))
    }
    ```
    Is that not encoded in the env? Why is this not tracked on the env in general?

> Instead of encode_board_features should we just save that directly on the env?

> Why the hell is there fallback here?:

```

    let current_piece = env.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
```

We don't want silent fallbacks when unexpected things happen!!! Please look through the whole codebase for similar cases, and fix them. Things failing loudly is fine and good! Also like why a max??:

```

pub fn denormalize_combo_feature(combo_feature: f32) -> u32 {
    (combo_feature.max(0.0) * COMBO_NORMALIZATION_MAX as f32).round() as u32
}
```

> I want to do a quicktest type of thing on an actual replay buffer. So the test is pointed to a replay buffer I track in git, with like 10K states, and we have a bunch of consistency checks there on the replay buffer which we check whether hold in practice. Suggest a long list of potential consistency checks we could do on the replay buffer.

> Is the minus the max logit just a way of making the calculation more stable? or what is it doing?:

```

/// Softmax with mask (invalid actions get 0 probability)
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&x, _)| x)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut result = vec![0.0; logits.len()];
    let mut sum = 0.0;

    for (i, (&logit, &valid)) in logits.iter().zip(mask.iter()).enumerate() {
        if valid {
            let exp_val = (logit - max_logit).exp();
            result[i] = exp_val;
            sum += exp_val;
        }
    }

    if sum > 0.0 {
        for x in &mut result {
            *x /= sum;
        }
    }

    result
```

> Lets try to do a memory .md file. In the repo, make a memories folder. Then have a main MEMORY.md file. Every time you learn something or get feedback from me, write to that. Like something goes wrong, you figure out how to call that command correctly write it down. Something was uintutivie and you had to think alot to figure it out, write down the final realization. Something was hard to find, write it down. Anything else, write it down. Do this liberally. Now once the MEMORY.md hits 200 lines, you do compaction, and group together relevant stuff and write out into other files .md in that memorieis folder. I want every session of you that starts to always start by reading MEMORY.md. For the long term files, if there already exists a file that matches the theme of the chunck of short term stuff that you are compacting, add to that. If not, just make a new one with a descriptive name. Now add these rules to AGENTS.md and make the memories folder and file.

## Python Code

### tetris_bot/config.py ✅

### tetris_bot/run_setup.py ✅

### scripts/train.py ✅

### tetris_bot/ml/loss.py ✅

# Backlog

- [ ] Is huber loss even better than MSE loss?
- [ ] Reading through and validating all code
- [ ] Do we need a larger network?
- [ ] Looking at training data
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
- [ ] Maybe this is just a hella slow learning algorithm, and we need to scale up compute.
- [ ] Visualizing MCTS search and verifying correctness
- [ ] Adding hand crafted heuristics to offload work off the neural network
  - can t-spin filter?
- [ ] Try to handcraft great Tetris bot.
- [ ] Make a readme

## Test Environment State

Current Piece:

```
T
```

Queue:

```
I,O,L,S,Z
```

Board:

```
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
....LLL...
LL...LLLLL
LLL.LLLLLL
```
