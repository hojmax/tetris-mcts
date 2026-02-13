//! MCTS Agent
//!
//! The main agent that runs MCTS search and self-play.

use pyo3::prelude::*;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;

use super::config::MCTSConfig;
use super::export::export_decision_node;
use super::results::{
    GameResult, GameStats, MCTSResult, MCTSTreeExport, TrainingExample, TreeNodeExport, TreeStats,
    TreeStatsAccumulator,
};
use super::search::{search_internal, search_internal_without_nn};
use crate::mcts::action_space::HOLD_ACTION_INDEX;

/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::nn::TetrisNN>,
}

fn compute_value_targets(
    attacks: &[u32],
    overhang_fields: &[u32],
    episode_ended_in_death: bool,
    terminal_placement_count: u32,
    max_placements: u32,
    death_penalty: f32,
    overhang_penalty_weight: f32,
) -> Vec<f32> {
    debug_assert_eq!(
        attacks.len(),
        overhang_fields.len(),
        "attacks and overhang_fields must have same length"
    );

    let num_states = attacks.len();
    let death_offset = if episode_ended_in_death {
        // Keep value targets aligned with MCTS terminal backup semantics:
        // penalty is determined by when the episode ended, not by source state index.
        super::utils::compute_death_penalty(terminal_placement_count, max_placements, death_penalty)
    } else {
        0.0
    };

    let mut values = vec![0.0f32; num_states];
    let mut cumulative = 0.0f32;
    for i in (0..num_states).rev() {
        let overhang_penalty =
            super::utils::compute_overhang_penalty(overhang_fields[i], overhang_penalty_weight);
        let step_reward = attacks[i] as f32 - overhang_penalty;
        cumulative += step_reward;
        values[i] = cumulative - death_offset;
    }

    values
}

fn compute_raw_value_targets(attacks: &[u32]) -> Vec<f32> {
    let mut values = vec![0.0f32; attacks.len()];
    let mut cumulative = 0.0f32;
    for i in (0..attacks.len()).rev() {
        cumulative += attacks[i] as f32;
        values[i] = cumulative;
    }
    values
}

#[pymethods]
impl MCTSAgent {
    #[new]
    pub fn new(config: MCTSConfig) -> Self {
        MCTSAgent { config, nn: None }
    }

    /// Load a neural network model
    ///
    /// Args:
    ///     path: Path to ONNX model file
    ///
    /// Returns:
    ///     True if loaded successfully, False otherwise
    pub fn load_model(&mut self, path: &str) -> bool {
        match crate::nn::TetrisNN::load(path) {
            Ok(nn) => {
                self.nn = Some(nn);
                true
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                false
            }
        }
    }

    /// Check if a neural network model is loaded
    pub fn has_model(&self) -> bool {
        self.nn.is_some()
    }

    /// Enable or disable board-embedding cache inside Rust NN inference.
    ///
    /// Returns False if no model is loaded.
    pub fn set_board_cache_enabled(&self, enabled: bool) -> bool {
        if let Some(nn) = self.nn.as_ref() {
            nn.set_board_cache_enabled(enabled);
            true
        } else {
            false
        }
    }

    /// Read and reset board-embedding cache stats (hits, misses, cache_size).
    ///
    /// Returns None if no model is loaded.
    pub fn get_and_reset_cache_stats(&self) -> Option<(u64, u64, usize)> {
        self.nn
            .as_ref()
            .map(crate::nn::TetrisNN::get_and_reset_cache_stats)
    }

    /// Play a full game using MCTS with the loaded model
    ///
    /// All neural network inference happens in Rust. Returns training data.
    ///
    /// Args:
    ///     max_placements: Maximum placements per game (hold actions don't count)
    ///     add_noise: Whether to add Dirichlet noise (for exploration)
    ///
    /// Returns:
    ///     GameResult with training examples
    #[pyo3(signature = (max_placements=100, add_noise=true))]
    pub fn play_game(&self, max_placements: u32, add_noise: bool) -> Option<GameResult> {
        let mut env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
        let mut states: Vec<(TetrisEnv, u32, u32, Vec<f32>, Vec<bool>)> = Vec::new();
        let mut attacks: Vec<u32> = Vec::new();
        let mut overhang_fields: Vec<u32> = Vec::new();
        let mut stats = GameStats::default();
        let mut valid_moves_sum: u32 = 0;
        let mut max_valid_moves: u32 = 0;
        let mut tree_stats_acc = TreeStatsAccumulator::new();
        let mut frame_index: u32 = 0;
        let mut placement_count: u32 = 0;

        while placement_count < max_placements {
            if env.game_over {
                break;
            }

            // Get action mask
            let mask = crate::nn::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                // This should only happen if game is over - if not, it's a bug
                debug_assert!(
                    env.game_over,
                    "No valid actions but game not over - this is a bug"
                );
                break;
            }
            let valid_moves = mask.iter().filter(|&&is_valid| is_valid).count() as u32;

            // Run MCTS search (NN-guided when model is loaded, otherwise uniform+zero bootstrap mode)
            let (result, move_tree_stats) = if let Some(nn) = self.nn.as_ref() {
                let (policy, nn_value) = match nn.predict_masked(
                    &env,
                    placement_count as usize,
                    &mask,
                    max_placements as usize,
                ) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!(
                            "[MCTSAgent] NN prediction failed at placement {}: {}. Ending game early.",
                            placement_count, e
                        );
                        break;
                    }
                };
                self.search(&env, policy, nn_value, add_noise, placement_count)
            } else {
                let (mcts_result, _root, tree_stats) =
                    search_internal_without_nn(&self.config, &env, add_noise, placement_count);
                (mcts_result, tree_stats)
            };

            valid_moves_sum += valid_moves;
            max_valid_moves = max_valid_moves.max(valid_moves);
            tree_stats_acc.add(move_tree_stats);
            if result.action == HOLD_ACTION_INDEX {
                stats.holds += 1;
            }

            // Store state only after search succeeds so state/reward arrays stay aligned.
            states.push((
                env.clone(),
                frame_index,
                placement_count,
                result.policy.clone(),
                mask.clone(),
            ));

            // Execute the selected action
            let attack = env
                .execute_action_index(result.action)
                .expect("MCTS selected action is not executable");
            attacks.push(attack);
            overhang_fields.push(super::utils::count_overhang_fields(&env));
            if result.action != HOLD_ACTION_INDEX {
                placement_count += 1;
            }
            frame_index += 1;

            // Collect stats from the attack result
            if let Some(ref attack_result) = env.get_last_attack_result() {
                let lines = attack_result.lines_cleared;
                stats.total_lines += lines;

                // Track combo
                if attack_result.combo > stats.max_combo {
                    stats.max_combo = attack_result.combo;
                }

                // Track back-to-backs
                if attack_result.back_to_back_attack > 0 {
                    stats.back_to_backs += 1;
                }

                // Track perfect clears
                if attack_result.is_perfect_clear {
                    stats.perfect_clears += 1;
                }

                // Categorize clear type
                if attack_result.is_tspin {
                    match lines {
                        1 => {
                            // Check if mini (mini has 0 base attack for single)
                            if attack_result.base_attack == 0 {
                                stats.tspin_minis += 1;
                            } else {
                                stats.tspin_singles += 1;
                            }
                        }
                        2 => stats.tspin_doubles += 1,
                        3 => stats.tspin_triples += 1,
                        _ => {}
                    }
                } else {
                    match lines {
                        1 => stats.singles += 1,
                        2 => stats.doubles += 1,
                        3 => stats.triples += 1,
                        4 => stats.tetrises += 1,
                        _ => {}
                    }
                }
            }
        }

        // Compute value targets (cumulative step reward from each position, minus death penalty)
        let num_states = states.len();
        debug_assert_eq!(
            states.len(),
            attacks.len(),
            "States and attacks should have same length"
        );
        debug_assert_eq!(
            states.len(),
            overhang_fields.len(),
            "States and overhang fields should have same length"
        );

        let values = compute_value_targets(
            &attacks,
            &overhang_fields,
            env.game_over,
            placement_count,
            max_placements,
            self.config.death_penalty,
            self.config.overhang_penalty_weight,
        );
        let raw_values = compute_raw_value_targets(&attacks);

        // Build training examples (use all moves)
        let mut examples = Vec::with_capacity(num_states);

        for i in 0..num_states {
            let (ref state, frame_idx, placement_idx, ref policy, ref mask) = states[i];

            // Convert board to binary (1 = filled, 0 = empty)
            let board: Vec<u8> = state
                .board_cells()
                .iter()
                .map(|&cell| if cell != 0 { 1 } else { 0 })
                .collect();

            let current_piece = state.get_current_piece().map(|p| p.piece_type).unwrap_or(0);

            let hold_piece = state.get_hold_piece().map(|p| p.piece_type).unwrap_or(7); // 7 = empty

            let hold_available = !state.is_hold_used();
            let next_queue = state.get_queue(5);

            examples.push(TrainingExample {
                board,
                current_piece,
                hold_piece,
                hold_available,
                next_queue,
                move_number: frame_idx,
                placement_count: placement_idx,
                policy: policy.clone(),
                value: values[i],
                raw_value: raw_values[i],
                action_mask: mask.clone(),
                overhang_fields: overhang_fields[i],
                game_number: 0,
                game_total_attack: 0,
            });
        }

        let total_attack: u32 = attacks.iter().sum();
        let total_overhang_fields: u32 = overhang_fields.iter().sum();
        let num_frames = states.len() as u32;
        let num_moves = placement_count;
        let avg_valid_moves = if num_frames > 0 {
            valid_moves_sum as f32 / num_frames as f32
        } else {
            0.0
        };
        let avg_overhang_fields = if num_frames > 0 {
            total_overhang_fields as f32 / num_frames as f32
        } else {
            0.0
        };

        let tree_stats = tree_stats_acc.finalize();
        let (cache_hits, cache_misses, cache_size) = if let Some(nn) = self.nn.as_ref() {
            nn.get_and_reset_cache_stats()
        } else {
            (0, 0, 0)
        };

        Some(GameResult {
            examples,
            total_attack,
            num_moves,
            avg_valid_actions: avg_valid_moves,
            max_valid_actions: max_valid_moves,
            stats,
            tree_stats,
            total_overhang_fields,
            avg_overhang_fields,
            cache_hits,
            cache_misses,
            cache_size,
        })
    }

    /// Generate multiple games of training data
    ///
    /// Args:
    ///     num_games: Number of games to play
    ///     max_placements: Maximum placements per game (hold actions don't count)
    ///     add_noise: Whether to add Dirichlet noise
    ///
    /// Returns:
    ///     List of all training examples from all games
    #[pyo3(signature = (num_games, max_placements=100, add_noise=true))]
    pub fn generate_games(
        &self,
        num_games: u32,
        max_placements: u32,
        add_noise: bool,
    ) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for _ in 0..num_games {
            if let Some(result) = self.play_game(max_placements, add_noise) {
                all_examples.extend(result.examples);
            }
        }

        all_examples
    }

    /// Run MCTS search and return both the result and the tree structure.
    ///
    /// This is for visualization/debugging - it exports the entire MCTS tree.
    ///
    /// Args:
    ///     env: The game state to search from
    ///     add_noise: Whether to add Dirichlet noise to root priors
    ///     placement_count: The current placement count in the game
    ///
    /// Returns:
    ///     Tuple of (MCTSResult, MCTSTreeExport), or None if no valid actions exist
    #[pyo3(signature = (env, add_noise=false, placement_count=0))]
    pub fn search_with_tree(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
        placement_count: u32,
    ) -> Option<(MCTSResult, MCTSTreeExport)> {
        // Keep parity with select_action/play_game: no valid actions means terminal state.
        let mask = crate::nn::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (mcts_result, root, _tree_stats) = if let Some(nn) = self.nn.as_ref() {
            let (policy, nn_value) = nn
                .predict_masked(
                    env,
                    placement_count as usize,
                    &mask,
                    self.config.max_placements as usize,
                )
                .expect("Neural network prediction failed");

            search_internal(
                &self.config,
                nn,
                env,
                policy,
                nn_value,
                add_noise,
                placement_count,
            )
        } else {
            search_internal_without_nn(&self.config, env, add_noise, placement_count)
        };

        // Export tree structure
        let mut nodes: Vec<TreeNodeExport> = Vec::new();
        export_decision_node(&root, None, None, &mut nodes);

        let tree_export = MCTSTreeExport {
            nodes,
            root_id: 0,
            num_simulations: self.config.num_simulations,
            selected_action: mcts_result.action,
            policy: mcts_result.policy.clone(),
        };

        Some((mcts_result, tree_export))
    }

    /// Run MCTS search and return the result (simpler than search_with_tree).
    ///
    /// This is the main method for selecting actions during play/evaluation.
    /// Handles NN inference internally.
    ///
    /// Args:
    ///     env: The game state to search from
    ///     add_noise: Whether to add Dirichlet noise to root priors
    ///     placement_count: The current placement count in the game
    ///
    /// Returns:
    ///     MCTSResult with selected action and policy, or None if no model loaded
    #[pyo3(signature = (env, add_noise=false, placement_count=0))]
    pub fn select_action(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
        placement_count: u32,
    ) -> Option<MCTSResult> {
        let nn = self.nn.as_ref()?;

        // Get action mask and initial policy
        let mask = crate::nn::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (policy, nn_value) = nn
            .predict_masked(
                env,
                placement_count as usize,
                &mask,
                self.config.max_placements as usize,
            )
            .expect("Neural network prediction failed");

        let (mcts_result, _root, _tree_stats) = search_internal(
            &self.config,
            nn,
            env,
            policy,
            nn_value,
            add_noise,
            placement_count,
        );
        Some(mcts_result)
    }
}

impl MCTSAgent {
    /// Get a reference to the neural network (if loaded).
    pub fn get_nn(&self) -> Option<&crate::nn::TetrisNN> {
        self.nn.as_ref()
    }

    /// Run MCTS search from a given state.
    pub fn search(
        &self,
        env: &TetrisEnv,
        policy: Vec<f32>,
        nn_value: f32,
        add_noise: bool,
        placement_count: u32,
    ) -> (MCTSResult, TreeStats) {
        let nn = self
            .nn
            .as_ref()
            .expect("Neural network required for search");
        let (result, _root, tree_stats) = search_internal(
            &self.config,
            nn,
            env,
            policy,
            nn_value,
            add_noise,
            placement_count,
        );
        (result, tree_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::super::nodes::{DecisionNode, MCTSNode};
    use super::*;
    use crate::mcts::NUM_ACTIONS;

    #[test]
    fn test_expand_action_updates_state() {
        // Create an agent with a model
        let config = MCTSConfig::default();
        let mut agent = MCTSAgent::new(config);

        // We need a model for expand_action to work
        // Skip test if no model is available
        if !agent.load_model("outputs/checkpoints/selfplay.onnx") {
            eprintln!("Skipping test - no model available");
            return;
        }

        // Create initial env with a known seed
        let env = TetrisEnv::with_seed(10, 20, 42);
        let initial_piece = env.get_current_piece().unwrap().piece_type;
        let initial_queue: Vec<usize> = env.get_queue(5);

        // Create a decision node
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let mut root = DecisionNode::new(env.clone(), 0);
        root.set_nn_output(&policy, 0.0);

        // Get a valid action
        // Run MCTS search to expand the action
        let mask = crate::nn::get_action_mask(&env);
        let nn = agent.nn.as_ref().unwrap();
        let (nn_policy, nn_value) = nn.predict_masked(&env, 0, &mask, 100).unwrap();

        let (_result, root_after, _tree_stats) =
            search_internal(&agent.config, nn, &env, nn_policy, nn_value, false, 0);

        // The root should have children after search
        assert!(
            !root_after.children.is_empty(),
            "Root should have children after search"
        );

        // Check that actions were expanded
        if let Some(child) = root_after.children.values().next() {
            match child {
                MCTSNode::Chance(chance_node) => {
                    let new_piece = chance_node.state.get_current_piece();

                    // The current piece should have changed (old piece was placed, new one spawned)
                    if let Some(ref new_p) = new_piece {
                        assert_ne!(
                            new_p.piece_type, initial_piece,
                            "Current piece should have changed after placing"
                        );
                    }

                    // The first piece from the old queue should now be current
                    if let Some(ref new_p) = new_piece {
                        assert_eq!(
                            new_p.piece_type, initial_queue[0],
                            "New current piece should be the first from old queue"
                        );
                    }
                }
                _ => panic!("First child should be a ChanceNode"),
            }
        }
    }

    #[test]
    fn test_compute_value_targets_without_death_penalty() {
        let attacks = vec![1, 2, 3];
        let overhang_fields = vec![0, 0, 0];

        let values = compute_value_targets(&attacks, &overhang_fields, false, 3, 100, 5.0, 0.0);

        assert_eq!(values.len(), 3);
        assert!((values[0] - 6.0).abs() < 1e-6);
        assert!((values[1] - 5.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_value_targets_death_penalty_uses_terminal_move_for_all_states() {
        let attacks = vec![1, 2, 3];
        let overhang_fields = vec![0, 0, 0];
        let max_placements = 100;
        let death_penalty = 5.0;

        let alive_values = compute_value_targets(
            &attacks,
            &overhang_fields,
            false,
            2,
            max_placements,
            death_penalty,
            0.0,
        );
        let dead_values = compute_value_targets(
            &attacks,
            &overhang_fields,
            true,
            2,
            max_placements,
            death_penalty,
            0.0,
        );

        let expected_offset =
            super::super::utils::compute_death_penalty(2, max_placements, death_penalty);

        for i in 0..attacks.len() {
            let observed_offset = alive_values[i] - dead_values[i];
            assert!(
                (observed_offset - expected_offset).abs() < 1e-6,
                "Death penalty offset mismatch at index {}: got {}, expected {}",
                i,
                observed_offset,
                expected_offset
            );
        }
    }

    #[test]
    fn test_compute_raw_value_targets() {
        let attacks = vec![1, 2, 3];
        let values = compute_raw_value_targets(&attacks);
        assert_eq!(values.len(), 3);
        assert!((values[0] - 6.0).abs() < 1e-6);
        assert!((values[1] - 5.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
    }
}
