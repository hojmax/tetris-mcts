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
use super::search::search_internal;
use crate::mcts::action_space::HOLD_ACTION_INDEX;

/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::nn::TetrisNN>,
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
    ///     max_moves: Maximum moves per game (default 100)
    ///     add_noise: Whether to add Dirichlet noise (for exploration)
    ///
    /// Returns:
    ///     GameResult with training examples, or None if no model loaded
    #[pyo3(signature = (max_moves=100, add_noise=true))]
    pub fn play_game(&self, max_moves: u32, add_noise: bool) -> Option<GameResult> {
        let nn = self.nn.as_ref()?;

        let mut env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
        let mut states: Vec<(TetrisEnv, u32, Vec<f32>, Vec<bool>)> = Vec::new();
        let mut attacks: Vec<u32> = Vec::new();
        let mut overhang_fields: Vec<u32> = Vec::new();
        let mut stats = GameStats::default();
        let mut valid_moves_sum: u32 = 0;
        let mut max_valid_moves: u32 = 0;
        let mut tree_stats_acc = TreeStatsAccumulator::new();

        for move_idx in 0..max_moves {
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
            valid_moves_sum += valid_moves;
            max_valid_moves = max_valid_moves.max(valid_moves);

            // Get NN policy and value for root
            let (policy, nn_value) =
                match nn.predict_masked(&env, move_idx as usize, &mask, max_moves as usize) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!(
                            "[MCTSAgent] NN prediction failed at move {}: {}. Ending game early.",
                            move_idx, e
                        );
                        break;
                    }
                };

            // Store state before making move
            states.push((env.clone(), move_idx, policy.clone(), mask.clone()));

            // Run MCTS search
            let (result, move_tree_stats) =
                self.search(&env, policy, nn_value, add_noise, move_idx);
            tree_stats_acc.add(move_tree_stats);
            if result.action == HOLD_ACTION_INDEX {
                stats.holds += 1;
            }

            // Execute the selected action
            let attack = env
                .execute_action_index(result.action)
                .expect("MCTS selected action is not executable");
            attacks.push(attack);
            overhang_fields.push(super::utils::count_overhang_fields(&env));

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

            // Update stored policy with MCTS policy
            if let Some(last) = states.last_mut() {
                last.2 = result.policy;
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

        let mut values = vec![0.0f32; num_states];
        let mut cumulative = 0.0f32;
        for i in (0..num_states).rev() {
            let overhang_penalty = super::utils::compute_overhang_penalty(
                overhang_fields[i],
                self.config.overhang_penalty_weight,
            );
            let step_reward = attacks[i] as f32 - overhang_penalty;
            cumulative += step_reward;
            let death_offset = if env.game_over {
                super::utils::compute_death_penalty(i as u32, max_moves, self.config.death_penalty)
            } else {
                0.0
            };
            values[i] = cumulative - death_offset;
        }

        // Build training examples (use all moves)
        let mut examples = Vec::with_capacity(num_states);

        for i in 0..num_states {
            let (ref state, move_num, ref policy, ref mask) = states[i];

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
                move_number: move_num,
                policy: policy.clone(),
                value: values[i],
                action_mask: mask.clone(),
                overhang_fields: overhang_fields[i],
            });
        }

        let total_attack: u32 = attacks.iter().sum();
        let total_overhang_fields: u32 = overhang_fields.iter().sum();
        let num_moves = states.len() as u32;
        let avg_valid_moves = if num_moves > 0 {
            valid_moves_sum as f32 / num_moves as f32
        } else {
            0.0
        };
        let avg_overhang_fields = if num_moves > 0 {
            total_overhang_fields as f32 / num_moves as f32
        } else {
            0.0
        };

        let tree_stats = tree_stats_acc.finalize();
        let (cache_hits, cache_misses, cache_size) = nn.get_and_reset_cache_stats();

        Some(GameResult {
            examples,
            total_attack,
            num_moves,
            avg_moves: avg_valid_moves,
            max_moves: max_valid_moves,
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
    ///     max_moves: Maximum moves per game
    ///     add_noise: Whether to add Dirichlet noise
    ///
    /// Returns:
    ///     List of all training examples from all games
    #[pyo3(signature = (num_games, max_moves=100, add_noise=true))]
    pub fn generate_games(
        &self,
        num_games: u32,
        max_moves: u32,
        add_noise: bool,
    ) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for _ in 0..num_games {
            if let Some(result) = self.play_game(max_moves, add_noise) {
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
    ///     move_number: The current move number in the game
    ///
    /// Returns:
    ///     Tuple of (MCTSResult, MCTSTreeExport) or None if no model loaded
    #[pyo3(signature = (env, add_noise=false, move_number=0))]
    pub fn search_with_tree(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
        move_number: u32,
    ) -> Option<(MCTSResult, MCTSTreeExport)> {
        let nn = self.nn.as_ref()?;

        // Get action mask and initial policy
        let mask = crate::nn::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (policy, nn_value) = nn
            .predict_masked(
                env,
                move_number as usize,
                &mask,
                self.config.max_moves as usize,
            )
            .expect("Neural network prediction failed");

        let (mcts_result, root, _tree_stats) = search_internal(
            &self.config,
            nn,
            env,
            policy,
            nn_value,
            add_noise,
            move_number,
        );

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
    ///     move_number: The current move number in the game
    ///
    /// Returns:
    ///     MCTSResult with selected action and policy, or None if no model loaded
    #[pyo3(signature = (env, add_noise=false, move_number=0))]
    pub fn select_action(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
        move_number: u32,
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
                move_number as usize,
                &mask,
                self.config.max_moves as usize,
            )
            .expect("Neural network prediction failed");

        let (mcts_result, _root, _tree_stats) = search_internal(
            &self.config,
            nn,
            env,
            policy,
            nn_value,
            add_noise,
            move_number,
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
        move_number: u32,
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
            move_number,
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
}
