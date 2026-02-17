//! MCTS Agent
//!
//! The main agent that runs MCTS search and self-play.

use pyo3::prelude::*;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;
use crate::generator::evaluation::ReplayMove;

use super::config::MCTSConfig;
use super::export::export_decision_node;
use super::results::{
    GameResult, GameStats, MCTSResult, MCTSTreeExport, TrainingExample, TreeNodeExport, TreeStats,
    TreeStatsAccumulator,
};
use super::nodes::DecisionNode;
use super::search::{
    continue_search_internal, extract_subtree, search_internal, search_internal_without_nn,
};
use crate::constants::QUEUE_SIZE;
use crate::mcts::action_space::HOLD_ACTION_INDEX;
use crate::nn::TetrisNN;

/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::nn::TetrisNN>,
}

struct StateSnapshot {
    state: TetrisEnv,
    frame_idx: u32,
    placement_idx: u32,
    policy: Vec<f32>,
    mask: Vec<bool>,
    overhang_fields: u32,
    hole_count: u32,
}

fn compute_value_targets(attacks: &[u32]) -> Vec<f32> {
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
        let env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
        self.play_game_on_env(env, max_placements, add_noise)
            .map(|(result, _replay)| result)
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

    /// Play a full game on a pre-created environment, returning both the
    /// GameResult (with training examples) and a Vec of ReplayMove for replay.
    pub(crate) fn play_game_on_env(
        &self,
        mut env: TetrisEnv,
        max_placements: u32,
        add_noise: bool,
    ) -> Option<(GameResult, Vec<ReplayMove>)> {
        let mut states: Vec<StateSnapshot> = Vec::new();
        let mut attacks: Vec<u32> = Vec::new();
        let mut replay_moves: Vec<ReplayMove> = Vec::new();
        let mut stats = GameStats::default();
        let mut valid_moves_sum: u32 = 0;
        let mut max_valid_moves: u32 = 0;
        let mut tree_stats_acc = TreeStatsAccumulator::new();
        let mut frame_index: u32 = 0;
        let mut placement_count: u32 = 0;

        // Tree reuse state: carry the subtree from previous move
        let mut reused_root: Option<DecisionNode> = None;
        let mut tree_reuse_hits: u32 = 0;
        let mut tree_reuse_misses: u32 = 0;

        while placement_count < max_placements {
            if env.game_over {
                break;
            }

            // Get action mask
            let mask = crate::nn::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                debug_assert!(
                    env.game_over,
                    "No valid actions but game not over - this is a bug"
                );
                break;
            }
            let valid_moves = mask.iter().filter(|&&is_valid| is_valid).count() as u32;

            // Run MCTS search (with tree reuse if available)
            let (result, root, move_tree_stats) =
                self.search_maybe_reuse(&env, &mask, add_noise, placement_count, max_placements, reused_root.take())?;

            valid_moves_sum += valid_moves;
            max_valid_moves = max_valid_moves.max(valid_moves);
            tree_stats_acc.add(move_tree_stats);
            if result.action == HOLD_ACTION_INDEX {
                stats.holds += 1;
            }

            let (overhang_count, hole_count) =
                super::utils::count_overhang_fields_and_holes(&env);
            states.push(StateSnapshot {
                state: env.clone(),
                frame_idx: frame_index,
                placement_idx: placement_count,
                policy: result.policy.clone(),
                mask: mask.clone(),
                overhang_fields: overhang_count,
                hole_count,
            });

            // Execute the selected action
            let selected_action = result.action;
            let attack = env
                .execute_action_index(selected_action)
                .expect("MCTS selected action is not executable");
            attacks.push(attack);
            replay_moves.push(ReplayMove {
                action: selected_action,
                attack,
            });
            if selected_action != HOLD_ACTION_INDEX {
                placement_count += 1;
            }
            frame_index += 1;

            // Try to extract subtree for reuse on the next move.
            // The hidden piece is at queue position QUEUE_SIZE (index 5) in the real env.
            // For placement actions: after spawn_piece_internal, fill_queue(7) + pop_front
            //   → at least 6 items remain, position QUEUE_SIZE is the hidden piece.
            // For hold actions: queue doesn't change, but the tree still models a chance
            //   node for the hidden piece, so the same extraction logic applies.
            if self.config.reuse_tree {
                if let Some(root) = root {
                    let hidden_piece = env.piece_queue.get(QUEUE_SIZE).copied();
                    if let Some(piece) = hidden_piece {
                        if let Some(subtree) =
                            extract_subtree(root, selected_action, piece)
                        {
                            reused_root = Some(subtree);
                            tree_reuse_hits += 1;
                        } else {
                            tree_reuse_misses += 1;
                        }
                    } else {
                        tree_reuse_misses += 1;
                    }
                }
            }

            // Collect stats from the attack result
            if let Some(ref attack_result) = env.get_last_attack_result() {
                let lines = attack_result.lines_cleared;
                stats.total_lines += lines;

                if attack_result.combo > stats.max_combo {
                    stats.max_combo = attack_result.combo;
                }
                if attack_result.back_to_back_attack > 0 {
                    stats.back_to_backs += 1;
                }
                if attack_result.is_perfect_clear {
                    stats.perfect_clears += 1;
                }

                if attack_result.is_tspin {
                    match lines {
                        1 => {
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

        let num_states = states.len();
        debug_assert_eq!(
            states.len(),
            attacks.len(),
            "States and attacks should have same length"
        );

        let values = compute_value_targets(&attacks);
        let mut examples = Vec::with_capacity(num_states);

        for (snapshot, value) in states.iter().zip(values.iter().copied()) {
            let state = &snapshot.state;
            let board: Vec<u8> = state
                .board_cells()
                .iter()
                .map(|&cell| if cell != 0 { 1 } else { 0 })
                .collect();

            let current_piece = state.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
            let hold_piece = state.get_hold_piece().map(|p| p.piece_type).unwrap_or(7);
            let hold_available = !state.is_hold_used();
            let next_queue = state.get_queue(5);
            let next_hidden_piece_probs = crate::nn::next_hidden_piece_distribution(state);
            let raw_bumpiness = super::utils::compute_bumpiness(&state.column_heights[..state.width]);
            let column_heights =
                super::utils::normalize_column_heights(&state.column_heights[..state.width]);
            let row_fill_counts =
                super::utils::normalize_row_fill_counts(&state.row_fill_counts[..state.height], state.width);
            let total_blocks = super::utils::normalize_total_blocks(state.total_blocks);
            let bumpiness = super::utils::normalize_bumpiness(raw_bumpiness);
            let holes = super::utils::normalize_holes(snapshot.hole_count);
            let overhang_fields =
                super::utils::normalize_overhang_fields(snapshot.overhang_fields);
            let raw_max = state.column_heights[..state.width].iter().copied().max().unwrap_or(0);
            let raw_min = state.column_heights[..state.width].iter().copied().min().unwrap_or(0);
            let max_column_height = super::utils::normalize_max_column_height(raw_max);
            let min_column_height = super::utils::normalize_min_column_height(raw_min);

            examples.push(TrainingExample {
                board,
                current_piece,
                hold_piece,
                hold_available,
                next_queue,
                move_number: snapshot.frame_idx,
                placement_count: snapshot.placement_idx,
                combo: state.combo,
                back_to_back: state.back_to_back,
                next_hidden_piece_probs,
                column_heights,
                max_column_height,
                min_column_height,
                row_fill_counts,
                total_blocks,
                bumpiness,
                holes,
                policy: snapshot.policy.clone(),
                value,
                action_mask: snapshot.mask.clone(),
                overhang_fields,
                game_number: 0,
                game_total_attack: 0,
            });
        }

        let total_attack: u32 = attacks.iter().sum();
        let total_overhang_fields: u32 = states.iter().map(|s| s.overhang_fields).sum();
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

        Some((
            GameResult {
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
                tree_reuse_hits,
                tree_reuse_misses,
            },
            replay_moves,
        ))
    }

    /// Run MCTS search, optionally reusing a subtree from a previous search.
    ///
    /// Returns (MCTSResult, Option<DecisionNode>, TreeStats).
    /// The DecisionNode is returned for tree reuse extraction on the next move.
    /// Returns None (propagated from NN prediction failure) to discard the game.
    fn search_maybe_reuse(
        &self,
        env: &TetrisEnv,
        mask: &[bool],
        add_noise: bool,
        placement_count: u32,
        max_placements: u32,
        reused_root: Option<DecisionNode>,
    ) -> Option<(MCTSResult, Option<DecisionNode>, TreeStats)> {
        if let Some(nn) = self.nn.as_ref() {
            // NN mode: try tree reuse, fall back to fresh search
            if let Some(mut root) = reused_root {
                let (result, tree_stats) = self.continue_search_nn(nn, &mut root, add_noise);
                return Some((result, Some(root), tree_stats));
            }
            let (policy, nn_value) = match nn.predict_masked(
                env,
                placement_count as usize,
                mask,
                max_placements as usize,
            ) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!(
                        "[MCTSAgent] NN prediction failed at placement {}: {}. Discarding rollout.",
                        placement_count, e
                    );
                    return None;
                }
            };
            let (mcts_result, root, tree_stats) = search_internal(
                &self.config,
                nn,
                env,
                policy,
                nn_value,
                add_noise,
                placement_count,
            );
            Some((mcts_result, Some(root), tree_stats))
        } else {
            // Bootstrap mode: tree reuse still works with uniform priors
            if let Some(mut root) = reused_root {
                let evaluator = super::search::BootstrapLeafEvaluator;
                let (result, tree_stats) =
                    continue_search_internal(&self.config, &evaluator, &mut root, add_noise);
                return Some((result, Some(root), tree_stats));
            }
            let (mcts_result, root, tree_stats) =
                search_internal_without_nn(&self.config, env, add_noise, placement_count);
            Some((mcts_result, Some(root), tree_stats))
        }
    }

    /// Continue MCTS search on a reused tree root using NN evaluation.
    fn continue_search_nn(
        &self,
        nn: &TetrisNN,
        root: &mut DecisionNode,
        add_noise: bool,
    ) -> (MCTSResult, TreeStats) {
        let evaluator = super::search::NeuralLeafEvaluator {
            nn,
            nn_value_weight: self.config.nn_value_weight,
        };
        continue_search_internal(&self.config, &evaluator, root, add_noise)
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
    fn test_compute_value_targets_without_penalties() {
        let attacks = vec![1, 2, 3];
        let values = compute_value_targets(&attacks);

        assert_eq!(values.len(), 3);
        assert!((values[0] - 6.0).abs() < 1e-6);
        assert!((values[1] - 5.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_play_game_examples_use_current_state_overhang_and_holes() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 5;
        let agent = MCTSAgent::new(config);
        let result = agent
            .play_game(12, false)
            .expect("play_game should return a result");
        assert!(
            !result.examples.is_empty(),
            "play_game should generate at least one training example"
        );

        for example in result.examples.iter() {
            let mut env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
            env.board.fill(0);
            env.board[..example.board.len()].copy_from_slice(&example.board);
            env.invalidate_board_analysis_cache();

            let (expected_overhang, expected_holes_raw) =
                super::super::utils::count_overhang_fields_and_holes(&env);
            let expected_normalized_overhang =
                super::super::utils::normalize_overhang_fields(expected_overhang);
            let expected_holes =
                super::super::utils::normalize_holes(expected_holes_raw);

            assert!(
                (example.overhang_fields - expected_normalized_overhang).abs() < 1e-6,
                "overhang_fields must match the saved board at move {}",
                example.move_number
            );
            assert!(
                (example.holes - expected_holes).abs() < 1e-6,
                "holes must match the saved board at move {}",
                example.move_number
            );
        }
    }
}
