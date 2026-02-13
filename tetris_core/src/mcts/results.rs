//! MCTS Result Types
//!
//! Data structures for search results and training data.

use pyo3::prelude::*;

use crate::env::TetrisEnv;

/// MCTS search result
#[pyclass]
#[derive(Clone)]
pub struct MCTSResult {
    /// Policy (visit counts normalized) over all actions
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Selected action index
    #[pyo3(get)]
    pub action: usize,
    /// Root value estimate
    #[pyo3(get)]
    pub value: f32,
    /// Number of simulations run
    #[pyo3(get)]
    pub num_simulations: u32,
}

/// Training example returned from self-play
#[pyclass]
#[derive(Clone)]
pub struct TrainingExample {
    /// Board state flattened (20*10 = 200 values, 0 or 1)
    #[pyo3(get)]
    pub board: Vec<u8>,
    /// Current piece type (0-6)
    #[pyo3(get)]
    pub current_piece: usize,
    /// Hold piece type (0-6) or 7 if empty
    #[pyo3(get)]
    pub hold_piece: usize,
    /// Whether hold is available
    #[pyo3(get)]
    pub hold_available: bool,
    /// Next queue (up to 5 piece types)
    #[pyo3(get)]
    pub next_queue: Vec<usize>,
    /// Frame index in the game trajectory (includes hold actions)
    #[pyo3(get)]
    pub move_number: u32,
    /// Placement count at this frame (excludes hold actions)
    #[pyo3(get)]
    pub placement_count: u32,
    /// Current combo counter at this frame
    #[pyo3(get)]
    pub combo: u32,
    /// Whether back-to-back is active at this frame
    #[pyo3(get)]
    pub back_to_back: bool,
    /// Distribution over hidden next piece implied by current 7-bag state
    #[pyo3(get)]
    pub next_hidden_piece_probs: Vec<f32>,
    /// Height per column normalized by board height (0.0..1.0)
    #[pyo3(get)]
    pub column_heights: Vec<f32>,
    /// Maximum normalized column height
    #[pyo3(get)]
    pub max_column_height: f32,
    /// Minimum normalized column height
    #[pyo3(get)]
    pub min_column_height: f32,
    /// Filled cells per row normalized by board width (0.0..1.0)
    #[pyo3(get)]
    pub row_fill_counts: Vec<f32>,
    /// Total filled cells normalized by board area (0.0..1.0)
    #[pyo3(get)]
    pub total_blocks: f32,
    /// Sum of squared adjacent column-height deltas normalized to 0.0..1.0
    #[pyo3(get)]
    pub bumpiness: f32,
    /// Hole count normalized by maximum possible holes
    #[pyo3(get)]
    pub holes: f32,
    /// MCTS policy target (NUM_ACTIONS values)
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Value target (cumulative attack - overhang penalty - death penalty)
    #[pyo3(get)]
    pub value: f32,
    /// Raw value target (cumulative attack only, no penalties)
    #[pyo3(get)]
    pub raw_value: f32,
    /// Action mask (NUM_ACTIONS values, true = valid)
    #[pyo3(get)]
    pub action_mask: Vec<bool>,
    /// Overhang fields in the post-action board used for this state's step penalty
    #[pyo3(get)]
    pub overhang_fields: u32,
    /// 1-indexed global game number used for WandB per-game metrics
    #[pyo3(get)]
    pub game_number: u64,
    /// Total raw attack for the full game that produced this example
    #[pyo3(get)]
    pub game_total_attack: u32,
}

/// Statistics about the MCTS tree structure after a single search.
#[derive(Clone, Debug, Default)]
pub struct TreeStats {
    /// Average number of children per non-leaf node
    pub branching_factor: f32,
    /// Number of leaf nodes (nodes with no children)
    pub num_leaves: u32,
    /// Total number of nodes in the tree
    pub total_nodes: u32,
    /// Maximum depth from root
    pub max_depth: u32,
    /// Maximum actual attack value seen in any ChanceNode
    pub max_attack: u32,
}

/// Aggregated tree statistics across all moves in a game.
#[derive(Clone, Debug, Default)]
pub struct GameTreeStats {
    pub avg_branching_factor: f32,
    pub avg_leaves: f32,
    pub avg_total_nodes: f32,
    pub avg_max_depth: f32,
    pub max_tree_attack: u32,
}

/// Accumulator for building GameTreeStats from per-move TreeStats.
pub struct TreeStatsAccumulator {
    branching_factors: Vec<f32>,
    leaves: Vec<u32>,
    total_nodes: Vec<u32>,
    max_depths: Vec<u32>,
    max_attack: u32,
}

impl TreeStatsAccumulator {
    pub fn new() -> Self {
        Self {
            branching_factors: Vec::new(),
            leaves: Vec::new(),
            total_nodes: Vec::new(),
            max_depths: Vec::new(),
            max_attack: 0,
        }
    }

    pub fn add(&mut self, stats: TreeStats) {
        self.branching_factors.push(stats.branching_factor);
        self.leaves.push(stats.num_leaves);
        self.total_nodes.push(stats.total_nodes);
        self.max_depths.push(stats.max_depth);
        self.max_attack = self.max_attack.max(stats.max_attack);
    }

    pub fn finalize(self) -> GameTreeStats {
        let n = self.branching_factors.len() as f32;
        if n == 0.0 {
            return GameTreeStats::default();
        }
        GameTreeStats {
            avg_branching_factor: self.branching_factors.iter().sum::<f32>() / n,
            avg_leaves: self.leaves.iter().sum::<u32>() as f32 / n,
            avg_total_nodes: self.total_nodes.iter().sum::<u32>() as f32 / n,
            avg_max_depth: self.max_depths.iter().sum::<u32>() as f32 / n,
            max_tree_attack: self.max_attack,
        }
    }
}

/// Detailed game statistics for training logging
#[pyclass]
#[derive(Clone, Default)]
pub struct GameStats {
    /// Number of line clears by type
    #[pyo3(get)]
    pub singles: u32,
    #[pyo3(get)]
    pub doubles: u32,
    #[pyo3(get)]
    pub triples: u32,
    #[pyo3(get)]
    pub tetrises: u32,
    /// T-spin statistics
    #[pyo3(get)]
    pub tspin_minis: u32,
    #[pyo3(get)]
    pub tspin_singles: u32,
    #[pyo3(get)]
    pub tspin_doubles: u32,
    #[pyo3(get)]
    pub tspin_triples: u32,
    /// Perfect clears
    #[pyo3(get)]
    pub perfect_clears: u32,
    /// Back-to-back count (number of consecutive difficult clears)
    #[pyo3(get)]
    pub back_to_backs: u32,
    /// Maximum combo reached
    #[pyo3(get)]
    pub max_combo: u32,
    /// Total lines cleared
    #[pyo3(get)]
    pub total_lines: u32,
    /// Number of times hold action was used
    #[pyo3(get)]
    pub holds: u32,
}

#[pymethods]
impl GameStats {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result from playing a full game
#[pyclass]
#[derive(Clone)]
pub struct GameResult {
    /// Training examples from this game
    #[pyo3(get)]
    pub examples: Vec<TrainingExample>,
    /// Total attack scored
    #[pyo3(get)]
    pub total_attack: u32,
    /// Number of placements played (hold actions excluded)
    #[pyo3(get)]
    pub num_moves: u32,
    /// Average number of valid actions per frame in this game
    #[pyo3(get)]
    pub avg_valid_actions: f32,
    /// Maximum number of valid actions at any frame in this game
    #[pyo3(get)]
    pub max_valid_actions: u32,
    /// Detailed game statistics
    #[pyo3(get)]
    pub stats: GameStats,
    /// MCTS tree statistics aggregated across all moves
    pub tree_stats: GameTreeStats,
    /// Sum of per-move overhang fields across the game
    #[pyo3(get)]
    pub total_overhang_fields: u32,
    /// Average overhang fields per move across the game
    #[pyo3(get)]
    pub avg_overhang_fields: f32,
    /// Board embedding cache hits during this game
    #[pyo3(get)]
    pub cache_hits: u64,
    /// Board embedding cache misses during this game
    #[pyo3(get)]
    pub cache_misses: u64,
    /// Board embedding cache size at end of game
    #[pyo3(get)]
    pub cache_size: usize,
}

// =============================================================================
// Tree Export Types for Visualization
// =============================================================================

/// Exported tree node for visualization
#[pyclass]
#[derive(Clone)]
pub struct TreeNodeExport {
    /// Unique node ID
    #[pyo3(get)]
    pub id: usize,
    /// Node type: "decision" or "chance"
    #[pyo3(get)]
    pub node_type: String,
    /// Visit count
    #[pyo3(get)]
    pub visit_count: u32,
    /// Value sum
    #[pyo3(get)]
    pub value_sum: f32,
    /// Mean value (value_sum / visit_count)
    #[pyo3(get)]
    pub mean_value: f32,
    /// Individual backed-up values averaged into mean_value
    #[pyo3(get)]
    pub value_history: Vec<f32>,
    /// Raw neural network value estimate (for decision nodes)
    #[pyo3(get)]
    pub nn_value: f32,
    /// Is terminal state (for decision nodes)
    #[pyo3(get)]
    pub is_terminal: bool,
    /// Move number in game
    #[pyo3(get)]
    pub move_number: u32,
    /// Attack gained at this node (for chance nodes)
    #[pyo3(get)]
    pub attack: u32,
    /// Game state at this node
    #[pyo3(get)]
    pub state: TetrisEnv,
    /// Parent node ID (None for root)
    #[pyo3(get)]
    pub parent_id: Option<usize>,
    /// Edge label from parent (action index for decision->chance, piece type for chance->decision)
    #[pyo3(get)]
    pub edge_from_parent: Option<usize>,
    /// Child node IDs
    #[pyo3(get)]
    pub children: Vec<usize>,
    /// For decision nodes: valid action indices
    #[pyo3(get)]
    pub valid_actions: Vec<usize>,
    /// For decision nodes: priors for each valid action
    #[pyo3(get)]
    pub action_priors: Vec<f32>,
}

/// Exported MCTS tree for visualization
#[pyclass]
#[derive(Clone)]
pub struct MCTSTreeExport {
    /// All nodes in the tree (indexed by ID)
    #[pyo3(get)]
    pub nodes: Vec<TreeNodeExport>,
    /// Root node ID
    #[pyo3(get)]
    pub root_id: usize,
    /// Total simulations run
    #[pyo3(get)]
    pub num_simulations: u32,
    /// Selected action from root
    #[pyo3(get)]
    pub selected_action: usize,
    /// Policy from search
    #[pyo3(get)]
    pub policy: Vec<f32>,
}

#[pymethods]
impl MCTSTreeExport {
    /// Get the root node
    fn get_root(&self) -> TreeNodeExport {
        self.nodes[self.root_id].clone()
    }

    /// Get a node by ID
    fn get_node(&self, id: usize) -> Option<TreeNodeExport> {
        self.nodes.get(id).cloned()
    }

    /// Get children of a node
    fn get_children(&self, id: usize) -> Vec<TreeNodeExport> {
        if let Some(node) = self.nodes.get(id) {
            node.children
                .iter()
                .filter_map(|&child_id| self.nodes.get(child_id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get total number of nodes
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get maximum depth of tree
    fn max_depth(&self) -> usize {
        self.compute_depth(self.root_id)
    }

    /// Get all nodes at a given depth
    fn nodes_at_depth(&self, depth: usize) -> Vec<TreeNodeExport> {
        let mut result = Vec::new();
        self.collect_at_depth(self.root_id, 0, depth, &mut result);
        result
    }
}

impl MCTSTreeExport {
    fn compute_depth(&self, node_id: usize) -> usize {
        if let Some(node) = self.nodes.get(node_id) {
            if node.children.is_empty() {
                0
            } else {
                1 + node
                    .children
                    .iter()
                    .map(|&child_id| self.compute_depth(child_id))
                    .max()
                    .expect("non-empty children guaranteed by is_empty check")
            }
        } else {
            0
        }
    }

    fn collect_at_depth(
        &self,
        node_id: usize,
        current_depth: usize,
        target_depth: usize,
        result: &mut Vec<TreeNodeExport>,
    ) {
        if let Some(node) = self.nodes.get(node_id) {
            if current_depth == target_depth {
                result.push(node.clone());
            } else {
                for &child_id in &node.children {
                    self.collect_at_depth(child_id, current_depth + 1, target_depth, result);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::NUM_ACTIONS;

    #[test]
    fn test_mcts_result_creation() {
        let result = MCTSResult {
            policy: vec![0.1; NUM_ACTIONS],
            action: 42,
            value: 0.75,
            num_simulations: 800,
        };

        assert_eq!(result.action, 42);
        assert!((result.value - 0.75).abs() < 0.001);
        assert_eq!(result.num_simulations, 800);
        assert_eq!(result.policy.len(), NUM_ACTIONS);
    }

    #[test]
    fn test_mcts_result_clone() {
        let result = MCTSResult {
            policy: vec![0.5; NUM_ACTIONS],
            action: 10,
            value: 2.0,
            num_simulations: 500,
        };

        let cloned = result.clone();
        assert_eq!(cloned.action, result.action);
        assert_eq!(cloned.value, result.value);
        assert_eq!(cloned.num_simulations, result.num_simulations);
        assert_eq!(cloned.policy.len(), result.policy.len());
    }

    #[test]
    fn test_training_example_creation() {
        let example = TrainingExample {
            board: vec![0; 200],
            current_piece: 2,
            hold_piece: 7, // Empty
            hold_available: true,
            next_queue: vec![0, 1, 2, 3, 4],
            move_number: 50,
            placement_count: 45,
            combo: 0,
            back_to_back: false,
            next_hidden_piece_probs: vec![0.0; 7],
            column_heights: vec![0.0; 10],
            max_column_height: 0.0,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy: vec![0.0; NUM_ACTIONS],
            value: 10.5,
            raw_value: 13.0,
            action_mask: vec![false; NUM_ACTIONS],
            overhang_fields: 0,
            game_number: 0,
            game_total_attack: 0,
        };

        assert_eq!(example.board.len(), 200);
        assert_eq!(example.current_piece, 2);
        assert_eq!(example.hold_piece, 7);
        assert!(example.hold_available);
        assert_eq!(example.next_queue.len(), 5);
        assert_eq!(example.move_number, 50);
        assert_eq!(example.placement_count, 45);
        assert_eq!(example.combo, 0);
        assert!(!example.back_to_back);
        assert_eq!(example.next_hidden_piece_probs.len(), 7);
        assert_eq!(example.column_heights.len(), 10);
        assert_eq!(example.max_column_height, 0.0);
        assert_eq!(example.min_column_height, 0.0);
        assert_eq!(example.row_fill_counts.len(), 20);
        assert_eq!(example.total_blocks, 0.0);
        assert_eq!(example.bumpiness, 0.0);
        assert_eq!(example.holes, 0.0);
        assert_eq!(example.policy.len(), NUM_ACTIONS);
        assert!((example.value - 10.5).abs() < 0.001);
        assert!((example.raw_value - 13.0).abs() < 0.001);
        assert_eq!(example.action_mask.len(), NUM_ACTIONS);
    }

    #[test]
    fn test_training_example_valid_piece_types() {
        // current_piece should be 0-6
        for piece_type in 0..7 {
            let example = TrainingExample {
                board: vec![0; 200],
                current_piece: piece_type,
                hold_piece: 7,
                hold_available: true,
                next_queue: vec![],
                move_number: 0,
                placement_count: 0,
                combo: 0,
                back_to_back: false,
                next_hidden_piece_probs: vec![0.0; 7],
                column_heights: vec![0.0; 10],
                max_column_height: 0.0,
                min_column_height: 0.0,
                row_fill_counts: vec![0.0; 20],
                total_blocks: 0.0,
                bumpiness: 0.0,
                holes: 0.0,
                policy: vec![],
                value: 0.0,
                raw_value: 0.0,
                action_mask: vec![],
                overhang_fields: 0,
                game_number: 0,
                game_total_attack: 0,
            };
            assert!(example.current_piece < 7);
        }
    }

    #[test]
    fn test_training_example_hold_piece_empty() {
        let example = TrainingExample {
            board: vec![0; 200],
            current_piece: 0,
            hold_piece: 7, // 7 indicates empty
            hold_available: true,
            next_queue: vec![],
            move_number: 0,
            placement_count: 0,
            combo: 0,
            back_to_back: false,
            next_hidden_piece_probs: vec![0.0; 7],
            column_heights: vec![0.0; 10],
            max_column_height: 0.0,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy: vec![],
            value: 0.0,
            raw_value: 0.0,
            action_mask: vec![],
            overhang_fields: 0,
            game_number: 0,
            game_total_attack: 0,
        };
        assert_eq!(example.hold_piece, 7);
    }

    #[test]
    fn test_training_example_hold_piece_occupied() {
        let example = TrainingExample {
            board: vec![0; 200],
            current_piece: 0,
            hold_piece: 3,         // S piece in hold
            hold_available: false, // Already used hold this turn
            next_queue: vec![],
            move_number: 0,
            placement_count: 0,
            combo: 0,
            back_to_back: false,
            next_hidden_piece_probs: vec![0.0; 7],
            column_heights: vec![0.0; 10],
            max_column_height: 0.0,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy: vec![],
            value: 0.0,
            raw_value: 0.0,
            action_mask: vec![],
            overhang_fields: 0,
            game_number: 0,
            game_total_attack: 0,
        };
        assert_eq!(example.hold_piece, 3);
        assert!(!example.hold_available);
    }

    #[test]
    fn test_training_example_clone() {
        let example = TrainingExample {
            board: vec![1; 200],
            current_piece: 5,
            hold_piece: 2,
            hold_available: false,
            next_queue: vec![6, 0, 1],
            move_number: 99,
            placement_count: 75,
            combo: 4,
            back_to_back: true,
            next_hidden_piece_probs: vec![0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0],
            column_heights: vec![0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0],
            max_column_height: 0.2,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.1,
            bumpiness: 0.03,
            holes: 0.12,
            policy: vec![0.1; NUM_ACTIONS],
            value: 25.0,
            raw_value: 28.0,
            action_mask: vec![true; NUM_ACTIONS],
            overhang_fields: 12,
            game_number: 0,
            game_total_attack: 0,
        };

        let cloned = example.clone();
        assert_eq!(cloned.board, example.board);
        assert_eq!(cloned.current_piece, example.current_piece);
        assert_eq!(cloned.hold_piece, example.hold_piece);
        assert_eq!(cloned.hold_available, example.hold_available);
        assert_eq!(cloned.next_queue, example.next_queue);
        assert_eq!(cloned.move_number, example.move_number);
        assert_eq!(cloned.placement_count, example.placement_count);
        assert_eq!(cloned.combo, example.combo);
        assert_eq!(cloned.back_to_back, example.back_to_back);
        assert_eq!(
            cloned.next_hidden_piece_probs,
            example.next_hidden_piece_probs
        );
        assert_eq!(cloned.column_heights, example.column_heights);
        assert_eq!(cloned.max_column_height, example.max_column_height);
        assert_eq!(cloned.min_column_height, example.min_column_height);
        assert_eq!(cloned.row_fill_counts, example.row_fill_counts);
        assert_eq!(cloned.total_blocks, example.total_blocks);
        assert_eq!(cloned.bumpiness, example.bumpiness);
        assert_eq!(cloned.holes, example.holes);
        assert_eq!(cloned.value, example.value);
        assert_eq!(cloned.raw_value, example.raw_value);
        assert_eq!(cloned.overhang_fields, example.overhang_fields);
    }

    #[test]
    fn test_game_result_creation() {
        let result = GameResult {
            examples: vec![],
            total_attack: 150,
            num_moves: 75,
            avg_valid_actions: 0.0,
            max_valid_actions: 0,
            stats: GameStats::default(),
            tree_stats: GameTreeStats::default(),
            total_overhang_fields: 0,
            avg_overhang_fields: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            cache_size: 0,
        };

        assert!(result.examples.is_empty());
        assert_eq!(result.total_attack, 150);
        assert_eq!(result.num_moves, 75);
        assert_eq!(result.avg_overhang_fields, 0.0);
    }

    #[test]
    fn test_game_result_with_examples() {
        let example1 = TrainingExample {
            board: vec![0; 200],
            current_piece: 0,
            hold_piece: 7,
            hold_available: true,
            next_queue: vec![1, 2, 3, 4, 5],
            move_number: 0,
            placement_count: 0,
            combo: 0,
            back_to_back: false,
            next_hidden_piece_probs: vec![0.0; 7],
            column_heights: vec![0.0; 10],
            max_column_height: 0.0,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy: vec![0.0; NUM_ACTIONS],
            value: 100.0,
            raw_value: 105.0,
            action_mask: vec![true; NUM_ACTIONS],
            overhang_fields: 5,
            game_number: 0,
            game_total_attack: 0,
        };

        let example2 = TrainingExample {
            board: vec![0; 200],
            current_piece: 1,
            hold_piece: 0,
            hold_available: false,
            next_queue: vec![2, 3, 4, 5, 6],
            move_number: 1,
            placement_count: 1,
            combo: 1,
            back_to_back: false,
            next_hidden_piece_probs: vec![0.0; 7],
            column_heights: vec![0.0; 10],
            max_column_height: 0.0,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy: vec![0.0; NUM_ACTIONS],
            value: 95.0,
            raw_value: 99.0,
            action_mask: vec![true; NUM_ACTIONS],
            overhang_fields: 4,
            game_number: 0,
            game_total_attack: 0,
        };

        let result = GameResult {
            examples: vec![example1, example2],
            total_attack: 100,
            num_moves: 2,
            avg_valid_actions: 0.0,
            max_valid_actions: 0,
            stats: GameStats::default(),
            tree_stats: GameTreeStats::default(),
            total_overhang_fields: 0,
            avg_overhang_fields: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            cache_size: 0,
        };

        assert_eq!(result.examples.len(), 2);
        assert_eq!(result.examples[0].move_number, 0);
        assert_eq!(result.examples[1].move_number, 1);
        assert_eq!(result.examples[0].placement_count, 0);
        assert_eq!(result.examples[1].placement_count, 1);
        // Value should decrease as game progresses (less future attack remaining)
        assert!(result.examples[0].value > result.examples[1].value);
    }

    #[test]
    fn test_game_result_clone() {
        let result = GameResult {
            examples: vec![],
            total_attack: 200,
            num_moves: 100,
            avg_valid_actions: 0.0,
            max_valid_actions: 0,
            stats: GameStats::default(),
            tree_stats: GameTreeStats::default(),
            total_overhang_fields: 0,
            avg_overhang_fields: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            cache_size: 0,
        };

        let cloned = result.clone();
        assert_eq!(cloned.total_attack, result.total_attack);
        assert_eq!(cloned.num_moves, result.num_moves);
        assert_eq!(cloned.examples.len(), result.examples.len());
    }

    #[test]
    fn test_policy_sums_to_one() {
        // A valid policy should sum to approximately 1.0
        let mut policy = vec![0.0; NUM_ACTIONS];
        // Distribute probability over first 10 valid actions
        for i in 0..10 {
            policy[i] = 0.1;
        }

        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Policy should sum to 1.0");

        let result = MCTSResult {
            policy,
            action: 0,
            value: 0.0,
            num_simulations: 100,
        };

        let policy_sum: f32 = result.policy.iter().sum();
        assert!((policy_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_action_mask_consistency() {
        // If action_mask[i] is true, policy[i] could be non-zero
        // If action_mask[i] is false, policy[i] should be zero
        let mut policy = vec![0.0; NUM_ACTIONS];
        let mut action_mask = vec![false; NUM_ACTIONS];

        // Mark some actions as valid and give them probability
        for i in [0, 10, 20, 30, 40] {
            action_mask[i] = true;
            policy[i] = 0.2;
        }

        let example = TrainingExample {
            board: vec![0; 200],
            current_piece: 0,
            hold_piece: 7,
            hold_available: true,
            next_queue: vec![],
            move_number: 0,
            placement_count: 0,
            combo: 0,
            back_to_back: false,
            next_hidden_piece_probs: vec![0.0; 7],
            column_heights: vec![0.0; 10],
            max_column_height: 0.0,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; 20],
            total_blocks: 0.0,
            bumpiness: 0.0,
            holes: 0.0,
            policy: policy.clone(),
            value: 0.0,
            raw_value: 0.0,
            action_mask: action_mask.clone(),
            overhang_fields: 0,
            game_number: 0,
            game_total_attack: 0,
        };

        // Check consistency: invalid actions should have zero policy
        for i in 0..NUM_ACTIONS {
            if !example.action_mask[i] {
                assert_eq!(
                    example.policy[i], 0.0,
                    "Invalid action {} should have zero policy",
                    i
                );
            }
        }
    }
}
