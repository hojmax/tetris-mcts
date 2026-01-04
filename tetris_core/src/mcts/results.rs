//! MCTS Result Types
//!
//! Data structures for search results and training data.

use pyo3::prelude::*;

/// MCTS search result
#[pyclass]
#[derive(Clone)]
pub struct MCTSResult {
    /// Policy (visit counts normalized) over all 734 actions
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

#[pymethods]
impl MCTSResult {
    fn __repr__(&self) -> String {
        format!(
            "MCTSResult(action={}, value={:.3}, simulations={})",
            self.action, self.value, self.num_simulations
        )
    }
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
    /// Move number (0-99)
    #[pyo3(get)]
    pub move_number: u32,
    /// MCTS policy target (734 values)
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Value target (cumulative attack from this point)
    #[pyo3(get)]
    pub value: f32,
    /// Action mask (734 values, true = valid)
    #[pyo3(get)]
    pub action_mask: Vec<bool>,
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
    /// Number of moves played
    #[pyo3(get)]
    pub num_moves: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_result_creation() {
        let result = MCTSResult {
            policy: vec![0.1; 734],
            action: 42,
            value: 0.75,
            num_simulations: 800,
        };

        assert_eq!(result.action, 42);
        assert!((result.value - 0.75).abs() < 0.001);
        assert_eq!(result.num_simulations, 800);
        assert_eq!(result.policy.len(), 734);
    }

    #[test]
    fn test_mcts_result_repr() {
        let result = MCTSResult {
            policy: vec![0.0; 734],
            action: 100,
            value: 1.5,
            num_simulations: 1000,
        };

        let repr = result.__repr__();
        assert!(repr.contains("action=100"));
        assert!(repr.contains("simulations=1000"));
        assert!(repr.contains("1.500"));
    }

    #[test]
    fn test_mcts_result_clone() {
        let result = MCTSResult {
            policy: vec![0.5; 734],
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
            policy: vec![0.0; 734],
            value: 10.5,
            action_mask: vec![false; 734],
        };

        assert_eq!(example.board.len(), 200);
        assert_eq!(example.current_piece, 2);
        assert_eq!(example.hold_piece, 7);
        assert!(example.hold_available);
        assert_eq!(example.next_queue.len(), 5);
        assert_eq!(example.move_number, 50);
        assert_eq!(example.policy.len(), 734);
        assert!((example.value - 10.5).abs() < 0.001);
        assert_eq!(example.action_mask.len(), 734);
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
                policy: vec![],
                value: 0.0,
                action_mask: vec![],
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
            policy: vec![],
            value: 0.0,
            action_mask: vec![],
        };
        assert_eq!(example.hold_piece, 7);
    }

    #[test]
    fn test_training_example_hold_piece_occupied() {
        let example = TrainingExample {
            board: vec![0; 200],
            current_piece: 0,
            hold_piece: 3, // S piece in hold
            hold_available: false, // Already used hold this turn
            next_queue: vec![],
            move_number: 0,
            policy: vec![],
            value: 0.0,
            action_mask: vec![],
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
            policy: vec![0.1; 734],
            value: 25.0,
            action_mask: vec![true; 734],
        };

        let cloned = example.clone();
        assert_eq!(cloned.board, example.board);
        assert_eq!(cloned.current_piece, example.current_piece);
        assert_eq!(cloned.hold_piece, example.hold_piece);
        assert_eq!(cloned.hold_available, example.hold_available);
        assert_eq!(cloned.next_queue, example.next_queue);
        assert_eq!(cloned.move_number, example.move_number);
        assert_eq!(cloned.value, example.value);
    }

    #[test]
    fn test_game_result_creation() {
        let result = GameResult {
            examples: vec![],
            total_attack: 150,
            num_moves: 75,
        };

        assert!(result.examples.is_empty());
        assert_eq!(result.total_attack, 150);
        assert_eq!(result.num_moves, 75);
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
            policy: vec![0.0; 734],
            value: 100.0,
            action_mask: vec![true; 734],
        };

        let example2 = TrainingExample {
            board: vec![0; 200],
            current_piece: 1,
            hold_piece: 0,
            hold_available: false,
            next_queue: vec![2, 3, 4, 5, 6],
            move_number: 1,
            policy: vec![0.0; 734],
            value: 95.0,
            action_mask: vec![true; 734],
        };

        let result = GameResult {
            examples: vec![example1, example2],
            total_attack: 100,
            num_moves: 2,
        };

        assert_eq!(result.examples.len(), 2);
        assert_eq!(result.examples[0].move_number, 0);
        assert_eq!(result.examples[1].move_number, 1);
        // Value should decrease as game progresses (less future attack remaining)
        assert!(result.examples[0].value > result.examples[1].value);
    }

    #[test]
    fn test_game_result_clone() {
        let result = GameResult {
            examples: vec![],
            total_attack: 200,
            num_moves: 100,
        };

        let cloned = result.clone();
        assert_eq!(cloned.total_attack, result.total_attack);
        assert_eq!(cloned.num_moves, result.num_moves);
        assert_eq!(cloned.examples.len(), result.examples.len());
    }

    #[test]
    fn test_policy_sums_to_one() {
        // A valid policy should sum to approximately 1.0
        let mut policy = vec![0.0; 734];
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
        let mut policy = vec![0.0; 734];
        let mut action_mask = vec![false; 734];

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
            policy: policy.clone(),
            value: 0.0,
            action_mask: action_mask.clone(),
        };

        // Check consistency: invalid actions should have zero policy
        for i in 0..734 {
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
