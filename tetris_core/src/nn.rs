//! Neural Network Inference using tract-onnx
//!
//! Loads an ONNX model exported from PyTorch and provides
//! inference for Tetris policy and value prediction.

use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH, QUEUE_SIZE};
use crate::env::TetrisEnv;
use crate::piece::NUM_PIECE_TYPES;
const AUX_FEATURES: usize = 52; // 7 + 8 + 1 + 35 + 1

/// Neural network model wrapper
pub struct TetrisNN {
    model: Arc<TypedRunnableModel<TypedModel>>,
}

impl TetrisNN {
    /// Load an ONNX model from file
    pub fn load<P: AsRef<Path>>(path: P) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;

        Ok(TetrisNN {
            model: Arc::new(model),
        })
    }

    /// Run inference with action mask applied
    pub fn predict_masked(
        &self,
        env: &TetrisEnv,
        move_number: usize,
        action_mask: &[bool],
        max_moves: usize,
    ) -> TractResult<(Vec<f32>, f32)> {
        let (board_tensor, aux_tensor) = encode_state(env, move_number, max_moves);
        self.predict_masked_from_tensors(&board_tensor, &aux_tensor, action_mask)
    }

    pub fn predict_masked_from_tensors(
        &self,
        board_tensor: &[f32],
        aux_tensor: &[f32],
        action_mask: &[bool],
    ) -> TractResult<(Vec<f32>, f32)> {
        let board =
            tract_ndarray::Array4::from_shape_vec(
                (1, 1, BOARD_HEIGHT, BOARD_WIDTH),
                board_tensor.to_vec(),
            )?
                .into_tensor();

        let aux = tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor.to_vec())?
            .into_tensor();

        let outputs = self.model.run(tvec!(board.into(), aux.into()))?;

        let policy_logits: Vec<f32> = outputs[0].to_array_view::<f32>()?.iter().copied().collect();
        if policy_logits.len() != action_mask.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "NN policy output size mismatch: model={}, expected={} (action space changed; re-export ONNX)",
                    policy_logits.len(),
                    action_mask.len()
                ),
            )
            .into());
        }

        let value = outputs[1]
            .to_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .expect("NN value output tensor is empty - model is malformed");

        // Apply mask and softmax
        let policy = masked_softmax(&policy_logits, action_mask);

        Ok((policy, value))
    }
}

impl Clone for TetrisNN {
    fn clone(&self) -> Self {
        TetrisNN {
            model: Arc::clone(&self.model),
        }
    }
}

/// Encode a TetrisEnv state into neural network input tensors
pub fn encode_state(env: &TetrisEnv, move_number: usize, max_moves: usize) -> (Vec<f32>, Vec<f32>) {
    // Board tensor: binary (1 = filled, 0 = empty) - flatten to 200 values (will be reshaped to 1x20x10)
    let board = env.get_board();
    let board_tensor: Vec<f32> = board
        .iter()
        .flat_map(|row| row.iter().map(|&cell| if cell != 0 { 1.0 } else { 0.0 }))
        .collect();

    // Auxiliary features
    let mut aux = Vec::with_capacity(AUX_FEATURES);

    // Current piece: one-hot (7)
    let current_piece = env.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
    for i in 0..NUM_PIECE_TYPES {
        aux.push(if i == current_piece { 1.0 } else { 0.0 });
    }

    // Hold piece: one-hot (8) - 7 pieces + empty
    let hold_piece = env.get_hold_piece().map(|p| p.piece_type);
    for i in 0..NUM_PIECE_TYPES {
        aux.push(if hold_piece == Some(i) { 1.0 } else { 0.0 });
    }
    aux.push(if hold_piece.is_none() { 1.0 } else { 0.0 }); // Empty slot

    // Hold available: binary (1)
    aux.push(if !env.is_hold_used() { 1.0 } else { 0.0 });

    // Next queue: one-hot per slot (5 x 7 = 35)
    let queue = env.get_queue(QUEUE_SIZE);
    for slot in 0..QUEUE_SIZE {
        let piece_type = queue.get(slot).copied();
        for i in 0..NUM_PIECE_TYPES {
            aux.push(if piece_type == Some(i) { 1.0 } else { 0.0 });
        }
    }

    // Move number: normalized (1)
    let normalized_denominator = max_moves as f32;
    aux.push(move_number as f32 / normalized_denominator);

    (board_tensor, aux)
}

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
}

/// Get action mask from environment
pub fn get_action_mask(env: &TetrisEnv) -> Vec<bool> {
    use crate::mcts::{HOLD_ACTION_INDEX, NUM_ACTIONS};

    let current_placements = env.get_possible_placements();

    let mut mask = vec![false; NUM_ACTIONS];

    for p in current_placements {
        debug_assert!(p.action_index < NUM_ACTIONS);
        mask[p.action_index] = true;
    }

    let hold_is_available =
        !env.game_over && !env.is_hold_used() && env.get_current_piece().is_some();
    if hold_is_available {
        mask[HOLD_ACTION_INDEX] = true;
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_encoding_is_binary() {
        let mut env = TetrisEnv::new(10, 20);

        // Place some pieces to create a non-empty board
        env.hard_drop();
        env.hard_drop();

        let (board_tensor, _) = encode_state(&env, 0, 100);

        // Verify size
        assert_eq!(board_tensor.len(), BOARD_HEIGHT * BOARD_WIDTH);

        // Verify all values are 0.0 or 1.0
        for &val in &board_tensor {
            assert!(
                val == 0.0 || val == 1.0,
                "Board tensor should be binary, got value: {}",
                val
            );
        }

        // Verify encoding matches board state
        let board = env.get_board();
        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                let idx = y * BOARD_WIDTH + x;
                let expected = if board[y][x] != 0 { 1.0 } else { 0.0 };
                let actual = board_tensor[idx];
                assert_eq!(
                    actual, expected,
                    "Board[{},{}] with value {} should encode to {}, got {}",
                    y, x, board[y][x], expected, actual
                );
            }
        }
    }

    #[test]
    fn test_auxiliary_features_format() {
        let env = TetrisEnv::new(10, 20);
        let (_, aux) = encode_state(&env, 42, 100);

        // Total size: 7 + 8 + 1 + 35 + 1 = 52
        assert_eq!(aux.len(), AUX_FEATURES);
        assert_eq!(aux.len(), 52);

        let mut idx = 0;

        // Current piece: one-hot (7)
        let current_piece = env.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
        let current_onehot = &aux[idx..idx + NUM_PIECE_TYPES];
        let sum: f32 = current_onehot.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Current piece should be one-hot, sum = {}",
            sum
        );
        assert_eq!(
            current_onehot[current_piece], 1.0,
            "Current piece not encoded correctly"
        );
        idx += NUM_PIECE_TYPES;

        // Hold piece: one-hot (8) - 7 pieces + empty
        let hold_piece = env.get_hold_piece();
        let hold_onehot = &aux[idx..idx + NUM_PIECE_TYPES + 1];
        let sum: f32 = hold_onehot.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Hold piece should be one-hot, sum = {}",
            sum
        );
        if let Some(piece) = hold_piece {
            assert_eq!(
                hold_onehot[piece.piece_type], 1.0,
                "Hold piece not encoded correctly"
            );
        } else {
            assert_eq!(
                hold_onehot[NUM_PIECE_TYPES], 1.0,
                "Empty hold should set index 7"
            );
        }
        idx += NUM_PIECE_TYPES + 1;

        // Hold available: binary (1)
        let hold_avail = aux[idx];
        let expected_hold = if !env.is_hold_used() { 1.0 } else { 0.0 };
        assert_eq!(hold_avail, expected_hold, "Hold available incorrect");
        idx += 1;

        // Next queue: one-hot per slot (5 x 7 = 35)
        let queue = env.get_queue(QUEUE_SIZE);
        for i in 0..QUEUE_SIZE {
            let queue_slot = &aux[idx..idx + NUM_PIECE_TYPES];
            let sum: f32 = queue_slot.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Queue slot {} should be one-hot, sum = {}",
                i,
                sum
            );
            if i < queue.len() {
                assert_eq!(
                    queue_slot[queue[i]], 1.0,
                    "Queue slot {} not encoded correctly",
                    i
                );
            }
            idx += NUM_PIECE_TYPES;
        }

        // Move number: normalized (1)
        let move_norm = aux[idx];
        let expected_norm = 42.0 / 100.0;
        assert!(
            (move_norm - expected_norm).abs() < 1e-6,
            "Move number should be {}, got {}",
            expected_norm,
            move_norm
        );
        idx += 1;

        assert_eq!(
            idx, AUX_FEATURES,
            "Should consume all {} aux features",
            AUX_FEATURES
        );
    }

    #[test]
    fn test_encoding_specification() {
        // Verify the exact specification from CLAUDE.md:
        // | Board state    | 20 x 10    | Binary (1 = filled, 0 = empty)  |
        // | Current piece  | 7          | One-hot encoded                 |
        // | Hold piece     | 8          | One-hot (7 pieces + empty)      |
        // | Hold available | 1          | Binary (can use hold this turn) |
        // | Next queue     | 5 x 7 = 35 | One-hot encoded per slot        |
        // | Move number    | 1          | Normalized: move_idx / 100      |

        let env = TetrisEnv::new(10, 20);
        let (board, aux) = encode_state(&env, 50, 100);

        assert_eq!(board.len(), 20 * 10, "Board should be 20x10 = 200 values");
        assert_eq!(
            aux.len(),
            7 + 8 + 1 + 35 + 1,
            "Aux should be 7+8+1+35+1 = 52 values"
        );

        // Verify board is binary
        for &val in &board {
            assert!(val == 0.0 || val == 1.0, "Board must be binary");
        }

        // Verify current piece is one-hot (7)
        let current_sum: f32 = aux[0..7].iter().sum();
        assert!(
            (current_sum - 1.0).abs() < 1e-6,
            "Current piece must be one-hot"
        );

        // Verify hold piece is one-hot (8)
        let hold_sum: f32 = aux[7..15].iter().sum();
        assert!(
            (hold_sum - 1.0).abs() < 1e-6,
            "Hold piece must be one-hot (8 values)"
        );

        // Verify hold available is binary (1)
        let hold_avail = aux[15];
        assert!(
            hold_avail == 0.0 || hold_avail == 1.0,
            "Hold available must be binary"
        );

        // Verify queue is one-hot per slot (5 x 7 = 35)
        for i in 0..5 {
            let start = 16 + i * 7;
            let queue_sum: f32 = aux[start..start + 7].iter().sum();
            assert!(
                (queue_sum - 1.0).abs() < 1e-6,
                "Queue slot {} must be one-hot",
                i
            );
        }

        // Verify move number is normalized [0, 1]
        let move_norm = aux[51];
        assert!(
            move_norm >= 0.0 && move_norm <= 1.0,
            "Move number must be in [0, 1]"
        );
        assert!(
            (move_norm - 0.5).abs() < 1e-6,
            "Move 50 should normalize to 0.5"
        );
    }

    #[test]
    fn test_masked_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let probs = masked_softmax(&logits, &mask);

        assert!((probs[0] + probs[2] - 1.0).abs() < 1e-5);
        assert_eq!(probs[1], 0.0);
        assert_eq!(probs[3], 0.0);
    }
}
