//! Neural Network Inference using tract-onnx
//!
//! Loads an ONNX model exported from PyTorch and provides
//! inference for Tetris policy and value prediction.

use tract_onnx::prelude::*;
use std::path::Path;
use std::sync::Arc;

use crate::env::TetrisEnv;
use crate::piece::NUM_PIECE_TYPES;

// Constants matching Python network
const BOARD_HEIGHT: usize = 20;
const BOARD_WIDTH: usize = 10;
const QUEUE_SIZE: usize = 5;
const AUX_FEATURES: usize = 52;  // 7 + 8 + 1 + 35 + 1
const MAX_MOVES: usize = 100;

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
    ) -> TractResult<(Vec<f32>, f32)> {
        let (board_tensor, aux_tensor) = encode_state(env, move_number);

        let board = tract_ndarray::Array4::from_shape_vec(
            (1, 1, BOARD_HEIGHT, BOARD_WIDTH),
            board_tensor,
        )?.into_tensor();

        let aux = tract_ndarray::Array2::from_shape_vec(
            (1, AUX_FEATURES),
            aux_tensor,
        )?.into_tensor();

        let outputs = self.model.run(tvec!(board.into(), aux.into()))?;

        let policy_logits: Vec<f32> = outputs[0]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();

        let value = outputs[1]
            .to_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0);

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
fn encode_state(env: &TetrisEnv, move_number: usize) -> (Vec<f32>, Vec<f32>) {
    // Board tensor: flatten to 400 values (will be reshaped to 1x20x10)
    let board = env.get_board();
    let board_tensor: Vec<f32> = board
        .iter()
        .flat_map(|row| row.iter().map(|&cell| cell as f32))
        .collect();

    // Auxiliary features
    let mut aux = Vec::with_capacity(AUX_FEATURES);

    // Current piece: one-hot (7)
    let current_piece = env.get_current_piece()
        .map(|p| p.piece_type)
        .unwrap_or(0);
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
    aux.push(move_number as f32 / MAX_MOVES as f32);

    (board_tensor, aux)
}

/// Softmax with mask (invalid actions get 0 probability)
fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
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
    use crate::mcts::{get_action_space, NUM_ACTIONS};

    let action_space = get_action_space();
    let placements = env.get_possible_placements();

    let mut mask = vec![false; NUM_ACTIONS];

    for p in placements {
        if let Some(idx) = action_space.placement_to_index(p.piece.x, p.piece.y, p.piece.rotation) {
            mask[idx] = true;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_state() {
        let env = TetrisEnv::new(10, 20);
        let (board, aux) = encode_state(&env, 0);

        assert_eq!(board.len(), BOARD_HEIGHT * BOARD_WIDTH);
        assert_eq!(aux.len(), AUX_FEATURES);
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
