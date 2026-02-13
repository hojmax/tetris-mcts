//! Neural Network Inference using tract-onnx
//!
//! Loads split ONNX models (conv backbone + heads) and board projection weights from binary.
//! Caches board embeddings to skip conv + board projection on repeated board states.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use tract_onnx::prelude::*;

use crate::constants::{
    AUX_FEATURES, BOARD_HEIGHT, BOARD_WIDTH, COMBO_NORMALIZATION_MAX, NUM_PIECE_TYPES, QUEUE_SIZE,
};
use crate::env::TetrisEnv;

/// Neural network model wrapper with board embedding cache
pub struct TetrisNN {
    conv_model: Arc<TypedRunnableModel<TypedModel>>,
    heads_model: Arc<TypedRunnableModel<TypedModel>>,
    board_proj_weight: Arc<Array2<f32>>, // (board_hidden, conv_out_size)
    board_proj_bias: Arc<Array1<f32>>,   // (board_hidden,)
    board_hidden: usize,
    conv_out_size: usize,
    board_cache: RefCell<HashMap<[u64; 4], Array1<f32>>>,
    cache_hits: Cell<u64>,
    cache_misses: Cell<u64>,
    cache_enabled: Cell<bool>,
}

impl TetrisNN {
    /// Load split models from file.
    /// Given a base path like "latest.onnx", loads:
    /// - "latest.conv.onnx" (conv backbone)
    /// - "latest.heads.onnx" (gated fusion + heads)
    /// - "latest.fc.bin" (board projection weight + bias)
    pub fn load<P: AsRef<Path>>(path: P) -> TractResult<Self> {
        let base = path.as_ref().with_extension("");
        let conv_path = base.with_extension("conv.onnx");
        let heads_path = base.with_extension("heads.onnx");
        let fc_path = base.with_extension("fc.bin");

        let conv_model = tract_onnx::onnx()
            .model_for_path(&conv_path)?
            .into_optimized()?
            .into_runnable()?;

        let heads_model = tract_onnx::onnx()
            .model_for_path(&heads_path)?
            .into_optimized()?
            .into_runnable()?;

        let (board_proj_weight, board_proj_bias) = load_fc_binary(&fc_path)?;
        let board_hidden = board_proj_weight.nrows();
        let conv_out_size = board_proj_weight.ncols();
        if conv_out_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Board projection width {} is invalid; expected > 0",
                    conv_out_size
                ),
            )
            .into());
        }

        Ok(TetrisNN {
            conv_model: Arc::new(conv_model),
            heads_model: Arc::new(heads_model),
            board_proj_weight: Arc::new(board_proj_weight),
            board_proj_bias: Arc::new(board_proj_bias),
            board_hidden,
            conv_out_size,
            board_cache: RefCell::new(HashMap::new()),
            cache_hits: Cell::new(0),
            cache_misses: Cell::new(0),
            cache_enabled: Cell::new(true),
        })
    }

    fn compute_board_embedding(&self, board_f32: &[f32]) -> TractResult<Array1<f32>> {
        let board_tensor = tract_ndarray::Array4::from_shape_vec(
            (1, 1, BOARD_HEIGHT, BOARD_WIDTH),
            board_f32.to_vec(),
        )?
        .into_tensor();

        let conv_output = self.conv_model.run(tvec!(board_tensor.into()))?;
        let conv_out: Vec<f32> = conv_output[0]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();
        if conv_out.len() != self.conv_out_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Conv output size mismatch: conv.onnx={} fc.bin={} (re-export split models together)",
                    conv_out.len(),
                    self.conv_out_size
                ),
            )
            .into());
        }

        let conv_arr = Array1::from_vec(conv_out);
        Ok(self.board_proj_weight.dot(&conv_arr) + self.board_proj_bias.as_ref())
    }

    /// Run inference with action mask applied, using board embedding cache.
    pub fn predict_masked(
        &self,
        env: &TetrisEnv,
        placement_count: usize,
        action_mask: &[bool],
        max_placements: usize,
    ) -> TractResult<(Vec<f32>, f32)> {
        let (board_f32, aux_vec) = encode_state_features(env, placement_count, max_placements)?;

        let board_embed = if self.cache_enabled.get() {
            let board_key = pack_board(env);
            let board_embed = {
                let cache = self.board_cache.borrow();
                cache.get(&board_key).cloned()
            };

            match board_embed {
                Some(embed) => {
                    self.cache_hits.set(self.cache_hits.get() + 1);
                    embed
                }
                None => {
                    self.cache_misses.set(self.cache_misses.get() + 1);
                    let embed = self.compute_board_embedding(&board_f32)?;
                    self.board_cache
                        .borrow_mut()
                        .insert(board_key, embed.clone());
                    embed
                }
            }
        } else {
            self.compute_board_embedding(&board_f32)?
        };

        let board_h_tensor =
            tract_ndarray::Array2::from_shape_vec((1, self.board_hidden), board_embed.to_vec())?
                .into_tensor();
        let aux_tensor =
            tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_vec)?.into_tensor();
        let heads_output = self
            .heads_model
            .run(tvec!(board_h_tensor.into(), aux_tensor.into()))?;

        let policy_logits: Vec<f32> = heads_output[0]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();

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

        let value = heads_output[1]
            .to_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .expect("NN value output tensor is empty - model is malformed");

        let policy = masked_softmax(&policy_logits, action_mask);

        Ok((policy, value))
    }

    pub fn predict_masked_from_tensors(
        &self,
        board_tensor: &[f32],
        aux_tensor: &[f32],
        action_mask: &[bool],
    ) -> TractResult<(Vec<f32>, f32)> {
        // No caching for raw tensor inputs (used only by debug functions)
        let board_embed = self.compute_board_embedding(board_tensor)?;
        let board_h_tensor =
            tract_ndarray::Array2::from_shape_vec((1, self.board_hidden), board_embed.to_vec())?
                .into_tensor();
        let aux_tensor =
            tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor.to_vec())?
                .into_tensor();
        let heads_output = self
            .heads_model
            .run(tvec!(board_h_tensor.into(), aux_tensor.into()))?;

        let policy_logits: Vec<f32> = heads_output[0]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();

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

        let value = heads_output[1]
            .to_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .expect("NN value output tensor is empty - model is malformed");

        let policy = masked_softmax(&policy_logits, action_mask);

        Ok((policy, value))
    }
}

impl TetrisNN {
    pub fn set_board_cache_enabled(&self, enabled: bool) {
        self.cache_enabled.set(enabled);
        if !enabled {
            self.board_cache.borrow_mut().clear();
        }
    }

    /// Read and reset cache hit/miss counters. Returns (hits, misses, cache_size).
    pub fn get_and_reset_cache_stats(&self) -> (u64, u64, usize) {
        let hits = self.cache_hits.replace(0);
        let misses = self.cache_misses.replace(0);
        let size = self.board_cache.borrow().len();
        (hits, misses, size)
    }
}

impl Clone for TetrisNN {
    fn clone(&self) -> Self {
        // Share Arc-wrapped models/weights, create fresh empty cache and counters
        TetrisNN {
            conv_model: Arc::clone(&self.conv_model),
            heads_model: Arc::clone(&self.heads_model),
            board_proj_weight: Arc::clone(&self.board_proj_weight),
            board_proj_bias: Arc::clone(&self.board_proj_bias),
            board_hidden: self.board_hidden,
            conv_out_size: self.conv_out_size,
            board_cache: RefCell::new(HashMap::new()),
            cache_hits: Cell::new(0),
            cache_misses: Cell::new(0),
            cache_enabled: Cell::new(self.cache_enabled.get()),
        }
    }
}

/// Pack 200 binary board cells into [u64; 4] for use as a hash key.
/// Zero-collision: each unique board maps to a unique key.
fn pack_board(env: &TetrisEnv) -> [u64; 4] {
    let cells = env.board_cells();
    let mut key = [0u64; 4];
    for (i, &cell) in cells.iter().enumerate() {
        if cell != 0 {
            key[i / 64] |= 1u64 << (i % 64);
        }
    }
    key
}

/// Load board-projection weight matrix and bias from binary file.
/// Format: [rows u32 LE][cols u32 LE][weight row-major f32][bias f32]
fn load_fc_binary(path: &Path) -> TractResult<(Array2<f32>, Array1<f32>)> {
    let mut file = File::open(path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Board projection binary not found at {}: {}",
                path.display(),
                e
            ),
        )
    })?;

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let rows = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let cols = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let weight_bytes = rows * cols * 4;
    let bias_bytes = rows * 4;
    let mut data = vec![0u8; weight_bytes + bias_bytes];
    file.read_exact(&mut data)?;

    let weight_f32: Vec<f32> = data[..weight_bytes]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let bias_f32: Vec<f32> = data[weight_bytes..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let weight = Array2::from_shape_vec((rows, cols), weight_f32).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Board projection weight shape mismatch: {}", e),
        )
    })?;

    let bias = Array1::from_vec(bias_f32);

    Ok((weight, bias))
}

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

pub fn encode_state_features(
    env: &TetrisEnv,
    placement_count: usize,
    max_placements: usize,
) -> TractResult<(Vec<f32>, Vec<f32>)> {
    if max_placements == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "max_placements must be > 0",
        )
        .into());
    }

    // Board tensor: binary (1 = filled, 0 = empty) - 200 values (will be reshaped to 1x20x10)
    let board_tensor: Vec<f32> = env
        .board_cells()
        .iter()
        .map(|&cell| if cell != 0 { 1.0 } else { 0.0 })
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

    // Placement count: normalized (1)
    let normalized_denominator = max_placements as f32;
    aux.push(placement_count as f32 / normalized_denominator);

    // Combo: normalized (1)
    aux.push(normalize_combo_for_feature(env.combo));

    // Back-to-back flag: binary (1)
    aux.push(if env.back_to_back { 1.0 } else { 0.0 });

    // Next hidden piece distribution from 7-bag (7)
    let hidden_piece_distribution = next_hidden_piece_distribution(env);
    aux.extend(hidden_piece_distribution);

    Ok((board_tensor, aux))
}

pub fn normalize_combo_for_feature(combo: u32) -> f32 {
    let capped = combo.min(COMBO_NORMALIZATION_MAX) as f32;
    capped / COMBO_NORMALIZATION_MAX as f32
}

/// Probability distribution over the next hidden queue piece implied by the 7-bag state.
pub fn next_hidden_piece_distribution(env: &TetrisEnv) -> Vec<f32> {
    let possible_pieces = env.get_possible_next_pieces();
    if possible_pieces.is_empty() {
        panic!("Possible hidden-piece set must never be empty");
    }

    let probability = 1.0 / possible_pieces.len() as f32;
    let mut distribution = vec![0.0; NUM_PIECE_TYPES];
    for piece in possible_pieces {
        if piece >= NUM_PIECE_TYPES {
            panic!("Invalid piece type {} in possible next pieces", piece);
        }
        distribution[piece] = probability;
    }
    distribution
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

        let (board_tensor, _) = encode_state(&env, 0, 100).expect("encoding failed");
        let board_array = board_tensor
            .to_array_view::<f32>()
            .expect("board tensor should contain f32");
        let board_values: Vec<f32> = board_array.iter().copied().collect();

        // Verify size
        assert_eq!(board_values.len(), BOARD_HEIGHT * BOARD_WIDTH);

        // Verify all values are 0.0 or 1.0
        for &val in &board_values {
            assert!(
                val == 0.0 || val == 1.0,
                "Board tensor should be binary, got value: {}",
                val
            );
        }

        // Verify encoding matches board state
        let board = env.board_cells();
        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                let idx = y * BOARD_WIDTH + x;
                let expected = if board[idx] != 0 { 1.0 } else { 0.0 };
                let actual = board_values[idx];
                assert_eq!(
                    actual, expected,
                    "Board[{},{}] with value {} should encode to {}, got {}",
                    y, x, board[idx], expected, actual
                );
            }
        }
    }

    #[test]
    fn test_auxiliary_features_format() {
        let env = TetrisEnv::new(10, 20);
        let (_, aux) = encode_state(&env, 42, 100).expect("encoding failed");
        let aux_array = aux
            .to_array_view::<f32>()
            .expect("aux tensor should contain f32");
        let aux: Vec<f32> = aux_array.iter().copied().collect();

        // Total size: 7 + 8 + 1 + 35 + 1 + 1 + 1 + 7 = 61
        assert_eq!(aux.len(), AUX_FEATURES);
        assert_eq!(aux.len(), 61);

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

        // Placement count: normalized (1)
        let move_norm = aux[idx];
        let expected_norm = 42.0 / 100.0;
        assert!(
            (move_norm - expected_norm).abs() < 1e-6,
            "Placement count feature should be {}, got {}",
            expected_norm,
            move_norm
        );
        idx += 1;

        // Combo: normalized (1)
        let combo_norm = aux[idx];
        let expected_combo_norm = normalize_combo_for_feature(env.combo);
        assert!(
            (combo_norm - expected_combo_norm).abs() < 1e-6,
            "Combo feature should be {}, got {}",
            expected_combo_norm,
            combo_norm
        );
        idx += 1;

        // Back-to-back flag: binary (1)
        let back_to_back = aux[idx];
        let expected_back_to_back = if env.back_to_back { 1.0 } else { 0.0 };
        assert_eq!(
            back_to_back, expected_back_to_back,
            "Back-to-back feature incorrect"
        );
        idx += 1;

        // Next hidden piece distribution (7)
        let hidden_distribution = &aux[idx..idx + NUM_PIECE_TYPES];
        let expected_hidden_distribution = next_hidden_piece_distribution(&env);
        let hidden_sum: f32 = hidden_distribution.iter().sum();
        assert!(
            (hidden_sum - 1.0).abs() < 1e-6,
            "Hidden-piece distribution should sum to 1.0, got {}",
            hidden_sum
        );
        for piece in 0..NUM_PIECE_TYPES {
            assert!(
                (hidden_distribution[piece] - expected_hidden_distribution[piece]).abs() < 1e-6,
                "Hidden-piece probability mismatch for piece {}: expected {}, got {}",
                piece,
                expected_hidden_distribution[piece],
                hidden_distribution[piece]
            );
        }
        idx += NUM_PIECE_TYPES;

        assert_eq!(
            idx, AUX_FEATURES,
            "Should consume all {} aux features",
            AUX_FEATURES
        );
    }

    #[test]
    fn test_encoding_specification() {
        // Verify the exact specification from AGENTS.md:
        // | Board state    | 20 x 10    | Binary (1 = filled, 0 = empty)  |
        // | Current piece  | 7          | One-hot encoded                 |
        // | Hold piece     | 8          | One-hot (7 pieces + empty)      |
        // | Hold available | 1          | Binary (can use hold this turn) |
        // | Next queue     | 5 x 7 = 35 | One-hot encoded per slot        |
        // | Placement count| 1          | Normalized: placements / 100     |
        // | Combo          | 1          | Normalized combo                 |
        // | Back-to-back   | 1          | Binary                           |
        // | Hidden piece   | 7          | 7-bag distribution               |

        let env = TetrisEnv::new(10, 20);
        let (board, aux) = encode_state(&env, 50, 100).expect("encoding failed");
        let board_array = board
            .to_array_view::<f32>()
            .expect("board tensor should contain f32");
        let aux_array = aux
            .to_array_view::<f32>()
            .expect("aux tensor should contain f32");
        let board: Vec<f32> = board_array.iter().copied().collect();
        let aux: Vec<f32> = aux_array.iter().copied().collect();

        assert_eq!(board.len(), 20 * 10, "Board should be 20x10 = 200 values");
        assert_eq!(
            aux.len(),
            7 + 8 + 1 + 35 + 1 + 1 + 1 + 7,
            "Aux should be 7+8+1+35+1+1+1+7 = 61 values"
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

        // Verify placement count is normalized [0, 1]
        let move_norm = aux[51];
        assert!(
            move_norm >= 0.0 && move_norm <= 1.0,
            "Placement count must be in [0, 1]"
        );
        assert!(
            (move_norm - 0.5).abs() < 1e-6,
            "Placement count 50 should normalize to 0.5"
        );

        // Verify combo is normalized [0, 1]
        let combo_norm = aux[52];
        assert!(
            combo_norm >= 0.0 && combo_norm <= 1.0,
            "Combo must be in [0, 1]"
        );
        assert_eq!(combo_norm, 0.0, "Initial combo should be 0.0");

        // Verify back-to-back is binary
        let back_to_back = aux[53];
        assert!(
            back_to_back == 0.0 || back_to_back == 1.0,
            "Back-to-back must be binary"
        );
        assert_eq!(back_to_back, 0.0, "Initial back-to-back should be 0.0");

        // Verify hidden-piece distribution is valid probabilities
        let hidden_distribution = &aux[54..61];
        let hidden_sum: f32 = hidden_distribution.iter().sum();
        assert!(
            (hidden_sum - 1.0).abs() < 1e-6,
            "Hidden-piece distribution must sum to 1.0"
        );
        for &p in hidden_distribution {
            assert!(
                (0.0..=1.0).contains(&p),
                "Hidden-piece probabilities must be in [0, 1]"
            );
        }
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

    #[test]
    fn test_pack_board_empty() {
        let env = TetrisEnv::new(10, 20);
        // Empty board should pack to all zeros (new board has no placed blocks)
        // Note: env may have a current piece but board_cells only contains placed blocks
        let key = pack_board(&env);
        assert_eq!(key, [0u64; 4], "Empty board should pack to all zeros");
    }

    #[test]
    fn test_pack_board_deterministic() {
        let mut env = TetrisEnv::new(10, 20);
        env.hard_drop();
        let key1 = pack_board(&env);
        let key2 = pack_board(&env);
        assert_eq!(key1, key2, "Same board should produce same key");
    }

    #[test]
    fn test_pack_board_different_boards() {
        let mut env1 = TetrisEnv::new(10, 20);
        env1.hard_drop();
        let key1 = pack_board(&env1);

        let mut env2 = TetrisEnv::new(10, 20);
        env2.hard_drop();
        env2.hard_drop();
        let key2 = pack_board(&env2);

        assert_ne!(key1, key2, "Different boards should produce different keys");
    }

    #[test]
    fn test_load_and_predict_split_model() {
        let model_path = "/tmp/tetris_split_test/test.onnx";
        if !std::path::Path::new("/tmp/tetris_split_test/test.conv.onnx").exists() {
            eprintln!("Skipping test - split model files not found (run Python export first)");
            return;
        }

        let nn = TetrisNN::load(model_path).expect("Failed to load split model");
        let env = TetrisEnv::new(10, 20);
        let mask = get_action_mask(&env);

        // First call - cache miss
        let (policy1, value1) = nn
            .predict_masked(&env, 0, &mask, 100)
            .expect("First inference failed");
        assert_eq!(policy1.len(), mask.len());
        let policy_sum: f32 = policy1.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 1e-4,
            "Policy should sum to ~1.0, got {}",
            policy_sum
        );

        // Second call with same board - cache hit, should produce identical results
        let (policy2, value2) = nn
            .predict_masked(&env, 0, &mask, 100)
            .expect("Second inference (cache hit) failed");
        assert_eq!(policy1, policy2, "Cache hit should produce same policy");
        assert_eq!(value1, value2, "Cache hit should produce same value");

        // Different aux (different placement count) with same board - different output
        let (_policy3, value3) = nn
            .predict_masked(&env, 50, &mask, 100)
            .expect("Third inference (different aux) failed");
        // Values should differ because aux features changed
        assert_ne!(
            value1, value3,
            "Different placement counts should produce different values"
        );

        // Verify cache has exactly 1 entry (same board both times)
        assert_eq!(
            nn.board_cache.borrow().len(),
            1,
            "Cache should have 1 entry for the single board state"
        );
    }
}
