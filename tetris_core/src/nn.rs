//! Neural Network Inference using tract-onnx
//!
//! Loads split ONNX models (conv backbone + heads) and FC weights from binary.
//! Caches board embeddings to skip conv + board FC on repeated board states.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use tract_onnx::prelude::*;

use crate::constants::{AUX_FEATURES, BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, QUEUE_SIZE};
use crate::env::TetrisEnv;

/// Number of conv output features (BOARD_HEIGHT * BOARD_WIDTH * conv_filters[-1])
const CONV_OUT_SIZE: usize = BOARD_HEIGHT * BOARD_WIDTH * 8; // 1600

/// Neural network model wrapper with board embedding cache
pub struct TetrisNN {
    conv_model: Arc<TypedRunnableModel<TypedModel>>,
    heads_model: Arc<TypedRunnableModel<TypedModel>>,
    fc_weight_board: Arc<Array2<f32>>, // (fc_hidden, CONV_OUT_SIZE)
    fc_weight_aux: Arc<Array2<f32>>,   // (fc_hidden, AUX_FEATURES)
    fc_bias: Arc<Array1<f32>>,         // (fc_hidden,)
    fc_hidden: usize,
    board_cache: RefCell<HashMap<[u64; 4], Array1<f32>>>,
}

impl TetrisNN {
    /// Load split models from file.
    /// Given a base path like "latest.onnx", loads:
    /// - "latest.conv.onnx" (conv backbone)
    /// - "latest.heads.onnx" (heads model)
    /// - "latest.fc.bin" (FC weight + bias)
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

        let (fc_weight, fc_bias) = load_fc_binary(&fc_path)?;
        let fc_hidden = fc_weight.nrows();
        let total_cols = fc_weight.ncols();

        // Split FC weight into board part and aux part
        let fc_weight_board = fc_weight.slice(ndarray::s![.., ..CONV_OUT_SIZE]).to_owned();
        let fc_weight_aux = fc_weight
            .slice(ndarray::s![.., CONV_OUT_SIZE..total_cols])
            .to_owned();

        debug_assert_eq!(fc_weight_board.ncols(), CONV_OUT_SIZE);
        debug_assert_eq!(fc_weight_aux.ncols(), AUX_FEATURES);

        Ok(TetrisNN {
            conv_model: Arc::new(conv_model),
            heads_model: Arc::new(heads_model),
            fc_weight_board: Arc::new(fc_weight_board),
            fc_weight_aux: Arc::new(fc_weight_aux),
            fc_bias: Arc::new(fc_bias),
            fc_hidden,
            board_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Run inference with action mask applied, using board embedding cache.
    pub fn predict_masked(
        &self,
        env: &TetrisEnv,
        move_number: usize,
        action_mask: &[bool],
        max_moves: usize,
    ) -> TractResult<(Vec<f32>, f32)> {
        let (board_f32, aux_vec) = encode_state_features(env, move_number, max_moves)?;

        // Pack board into cache key
        let board_key = pack_board(env);

        // Check cache for board embedding
        let board_embed = {
            let cache = self.board_cache.borrow();
            cache.get(&board_key).cloned()
        };

        let board_embed = match board_embed {
            Some(embed) => embed,
            None => {
                // Cache miss: run conv model
                let board_tensor = tract_ndarray::Array4::from_shape_vec(
                    (1, 1, BOARD_HEIGHT, BOARD_WIDTH),
                    board_f32,
                )?
                .into_tensor();

                let conv_output = self.conv_model.run(tvec!(board_tensor.into()))?;
                let conv_out: Vec<f32> = conv_output[0]
                    .to_array_view::<f32>()?
                    .iter()
                    .copied()
                    .collect();

                let conv_arr = Array1::from_vec(conv_out);

                // board_embed = W_board @ conv_out + bias
                let embed = self.fc_weight_board.dot(&conv_arr) + self.fc_bias.as_ref();

                self.board_cache.borrow_mut().insert(board_key, embed.clone());
                embed
            }
        };

        // fc_out = board_embed + W_aux @ aux
        let aux_arr = Array1::from_vec(aux_vec);
        let fc_out = &board_embed + &self.fc_weight_aux.dot(&aux_arr);

        // Run heads model
        let fc_pre_tensor = tract_ndarray::Array2::from_shape_vec(
            (1, self.fc_hidden),
            fc_out.to_vec(),
        )?
        .into_tensor();

        let heads_output = self.heads_model.run(tvec!(fc_pre_tensor.into()))?;

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
        let board = tract_ndarray::Array4::from_shape_vec(
            (1, 1, BOARD_HEIGHT, BOARD_WIDTH),
            board_tensor.to_vec(),
        )?
        .into_tensor();

        let conv_output = self.conv_model.run(tvec!(board.into()))?;
        let conv_out: Vec<f32> = conv_output[0]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();

        let conv_arr = Array1::from_vec(conv_out);
        let aux_arr = Array1::from_vec(aux_tensor.to_vec());

        let fc_out = self.fc_weight_board.dot(&conv_arr)
            + &self.fc_weight_aux.dot(&aux_arr)
            + self.fc_bias.as_ref();

        let fc_pre_tensor = tract_ndarray::Array2::from_shape_vec(
            (1, self.fc_hidden),
            fc_out.to_vec(),
        )?
        .into_tensor();

        let heads_output = self.heads_model.run(tvec!(fc_pre_tensor.into()))?;

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

impl Clone for TetrisNN {
    fn clone(&self) -> Self {
        // Share Arc-wrapped models/weights, create fresh empty cache
        TetrisNN {
            conv_model: Arc::clone(&self.conv_model),
            heads_model: Arc::clone(&self.heads_model),
            fc_weight_board: Arc::clone(&self.fc_weight_board),
            fc_weight_aux: Arc::clone(&self.fc_weight_aux),
            fc_bias: Arc::clone(&self.fc_bias),
            fc_hidden: self.fc_hidden,
            board_cache: RefCell::new(HashMap::new()),
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

/// Load FC weight matrix and bias from binary file.
/// Format: [rows u32 LE][cols u32 LE][weight row-major f32][bias f32]
fn load_fc_binary(path: &Path) -> TractResult<(Array2<f32>, Array1<f32>)> {
    let mut file = File::open(path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("FC binary not found at {}: {}", path.display(), e),
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
            format!("FC weight shape mismatch: {}", e),
        )
    })?;

    let bias = Array1::from_vec(bias_f32);

    Ok((weight, bias))
}

/// Encode a TetrisEnv state into neural network input tensors
#[cfg(test)]
fn encode_state(
    env: &TetrisEnv,
    move_number: usize,
    max_moves: usize,
) -> TractResult<(Tensor, Tensor)> {
    let (board_tensor, aux_tensor) = encode_state_features(env, move_number, max_moves)?;
    let board =
        tract_ndarray::Array4::from_shape_vec((1, 1, BOARD_HEIGHT, BOARD_WIDTH), board_tensor)?
            .into_tensor();
    let aux = tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor)?.into_tensor();
    Ok((board, aux))
}

pub fn encode_state_features(
    env: &TetrisEnv,
    move_number: usize,
    max_moves: usize,
) -> TractResult<(Vec<f32>, Vec<f32>)> {
    if max_moves == 0 {
        return Err(
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "max_moves must be > 0").into(),
        );
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

    // Move number: normalized (1)
    let normalized_denominator = max_moves as f32;
    aux.push(move_number as f32 / normalized_denominator);

    Ok((board_tensor, aux))
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

        // Different aux (different move_number) with same board - different output
        let (_policy3, value3) = nn
            .predict_masked(&env, 50, &mask, 100)
            .expect("Third inference (different aux) failed");
        // Values should differ because aux features changed
        assert_ne!(
            value1, value3,
            "Different move numbers should produce different values"
        );

        // Verify cache has exactly 1 entry (same board both times)
        assert_eq!(
            nn.board_cache.borrow().len(),
            1,
            "Cache should have 1 entry for the single board state"
        );
    }
}
