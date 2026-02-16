//! Neural Network Inference using tract-onnx
//!
//! Loads split ONNX models (conv backbone + heads) and board projection weights from binary.
//! Caches board embeddings to skip conv + board projection on repeated board states.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1};
use tract_onnx::prelude::*;

use crate::constants::{
    AUX_FEATURES, BOARD_CACHE_MAX_ENTRIES, BOARD_HEIGHT, BOARD_STATS_FEATURES, BOARD_WIDTH,
    COMBO_NORMALIZATION_MAX, NUM_PIECE_TYPES, PIECE_AUX_FEATURES, QUEUE_SIZE,
};
use crate::env::TetrisEnv;
use crate::mcts::{
    compute_bumpiness, count_overhang_fields_and_holes, normalize_bumpiness,
    normalize_column_heights, normalize_holes, normalize_overhang_fields,
    normalize_row_fill_counts, normalize_total_blocks,
};

/// Neural network model wrapper with board embedding cache
pub struct TetrisNN {
    conv_model: Arc<TypedRunnableModel<TypedModel>>,
    heads_model: Arc<TypedRunnableModel<TypedModel>>,
    board_proj_weight: Arc<Array2<f32>>, // (board_hidden, conv_out_size + BOARD_STATS_FEATURES)
    board_proj_bias: Arc<Array1<f32>>,   // (board_hidden,)
    board_hidden: usize,
    conv_out_size: usize, // conv model output dim (e.g. 1600)
    board_cache: RefCell<HashMap<[u64; 4], Array1<f32>>>,
    cache_order: RefCell<VecDeque<[u64; 4]>>,
    cache_capacity: usize,
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
        let board_proj_cols = board_proj_weight.ncols();
        if board_proj_cols <= BOARD_STATS_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Board projection width {} must be > BOARD_STATS_FEATURES ({})",
                    board_proj_cols, BOARD_STATS_FEATURES
                ),
            )
            .into());
        }
        let conv_out_size = board_proj_cols - BOARD_STATS_FEATURES;

        Ok(TetrisNN {
            conv_model: Arc::new(conv_model),
            heads_model: Arc::new(heads_model),
            board_proj_weight: Arc::new(board_proj_weight),
            board_proj_bias: Arc::new(board_proj_bias),
            board_hidden,
            conv_out_size,
            board_cache: RefCell::new(HashMap::new()),
            cache_order: RefCell::new(VecDeque::new()),
            cache_capacity: BOARD_CACHE_MAX_ENTRIES,
            cache_hits: Cell::new(0),
            cache_misses: Cell::new(0),
            cache_enabled: Cell::new(true),
        })
    }

    fn compute_board_embedding_owned(
        &self,
        board_f32: Vec<f32>,
        board_stats: &[f32],
    ) -> TractResult<Array1<f32>> {
        let board_tensor =
            tract_ndarray::Array4::from_shape_vec((1, 1, BOARD_HEIGHT, BOARD_WIDTH), board_f32)?
                .into_tensor();

        let conv_output = self.conv_model.run(tvec!(board_tensor.into()))?;
        let conv_out = conv_output[0].to_array_view::<f32>()?;
        if conv_out.len() != self.conv_out_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Conv output size mismatch: conv.onnx={} fc.bin expects conv_out={} (re-export split models together)",
                    conv_out.len(),
                    self.conv_out_size
                ),
            )
            .into());
        }
        let conv_slice = conv_out.as_slice_memory_order().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Conv output tensor is not contiguous",
            )
        })?;

        // Concatenate conv output (e.g. 1600) + board_stats (36) → board_proj input
        let mut combined = Vec::with_capacity(self.conv_out_size + BOARD_STATS_FEATURES);
        combined.extend_from_slice(conv_slice);
        combined.extend_from_slice(board_stats);

        let combined_arr = ArrayView1::from(&combined);
        let mut board_embed = self.board_proj_weight.dot(&combined_arr);
        board_embed += self.board_proj_bias.as_ref();
        Ok(board_embed)
    }

    fn compute_board_embedding_from_slice(
        &self,
        board_f32: &[f32],
        board_stats: &[f32],
    ) -> TractResult<Array1<f32>> {
        self.compute_board_embedding_owned(board_f32.to_vec(), board_stats)
    }

    fn insert_board_embedding_cache(&self, board_key: [u64; 4], embed: Array1<f32>) {
        let mut cache = self.board_cache.borrow_mut();
        let mut cache_order = self.cache_order.borrow_mut();

        if cache.insert(board_key, embed).is_none() {
            cache_order.push_back(board_key);
        }

        while cache.len() > self.cache_capacity {
            let Some(oldest_key) = cache_order.pop_front() else {
                break;
            };
            cache.remove(&oldest_key);
        }
    }

    fn clear_board_embedding_cache(&self) {
        self.board_cache.borrow_mut().clear();
        self.cache_order.borrow_mut().clear();
    }

    fn predict_policy_logits_tensor_and_value(
        &self,
        env: &TetrisEnv,
        placement_count: usize,
        max_placements: usize,
    ) -> TractResult<(Tensor, f32)> {
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
                    let board_f32 = encode_board_features(env);
                    let board_stats = encode_board_stats(env);
                    let embed = self.compute_board_embedding_owned(board_f32, &board_stats)?;
                    self.insert_board_embedding_cache(board_key, embed.clone());
                    embed
                }
            }
        } else {
            let board_f32 = encode_board_features(env);
            let board_stats = encode_board_stats(env);
            self.compute_board_embedding_owned(board_f32, &board_stats)?
        };

        // Encode only piece/game aux features (61-dim) for the heads model
        let piece_aux_vec = encode_piece_aux_features(env, placement_count, max_placements)?;

        let board_h_tensor = tract_ndarray::Array2::from_shape_vec(
            (1, self.board_hidden),
            board_embed.into_raw_vec(),
        )?
        .into_tensor();
        let aux_tensor =
            tract_ndarray::Array2::from_shape_vec((1, PIECE_AUX_FEATURES), piece_aux_vec)?
                .into_tensor();
        let heads_output = self
            .heads_model
            .run(tvec!(board_h_tensor.into(), aux_tensor.into()))?;

        let policy_logits = heads_output[0].clone().into_tensor();

        let value = heads_output[1]
            .to_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .expect("NN value output tensor is empty - model is malformed");

        Ok((policy_logits, value))
    }

    fn predict_logits_and_value(
        &self,
        env: &TetrisEnv,
        placement_count: usize,
        max_placements: usize,
    ) -> TractResult<(Vec<f32>, f32)> {
        let (policy_logits_tensor, value) =
            self.predict_policy_logits_tensor_and_value(env, placement_count, max_placements)?;
        let policy_logits: Vec<f32> = policy_logits_tensor
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .collect();
        Ok((policy_logits, value))
    }

    /// Run inference with action mask applied, using board embedding cache.
    pub fn predict_masked(
        &self,
        env: &TetrisEnv,
        placement_count: usize,
        action_mask: &[bool],
        max_placements: usize,
    ) -> TractResult<(Vec<f32>, f32)> {
        let (policy_logits, value) =
            self.predict_logits_and_value(env, placement_count, max_placements)?;

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

        let policy = masked_softmax(&policy_logits, action_mask);

        Ok((policy, value))
    }

    /// Run inference with precomputed valid action indices.
    pub fn predict_with_valid_actions(
        &self,
        env: &TetrisEnv,
        placement_count: usize,
        valid_actions: &[usize],
        max_placements: usize,
    ) -> TractResult<(Vec<f32>, f32)> {
        let (policy_logits_tensor, value) =
            self.predict_policy_logits_tensor_and_value(env, placement_count, max_placements)?;
        let policy_logits_view = policy_logits_tensor.to_array_view::<f32>()?;
        let policy_logits = policy_logits_view.as_slice_memory_order().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "NN policy output tensor is not contiguous",
            )
        })?;

        if let Some(&invalid_action) = valid_actions
            .iter()
            .find(|&&action_idx| action_idx >= policy_logits.len())
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Valid action index {} out of range for policy logits len {}",
                    invalid_action,
                    policy_logits.len()
                ),
            )
            .into());
        }

        let action_priors = softmax_over_valid_actions(policy_logits, valid_actions);
        Ok((action_priors, value))
    }

    pub fn predict_masked_from_tensors(
        &self,
        board_tensor: &[f32],
        aux_tensor: &[f32],
        action_mask: &[bool],
    ) -> TractResult<(Vec<f32>, f32)> {
        // No caching for raw tensor inputs (used only by debug functions).
        // aux_tensor is the full 97-dim vector; split into piece_aux (61) + board_stats (36).
        let piece_aux = &aux_tensor[..PIECE_AUX_FEATURES];
        let board_stats = &aux_tensor[PIECE_AUX_FEATURES..];
        let board_embed = self.compute_board_embedding_from_slice(board_tensor, board_stats)?;
        let board_h_tensor = tract_ndarray::Array2::from_shape_vec(
            (1, self.board_hidden),
            board_embed.into_raw_vec(),
        )?
        .into_tensor();
        let aux_tensor =
            tract_ndarray::Array2::from_shape_vec((1, PIECE_AUX_FEATURES), piece_aux.to_vec())?
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
            self.clear_board_embedding_cache();
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
            cache_order: RefCell::new(VecDeque::new()),
            cache_capacity: self.cache_capacity,
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
    let aux = encode_aux_state_features(env, placement_count, max_placements)?;
    let board_tensor = encode_board_features(env);
    Ok((board_tensor, aux))
}

fn encode_board_features(env: &TetrisEnv) -> Vec<f32> {
    // Board tensor: binary (1 = filled, 0 = empty) - 200 values (will be reshaped to 1x20x10)
    env.board_cells()
        .iter()
        .map(|&cell| if cell != 0 { 1.0 } else { 0.0 })
        .collect()
}

/// Encode the 36 board-derived statistics for the cached board embedding path.
fn encode_board_stats(env: &TetrisEnv) -> Vec<f32> {
    let normalized_column_heights = normalize_column_heights(&env.column_heights, env.height);
    let max_column_height = normalized_column_heights
        .iter()
        .copied()
        .reduce(f32::max)
        .unwrap_or(0.0);
    let min_column_height = normalized_column_heights
        .iter()
        .copied()
        .reduce(f32::min)
        .unwrap_or(0.0);
    let normalized_row_fill_counts = normalize_row_fill_counts(&env.row_fill_counts, env.width);
    let normalized_total_blocks = normalize_total_blocks(env.total_blocks, env.width, env.height);
    let raw_bumpiness = compute_bumpiness(&env.column_heights);
    let normalized_bumpiness = normalize_bumpiness(raw_bumpiness, env.width, env.height);
    let (raw_overhang_fields, raw_holes) = count_overhang_fields_and_holes(env);
    let normalized_holes = normalize_holes(raw_holes, env.width, env.height);
    let normalized_overhang_fields = normalize_overhang_fields(raw_overhang_fields);

    let mut stats = Vec::with_capacity(BOARD_STATS_FEATURES);
    stats.extend_from_slice(&normalized_column_heights);
    stats.push(max_column_height);
    stats.push(min_column_height);
    stats.extend_from_slice(&normalized_row_fill_counts);
    stats.push(normalized_total_blocks);
    stats.push(normalized_bumpiness);
    stats.push(normalized_holes);
    stats.push(normalized_overhang_fields);
    debug_assert_eq!(stats.len(), BOARD_STATS_FEATURES);
    stats
}

/// Encode only the 61 piece/game auxiliary features for the uncached heads model.
fn encode_piece_aux_features(
    env: &TetrisEnv,
    placement_count: usize,
    max_placements: usize,
) -> TractResult<Vec<f32>> {
    if max_placements == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "max_placements must be > 0",
        )
        .into());
    }

    let current_piece = env.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
    let hold_piece = env.get_hold_piece().map(|p| p.piece_type);
    let queue = env.get_queue(QUEUE_SIZE);
    let hidden_piece_distribution = next_hidden_piece_distribution(env);

    let mut aux = vec![0.0; PIECE_AUX_FEATURES];
    let mut aux_idx = 0;

    // Current piece: one-hot (7)
    aux[aux_idx + current_piece] = 1.0;
    aux_idx += NUM_PIECE_TYPES;

    // Hold piece: one-hot (8) - 7 pieces + empty
    if let Some(piece_type) = hold_piece {
        aux[aux_idx + piece_type] = 1.0;
    } else {
        aux[aux_idx + NUM_PIECE_TYPES] = 1.0;
    }
    aux_idx += NUM_PIECE_TYPES + 1;

    // Hold available: binary (1)
    aux[aux_idx] = if !env.is_hold_used() { 1.0 } else { 0.0 };
    aux_idx += 1;

    // Next queue: one-hot per slot (5 x 7 = 35)
    for (slot, &piece_type) in queue.iter().take(QUEUE_SIZE).enumerate() {
        aux[aux_idx + slot * NUM_PIECE_TYPES + piece_type] = 1.0;
    }
    aux_idx += QUEUE_SIZE * NUM_PIECE_TYPES;

    // Placement count: normalized (1)
    aux[aux_idx] = placement_count as f32 / max_placements as f32;
    aux_idx += 1;

    // Combo: normalized (1)
    aux[aux_idx] = normalize_combo_for_feature(env.combo);
    aux_idx += 1;

    // Back-to-back flag: binary (1)
    aux[aux_idx] = if env.back_to_back { 1.0 } else { 0.0 };
    aux_idx += 1;

    // Next hidden piece distribution from 7-bag (7)
    aux[aux_idx..aux_idx + NUM_PIECE_TYPES].copy_from_slice(&hidden_piece_distribution);
    aux_idx += NUM_PIECE_TYPES;

    debug_assert_eq!(aux_idx, PIECE_AUX_FEATURES);
    Ok(aux)
}

pub fn encode_aux_state_features(
    env: &TetrisEnv,
    placement_count: usize,
    max_placements: usize,
) -> TractResult<Vec<f32>> {
    if max_placements == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "max_placements must be > 0",
        )
        .into());
    }

    let current_piece = env.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
    let hold_piece = env.get_hold_piece().map(|p| p.piece_type);
    let queue = env.get_queue(QUEUE_SIZE);
    let hidden_piece_distribution = next_hidden_piece_distribution(env);
    let normalized_column_heights = normalize_column_heights(&env.column_heights, env.height);
    let max_column_height = normalized_column_heights
        .iter()
        .copied()
        .reduce(f32::max)
        .unwrap_or(0.0);
    let min_column_height = normalized_column_heights
        .iter()
        .copied()
        .reduce(f32::min)
        .unwrap_or(0.0);
    let normalized_row_fill_counts = normalize_row_fill_counts(&env.row_fill_counts, env.width);
    let normalized_total_blocks = normalize_total_blocks(env.total_blocks, env.width, env.height);
    let raw_bumpiness = compute_bumpiness(&env.column_heights);
    let normalized_bumpiness = normalize_bumpiness(raw_bumpiness, env.width, env.height);
    let (raw_overhang_fields, raw_holes) = count_overhang_fields_and_holes(env);
    let normalized_holes = normalize_holes(raw_holes, env.width, env.height);
    let normalized_overhang_fields = normalize_overhang_fields(raw_overhang_fields);
    let mut aux = vec![0.0; AUX_FEATURES];
    encode_aux_features(
        &mut aux,
        current_piece,
        hold_piece,
        !env.is_hold_used(),
        &queue,
        placement_count,
        max_placements,
        env.combo,
        env.back_to_back,
        &hidden_piece_distribution,
        &normalized_column_heights,
        max_column_height,
        min_column_height,
        &normalized_row_fill_counts,
        normalized_total_blocks,
        normalized_bumpiness,
        normalized_holes,
        normalized_overhang_fields,
    )?;

    Ok(aux)
}

pub fn normalize_combo_for_feature(combo: u32) -> f32 {
    let capped = combo.min(COMBO_NORMALIZATION_MAX) as f32;
    capped / COMBO_NORMALIZATION_MAX as f32
}

pub fn denormalize_combo_feature(combo_feature: f32) -> u32 {
    let clamped = combo_feature.clamp(0.0, 1.0);
    (clamped * COMBO_NORMALIZATION_MAX as f32).round() as u32
}

pub fn encode_aux_features(
    aux_out: &mut [f32],
    current_piece: usize,
    hold_piece: Option<usize>,
    hold_available: bool,
    next_queue: &[usize],
    placement_count: usize,
    max_placements: usize,
    combo: u32,
    back_to_back: bool,
    next_hidden_piece_probs: &[f32],
    column_heights: &[f32],
    max_column_height: f32,
    min_column_height: f32,
    row_fill_counts: &[f32],
    total_blocks: f32,
    bumpiness: f32,
    holes: f32,
    overhang_fields: f32,
) -> TractResult<()> {
    if max_placements == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "max_placements must be > 0",
        )
        .into());
    }
    if aux_out.len() != AUX_FEATURES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "aux_out length must be {} (got {})",
                AUX_FEATURES,
                aux_out.len()
            ),
        )
        .into());
    }
    if current_piece >= NUM_PIECE_TYPES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "current_piece must be in 0..{} (got {})",
                NUM_PIECE_TYPES, current_piece
            ),
        )
        .into());
    }
    if let Some(piece_type) = hold_piece {
        if piece_type >= NUM_PIECE_TYPES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "hold_piece must be in 0..{} when present (got {})",
                    NUM_PIECE_TYPES, piece_type
                ),
            )
            .into());
        }
    }
    if next_hidden_piece_probs.len() != NUM_PIECE_TYPES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "next_hidden_piece_probs length must be {} (got {})",
                NUM_PIECE_TYPES,
                next_hidden_piece_probs.len()
            ),
        )
        .into());
    }
    if column_heights.len() != BOARD_WIDTH {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "column_heights length must be {} (got {})",
                BOARD_WIDTH,
                column_heights.len()
            ),
        )
        .into());
    }
    if row_fill_counts.len() != BOARD_HEIGHT {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "row_fill_counts length must be {} (got {})",
                BOARD_HEIGHT,
                row_fill_counts.len()
            ),
        )
        .into());
    }
    if let Some(&invalid_piece) = next_queue
        .iter()
        .take(QUEUE_SIZE)
        .find(|&&piece_type| piece_type >= NUM_PIECE_TYPES)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "next_queue contains invalid piece type {} (expected < {})",
                invalid_piece, NUM_PIECE_TYPES
            ),
        )
        .into());
    }

    aux_out.fill(0.0);
    let mut aux_idx = 0;

    // Current piece: one-hot (7)
    aux_out[aux_idx + current_piece] = 1.0;
    aux_idx += NUM_PIECE_TYPES;

    // Hold piece: one-hot (8) - 7 pieces + empty
    if let Some(piece_type) = hold_piece {
        aux_out[aux_idx + piece_type] = 1.0;
    } else {
        aux_out[aux_idx + NUM_PIECE_TYPES] = 1.0;
    }
    aux_idx += NUM_PIECE_TYPES + 1;

    // Hold available: binary (1)
    aux_out[aux_idx] = if hold_available { 1.0 } else { 0.0 };
    aux_idx += 1;

    // Next queue: one-hot per slot (5 x 7 = 35)
    for (slot, &piece_type) in next_queue.iter().take(QUEUE_SIZE).enumerate() {
        aux_out[aux_idx + slot * NUM_PIECE_TYPES + piece_type] = 1.0;
    }
    aux_idx += QUEUE_SIZE * NUM_PIECE_TYPES;

    // Placement count: normalized (1)
    aux_out[aux_idx] = placement_count as f32 / max_placements as f32;
    aux_idx += 1;

    // Combo: normalized (1)
    aux_out[aux_idx] = normalize_combo_for_feature(combo);
    aux_idx += 1;

    // Back-to-back flag: binary (1)
    aux_out[aux_idx] = if back_to_back { 1.0 } else { 0.0 };
    aux_idx += 1;

    // Next hidden piece distribution from 7-bag (7)
    aux_out[aux_idx..aux_idx + NUM_PIECE_TYPES].copy_from_slice(next_hidden_piece_probs);
    aux_idx += NUM_PIECE_TYPES;

    // Column heights (10)
    aux_out[aux_idx..aux_idx + BOARD_WIDTH].copy_from_slice(column_heights);
    aux_idx += BOARD_WIDTH;

    // Max column height (1)
    aux_out[aux_idx] = max_column_height;
    aux_idx += 1;

    // Min column height (1)
    aux_out[aux_idx] = min_column_height;
    aux_idx += 1;

    // Row fill counts (20)
    aux_out[aux_idx..aux_idx + BOARD_HEIGHT].copy_from_slice(row_fill_counts);
    aux_idx += BOARD_HEIGHT;

    // Total blocks (1)
    aux_out[aux_idx] = total_blocks;
    aux_idx += 1;

    // Bumpiness (1)
    aux_out[aux_idx] = bumpiness;
    aux_idx += 1;

    // Holes (1)
    aux_out[aux_idx] = holes;
    aux_idx += 1;

    // Overhang fields (1)
    aux_out[aux_idx] = overhang_fields;
    aux_idx += 1;

    debug_assert_eq!(aux_idx, AUX_FEATURES);
    Ok(())
}

/// Probability distribution over the next hidden queue piece implied by the 7-bag state.
pub fn next_hidden_piece_distribution(env: &TetrisEnv) -> Vec<f32> {
    let mut visible_state = env.clone();
    visible_state.truncate_queue(QUEUE_SIZE);
    let possible_pieces = visible_state.get_possible_next_pieces();
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

/// Softmax over precomputed valid action indices.
///
/// Returns priors aligned with `valid_actions` order.
pub fn softmax_over_valid_actions(logits: &[f32], valid_actions: &[usize]) -> Vec<f32> {
    let mut result = vec![0.0; valid_actions.len()];
    if valid_actions.is_empty() {
        return result;
    }

    let mut max_logit = f32::NEG_INFINITY;
    for &action_idx in valid_actions {
        max_logit = max_logit.max(logits[action_idx]);
    }

    let mut sum = 0.0;
    for (i, &action_idx) in valid_actions.iter().enumerate() {
        let exp_val = (logits[action_idx] - max_logit).exp();
        result[i] = exp_val;
        sum += exp_val;
    }

    if sum > 0.0 {
        for prior in &mut result {
            *prior /= sum;
        }
    }

    result
}

/// Get action mask from environment
pub fn get_action_mask(env: &TetrisEnv) -> Vec<bool> {
    env.get_cached_action_mask().as_ref().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

        // Total size: 7 + 8 + 1 + 35 + 1 + 1 + 1 + 7 + 10 + 1 + 1 + 20 + 1 + 1 + 1 + 1 = 97
        assert_eq!(aux.len(), AUX_FEATURES);
        assert_eq!(aux.len(), 97);

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

        let expected_column_heights = normalize_column_heights(&env.column_heights, env.height);
        let encoded_column_heights = &aux[idx..idx + BOARD_WIDTH];
        for col in 0..BOARD_WIDTH {
            assert!(
                (encoded_column_heights[col] - expected_column_heights[col]).abs() < 1e-6,
                "Column height mismatch at col {}: expected {}, got {}",
                col,
                expected_column_heights[col],
                encoded_column_heights[col]
            );
        }
        idx += BOARD_WIDTH;

        let expected_max_column_height = expected_column_heights
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(0.0);
        assert!(
            (aux[idx] - expected_max_column_height).abs() < 1e-6,
            "Max column height should be {}, got {}",
            expected_max_column_height,
            aux[idx]
        );
        idx += 1;

        let expected_min_column_height = expected_column_heights
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.0);
        assert!(
            (aux[idx] - expected_min_column_height).abs() < 1e-6,
            "Min column height should be {}, got {}",
            expected_min_column_height,
            aux[idx]
        );
        idx += 1;

        let expected_row_fill_counts = normalize_row_fill_counts(&env.row_fill_counts, env.width);
        let encoded_row_fill_counts = &aux[idx..idx + BOARD_HEIGHT];
        for row in 0..BOARD_HEIGHT {
            assert!(
                (encoded_row_fill_counts[row] - expected_row_fill_counts[row]).abs() < 1e-6,
                "Row fill count mismatch at row {}: expected {}, got {}",
                row,
                expected_row_fill_counts[row],
                encoded_row_fill_counts[row]
            );
        }
        idx += BOARD_HEIGHT;

        let expected_total_blocks = normalize_total_blocks(env.total_blocks, env.width, env.height);
        assert!(
            (aux[idx] - expected_total_blocks).abs() < 1e-6,
            "Total blocks should be {}, got {}",
            expected_total_blocks,
            aux[idx]
        );
        idx += 1;

        let expected_bumpiness = normalize_bumpiness(
            compute_bumpiness(&env.column_heights),
            env.width,
            env.height,
        );
        assert!(
            (aux[idx] - expected_bumpiness).abs() < 1e-6,
            "Bumpiness should be {}, got {}",
            expected_bumpiness,
            aux[idx]
        );
        idx += 1;

        let (expected_overhang_raw, expected_holes_raw) = count_overhang_fields_and_holes(&env);
        let expected_holes = normalize_holes(expected_holes_raw, env.width, env.height);
        assert!(
            (aux[idx] - expected_holes).abs() < 1e-6,
            "Holes should be {}, got {}",
            expected_holes,
            aux[idx]
        );
        idx += 1;

        let expected_overhang = normalize_overhang_fields(expected_overhang_raw);
        assert!(
            (aux[idx] - expected_overhang).abs() < 1e-6,
            "Overhang fields should be {}, got {}",
            expected_overhang,
            aux[idx]
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
        // | Column heights | 10         | normalized by board height       |
        // | Max column h   | 1          | max normalized column height     |
        // | Min column h   | 1          | min normalized column height     |
        // | Row fill counts| 20         | normalized by board width        |
        // | Total blocks   | 1          | normalized by board area         |
        // | Bumpiness      | 1          | normalized bumpiness             |
        // | Holes          | 1          | normalized hole count            |
        // | Overhang       | 1          | normalized overhang count        |

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
            7 + 8 + 1 + 35 + 1 + 1 + 1 + 7 + 10 + 1 + 1 + 20 + 1 + 1 + 1 + 1,
            "Aux should be 97 values"
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

        let column_heights = &aux[61..71];
        for &height in column_heights {
            assert!(
                (0.0..=1.0).contains(&height),
                "Normalized column heights must be in [0, 1]"
            );
        }

        let max_column_height = aux[71];
        let min_column_height = aux[72];
        assert!(
            (0.0..=1.0).contains(&max_column_height),
            "Max column height must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&min_column_height),
            "Min column height must be in [0, 1]"
        );
        assert!(
            min_column_height <= max_column_height,
            "Min column height must be <= max column height"
        );

        let row_fill_counts = &aux[73..93];
        for &row_fill in row_fill_counts {
            assert!(
                (0.0..=1.0).contains(&row_fill),
                "Normalized row fill counts must be in [0, 1]"
            );
        }

        let total_blocks = aux[93];
        let bumpiness = aux[94];
        let holes = aux[95];
        let overhang = aux[96];
        assert!(
            (0.0..=1.0).contains(&total_blocks),
            "Total blocks must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&bumpiness),
            "Bumpiness must be in [0, 1]"
        );
        assert!((0.0..=1.0).contains(&holes), "Holes must be in [0, 1]");
        assert!(
            (0.0..=1.0).contains(&overhang),
            "Overhang fields must be in [0, 1]"
        );
    }

    #[test]
    fn test_hidden_distribution_changes_with_visible_queue_horizon() {
        let mut env = TetrisEnv::new(10, 20);

        let first = next_hidden_piece_distribution(&env);
        let first_non_zero = first.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(
            first_non_zero, 1,
            "At game start, hidden-piece distribution should be deterministic for the next hidden piece"
        );

        env.hard_drop();
        let second = next_hidden_piece_distribution(&env);
        let second_non_zero = second.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(
            second_non_zero, NUM_PIECE_TYPES,
            "After one placement, hidden-piece distribution should reset to full-bag uncertainty"
        );
    }

    proptest! {
        #[test]
        fn prop_encoded_diagnostics_match_env_state(
            seed in 0u64..10_000,
            actions in prop::collection::vec(0u8..8, 0..80),
            max_placements in 1usize..200,
        ) {
            let mut env = TetrisEnv::with_seed(10, 20, seed);

            for (step_idx, action) in actions.iter().copied().enumerate() {
                let placement_count = step_idx % max_placements;
                let (_, aux_tensor) =
                    encode_state(&env, placement_count, max_placements)
                        .expect("encoding should succeed for valid max_placements");
                let aux_array = aux_tensor
                    .to_array_view::<f32>()
                    .expect("aux tensor should contain f32");
                let aux: Vec<f32> = aux_array.iter().copied().collect();
                prop_assert_eq!(aux.len(), AUX_FEATURES);

                let diagnostics_start = NUM_PIECE_TYPES
                    + (NUM_PIECE_TYPES + 1)
                    + 1
                    + (QUEUE_SIZE * NUM_PIECE_TYPES)
                    + 1
                    + 1
                    + 1
                    + NUM_PIECE_TYPES;

                let mut idx = diagnostics_start;

                let expected_column_heights =
                    normalize_column_heights(&env.column_heights, env.height);
                for expected in expected_column_heights.iter().take(BOARD_WIDTH) {
                    prop_assert!((aux[idx] - *expected).abs() < 1e-6);
                    prop_assert!((0.0..=1.0).contains(&aux[idx]));
                    idx += 1;
                }

                let expected_max_column_height = expected_column_heights
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_or(0.0);
                prop_assert!((aux[idx] - expected_max_column_height).abs() < 1e-6);
                idx += 1;

                let expected_min_column_height = expected_column_heights
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                prop_assert!((aux[idx] - expected_min_column_height).abs() < 1e-6);
                idx += 1;

                let expected_row_fill_counts =
                    normalize_row_fill_counts(&env.row_fill_counts, env.width);
                for expected in expected_row_fill_counts.iter().take(BOARD_HEIGHT) {
                    prop_assert!((aux[idx] - *expected).abs() < 1e-6);
                    prop_assert!((0.0..=1.0).contains(&aux[idx]));
                    idx += 1;
                }

                let expected_total_blocks =
                    normalize_total_blocks(env.total_blocks, env.width, env.height);
                prop_assert!((aux[idx] - expected_total_blocks).abs() < 1e-6);
                prop_assert!((0.0..=1.0).contains(&aux[idx]));
                idx += 1;

                let expected_raw_bumpiness = compute_bumpiness(&env.column_heights);
                let expected_bumpiness =
                    normalize_bumpiness(expected_raw_bumpiness, env.width, env.height);
                prop_assert!((aux[idx] - expected_bumpiness).abs() < 1e-6);
                prop_assert!((0.0..=1.0).contains(&aux[idx]));
                idx += 1;

                let (expected_raw_overhang, expected_raw_holes) =
                    count_overhang_fields_and_holes(&env);
                let expected_holes = normalize_holes(expected_raw_holes, env.width, env.height);
                prop_assert!((aux[idx] - expected_holes).abs() < 1e-6);
                prop_assert!((0.0..=1.0).contains(&aux[idx]));
                idx += 1;

                let expected_overhang = normalize_overhang_fields(expected_raw_overhang);
                prop_assert!((aux[idx] - expected_overhang).abs() < 1e-6);
                prop_assert!((0.0..=1.0).contains(&aux[idx]));
                idx += 1;

                prop_assert_eq!(idx, AUX_FEATURES);

                if env.game_over {
                    break;
                }
                env.step(action);
            }
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
    fn test_combo_feature_round_trip() {
        assert_eq!(denormalize_combo_feature(normalize_combo_for_feature(7)), 7);
        assert_eq!(
            denormalize_combo_feature(normalize_combo_for_feature(99)),
            COMBO_NORMALIZATION_MAX
        );
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
