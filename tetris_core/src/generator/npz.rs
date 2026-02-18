//! NPZ File I/O
//!
//! Read and write training examples in NPZ format (compatible with Python numpy).

use std::fs::File;
use std::path::Path;

use npyz::{NpyFile, WriterBuilder};
use zip::read::ZipArchive;
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, QUEUE_SIZE};
use crate::mcts::{TrainingExample, NUM_ACTIONS};

/// Write training examples to NPZ format (compatible with Python numpy).
///
/// Convenience wrapper over [`write_examples_slices_to_npz`] for a single contiguous slice.
pub fn write_examples_to_npz(filepath: &Path, examples: &[TrainingExample]) -> Result<(), String> {
    write_examples_slices_to_npz(filepath, examples, &[])
}

/// Streaming NPZ writer that accepts two contiguous slices (to support VecDeque::as_slices()).
///
/// Each array is streamed directly from the examples to the zip file — no intermediate
/// column vectors or in-memory buffers are allocated. Peak memory overhead is negligible
/// beyond the examples themselves.
///
/// Format matches the Python training dataset layout expected by scripts:
/// - boards: (N, 20, 10) bool
/// - current_pieces: (N, 7) float32 one-hot
/// - hold_pieces: (N, 8) float32 one-hot
/// - hold_available: (N,) bool
/// - next_queue: (N, 5, 7) float32 one-hot
/// - move_numbers: (N,) uint32 raw frame indices (includes holds)
/// - placement_counts: (N,) float32 normalized (placement_idx / max_placements, [0,1])
/// - combos: (N,) float32 normalized (min(combo, COMBO_NORMALIZATION_MAX) / COMBO_NORMALIZATION_MAX, [0,1])
/// - back_to_back: (N,) bool
/// - next_hidden_piece_probs: (N, 7) float32
/// - column_heights: (N, 10) float32 normalized by board height
/// - max_column_heights: (N,) float32
/// - min_column_heights: (N,) float32
/// - row_fill_counts: (N, 20) float32 normalized by board width
/// - total_blocks: (N,) float32 normalized by board area
/// - bumpiness: (N,) float32 normalized
/// - holes: (N,) float32 normalized by maximum possible holes
/// - policy_targets: (N, 735) float32
/// - value_targets: (N,) float32
/// - action_masks: (N, 735) bool
/// - overhang_fields: (N,) float32 normalized by maximum possible overhang fields
/// - game_numbers: (N,) uint64 (1-indexed game IDs aligned with WandB game_number)
/// - game_total_attacks: (N,) uint32 (raw total attack for each example's source game)
pub(crate) fn write_examples_slices_to_npz(
    filepath: &Path,
    examples_a: &[TrainingExample],
    examples_b: &[TrainingExample],
) -> Result<(), String> {
    let n = examples_a.len() + examples_b.len();
    if n == 0 {
        return Ok(());
    }

    let examples = || examples_a.iter().chain(examples_b.iter());
    let n64 = n as u64;

    let file = File::create(filepath).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    stream_npy_to_zip(
        &mut zip,
        options,
        "boards.npy",
        &[n64, BOARD_HEIGHT as u64, BOARD_WIDTH as u64],
        examples().flat_map(|ex| ex.board.iter().copied()),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "current_pieces.npy",
        &[n64, NUM_PIECE_TYPES as u64],
        examples().flat_map(|ex| {
            let mut one_hot = [0.0f32; NUM_PIECE_TYPES];
            one_hot[ex.current_piece] = 1.0;
            one_hot.into_iter()
        }),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "hold_pieces.npy",
        &[n64, (NUM_PIECE_TYPES + 1) as u64],
        examples().flat_map(|ex| {
            let mut one_hot = [0.0f32; NUM_PIECE_TYPES + 1];
            if ex.hold_piece < NUM_PIECE_TYPES {
                one_hot[ex.hold_piece] = 1.0;
            } else {
                one_hot[NUM_PIECE_TYPES] = 1.0;
            }
            one_hot.into_iter()
        }),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "hold_available.npy",
        &[n64],
        examples().map(|ex| ex.hold_available as u8),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "next_queue.npy",
        &[n64, QUEUE_SIZE as u64, NUM_PIECE_TYPES as u64],
        examples().flat_map(|ex| {
            let mut queue_data = [0.0f32; QUEUE_SIZE * NUM_PIECE_TYPES];
            for (j, &piece) in ex.next_queue.iter().take(QUEUE_SIZE).enumerate() {
                queue_data[j * NUM_PIECE_TYPES + piece] = 1.0;
            }
            queue_data.into_iter()
        }),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "move_numbers.npy",
        &[n64],
        examples().map(|ex| ex.move_number),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "placement_counts.npy",
        &[n64],
        examples().map(|ex| ex.placement_count),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "combos.npy",
        &[n64],
        examples().map(|ex| ex.combo),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "back_to_back.npy",
        &[n64],
        examples().map(|ex| ex.back_to_back as u8),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "next_hidden_piece_probs.npy",
        &[n64, NUM_PIECE_TYPES as u64],
        examples().flat_map(|ex| ex.next_hidden_piece_probs.iter().copied()),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "column_heights.npy",
        &[n64, BOARD_WIDTH as u64],
        examples().flat_map(|ex| ex.column_heights.iter().copied()),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "max_column_heights.npy",
        &[n64],
        examples().map(|ex| ex.max_column_height),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "min_column_heights.npy",
        &[n64],
        examples().map(|ex| ex.min_column_height),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "row_fill_counts.npy",
        &[n64, BOARD_HEIGHT as u64],
        examples().flat_map(|ex| ex.row_fill_counts.iter().copied()),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "total_blocks.npy",
        &[n64],
        examples().map(|ex| ex.total_blocks),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "bumpiness.npy",
        &[n64],
        examples().map(|ex| ex.bumpiness),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "holes.npy",
        &[n64],
        examples().map(|ex| ex.holes),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "policy_targets.npy",
        &[n64, NUM_ACTIONS as u64],
        examples().flat_map(|ex| ex.policy.iter().copied()),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "value_targets.npy",
        &[n64],
        examples().map(|ex| ex.value),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "action_masks.npy",
        &[n64, NUM_ACTIONS as u64],
        examples().flat_map(|ex| ex.action_mask.iter().map(|&b| b as u8)),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "overhang_fields.npy",
        &[n64],
        examples().map(|ex| ex.overhang_fields),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "game_numbers.npy",
        &[n64],
        examples().map(|ex| ex.game_number),
    )?;

    stream_npy_to_zip(
        &mut zip,
        options,
        "game_total_attacks.npy",
        &[n64],
        examples().map(|ex| ex.game_total_attack),
    )?;

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Read training examples from NPZ format.
pub fn read_examples_from_npz(filepath: &Path) -> Result<Vec<TrainingExample>, String> {
    let file = File::open(filepath).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;

    let (boards, boards_shape) = read_npy_array::<u8>(&mut archive, "boards.npy")?;
    let (current_pieces, current_pieces_shape) =
        read_npy_array::<f32>(&mut archive, "current_pieces.npy")?;
    let (hold_pieces, hold_pieces_shape) = read_npy_array::<f32>(&mut archive, "hold_pieces.npy")?;
    let (hold_available, hold_available_shape) =
        read_npy_array_bool_like(&mut archive, "hold_available.npy")?;
    let (next_queue, next_queue_shape) = read_npy_array::<f32>(&mut archive, "next_queue.npy")?;
    let (move_numbers, move_numbers_shape) =
        read_npy_array::<u32>(&mut archive, "move_numbers.npy")?;
    let (placement_counts, placement_counts_shape) =
        read_npy_array::<f32>(&mut archive, "placement_counts.npy")?;
    let (combos, combos_shape) = read_npy_array::<f32>(&mut archive, "combos.npy")?;
    let (back_to_back, back_to_back_shape) =
        read_npy_array_bool_like(&mut archive, "back_to_back.npy")?;
    let (next_hidden_piece_probs, next_hidden_piece_probs_shape) =
        read_npy_array::<f32>(&mut archive, "next_hidden_piece_probs.npy")?;
    let (column_heights, column_heights_shape) =
        read_npy_array::<f32>(&mut archive, "column_heights.npy")?;
    let (max_column_heights, max_column_heights_shape) =
        read_npy_array::<f32>(&mut archive, "max_column_heights.npy")?;
    let (min_column_heights, min_column_heights_shape) =
        read_npy_array::<f32>(&mut archive, "min_column_heights.npy")?;
    let (row_fill_counts, row_fill_counts_shape) =
        read_npy_array::<f32>(&mut archive, "row_fill_counts.npy")?;
    let (total_blocks, total_blocks_shape) =
        read_npy_array::<f32>(&mut archive, "total_blocks.npy")?;
    let (bumpiness, bumpiness_shape) = read_npy_array::<f32>(&mut archive, "bumpiness.npy")?;
    let (holes, holes_shape) = read_npy_array::<f32>(&mut archive, "holes.npy")?;
    let (policy_targets, policy_targets_shape) =
        read_npy_array::<f32>(&mut archive, "policy_targets.npy")?;
    let (value_targets, value_targets_shape) =
        read_npy_array::<f32>(&mut archive, "value_targets.npy")?;
    let (action_masks, action_masks_shape) =
        read_npy_array_bool_like(&mut archive, "action_masks.npy")?;
    let (overhang_fields, overhang_fields_shape) =
        read_npy_array::<f32>(&mut archive, "overhang_fields.npy")?;
    let (game_numbers, game_numbers_shape) =
        read_npy_array::<u64>(&mut archive, "game_numbers.npy")?;
    let (game_total_attacks, game_total_attacks_shape) =
        read_npy_array::<u32>(&mut archive, "game_total_attacks.npy")?;

    let expected_boards_shape = vec![0, BOARD_HEIGHT as u64, BOARD_WIDTH as u64];
    let n = validate_shape_with_dynamic_batch("boards", &boards_shape, &expected_boards_shape)?;
    validate_shape(
        "current_pieces",
        &current_pieces_shape,
        &[n as u64, NUM_PIECE_TYPES as u64],
    )?;
    validate_shape(
        "hold_pieces",
        &hold_pieces_shape,
        &[n as u64, (NUM_PIECE_TYPES + 1) as u64],
    )?;
    validate_shape("hold_available", &hold_available_shape, &[n as u64])?;
    validate_shape(
        "next_queue",
        &next_queue_shape,
        &[n as u64, QUEUE_SIZE as u64, NUM_PIECE_TYPES as u64],
    )?;
    validate_shape("move_numbers", &move_numbers_shape, &[n as u64])?;
    validate_shape("placement_counts", &placement_counts_shape, &[n as u64])?;
    validate_shape("combos", &combos_shape, &[n as u64])?;
    validate_shape("back_to_back", &back_to_back_shape, &[n as u64])?;
    validate_shape(
        "next_hidden_piece_probs",
        &next_hidden_piece_probs_shape,
        &[n as u64, NUM_PIECE_TYPES as u64],
    )?;
    validate_shape(
        "column_heights",
        &column_heights_shape,
        &[n as u64, BOARD_WIDTH as u64],
    )?;
    validate_shape("max_column_heights", &max_column_heights_shape, &[n as u64])?;
    validate_shape("min_column_heights", &min_column_heights_shape, &[n as u64])?;
    validate_shape(
        "row_fill_counts",
        &row_fill_counts_shape,
        &[n as u64, BOARD_HEIGHT as u64],
    )?;
    validate_shape("total_blocks", &total_blocks_shape, &[n as u64])?;
    validate_shape("bumpiness", &bumpiness_shape, &[n as u64])?;
    validate_shape("holes", &holes_shape, &[n as u64])?;
    validate_shape(
        "policy_targets",
        &policy_targets_shape,
        &[n as u64, NUM_ACTIONS as u64],
    )?;
    validate_shape("value_targets", &value_targets_shape, &[n as u64])?;
    validate_shape(
        "action_masks",
        &action_masks_shape,
        &[n as u64, NUM_ACTIONS as u64],
    )?;
    validate_shape("overhang_fields", &overhang_fields_shape, &[n as u64])?;
    validate_shape("game_numbers", &game_numbers_shape, &[n as u64])?;
    validate_shape("game_total_attacks", &game_total_attacks_shape, &[n as u64])?;

    let mut examples = Vec::with_capacity(n);
    let board_size = BOARD_HEIGHT * BOARD_WIDTH;
    let hold_size = NUM_PIECE_TYPES + 1;
    let next_queue_size = QUEUE_SIZE * NUM_PIECE_TYPES;
    let next_hidden_piece_probs_size = NUM_PIECE_TYPES;
    let column_heights_size = BOARD_WIDTH;
    let row_fill_counts_size = BOARD_HEIGHT;

    for i in 0..n {
        let overhang_feature = overhang_fields[i];
        if !overhang_feature.is_finite() || !(0.0..=1.0).contains(&overhang_feature) {
            return Err(format!(
                "overhang_fields[{}] must be finite and in [0, 1], got {}",
                i, overhang_feature
            ));
        }

        let board_start = i * board_size;
        let board_end = board_start + board_size;

        let current_piece_start = i * NUM_PIECE_TYPES;
        let current_piece_end = current_piece_start + NUM_PIECE_TYPES;
        let current_piece = argmax_index(&current_pieces[current_piece_start..current_piece_end]);

        let hold_piece_start = i * hold_size;
        let hold_piece_end = hold_piece_start + hold_size;
        let hold_piece_index = argmax_index(&hold_pieces[hold_piece_start..hold_piece_end]);
        let hold_piece = if hold_piece_index < NUM_PIECE_TYPES {
            hold_piece_index
        } else {
            NUM_PIECE_TYPES
        };

        let next_queue_start = i * next_queue_size;
        let mut next_queue_pieces = Vec::with_capacity(QUEUE_SIZE);
        for slot in 0..QUEUE_SIZE {
            let slot_start = next_queue_start + slot * NUM_PIECE_TYPES;
            let slot_end = slot_start + NUM_PIECE_TYPES;
            next_queue_pieces.push(argmax_index(&next_queue[slot_start..slot_end]));
        }

        let policy_start = i * NUM_ACTIONS;
        let policy_end = policy_start + NUM_ACTIONS;
        let mask_start = i * NUM_ACTIONS;
        let mask_end = mask_start + NUM_ACTIONS;
        let hidden_probs_start = i * next_hidden_piece_probs_size;
        let hidden_probs_end = hidden_probs_start + next_hidden_piece_probs_size;
        let column_heights_start = i * column_heights_size;
        let column_heights_end = column_heights_start + column_heights_size;
        let row_fill_counts_start = i * row_fill_counts_size;
        let row_fill_counts_end = row_fill_counts_start + row_fill_counts_size;

        examples.push(TrainingExample {
            board: boards[board_start..board_end].to_vec(),
            current_piece,
            hold_piece,
            hold_available: hold_available[i] != 0,
            next_queue: next_queue_pieces,
            move_number: move_numbers[i],
            placement_count: placement_counts[i],
            combo: combos[i],
            back_to_back: back_to_back[i] != 0,
            next_hidden_piece_probs: next_hidden_piece_probs[hidden_probs_start..hidden_probs_end]
                .to_vec(),
            column_heights: column_heights[column_heights_start..column_heights_end].to_vec(),
            max_column_height: max_column_heights[i],
            min_column_height: min_column_heights[i],
            row_fill_counts: row_fill_counts[row_fill_counts_start..row_fill_counts_end].to_vec(),
            total_blocks: total_blocks[i],
            bumpiness: bumpiness[i],
            holes: holes[i],
            policy: policy_targets[policy_start..policy_end].to_vec(),
            value: value_targets[i],
            action_mask: action_masks[mask_start..mask_end]
                .iter()
                .map(|value| *value != 0)
                .collect(),
            overhang_fields: overhang_feature,
            game_number: game_numbers[i],
            game_total_attack: game_total_attacks[i],
        });
    }

    Ok(examples)
}

fn argmax_index(values: &[f32]) -> usize {
    let mut max_index = 0;
    let mut max_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > max_value {
            max_value = *value;
            max_index = index;
        }
    }
    max_index
}

fn validate_shape(name: &str, actual: &[u64], expected: &[u64]) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{} has shape {:?}, expected {:?}",
            name, actual, expected
        ));
    }
    Ok(())
}

fn validate_shape_with_dynamic_batch(
    name: &str,
    actual: &[u64],
    expected_with_batch_placeholder: &[u64],
) -> Result<usize, String> {
    if actual.len() != expected_with_batch_placeholder.len() {
        return Err(format!(
            "{} has rank {}, expected rank {}",
            name,
            actual.len(),
            expected_with_batch_placeholder.len()
        ));
    }
    for i in 1..actual.len() {
        if actual[i] != expected_with_batch_placeholder[i] {
            return Err(format!(
                "{} has shape {:?}, expected [N, {}, {}]",
                name,
                actual,
                expected_with_batch_placeholder[1],
                expected_with_batch_placeholder[2]
            ));
        }
    }
    Ok(actual[0] as usize)
}

fn read_npy_array<T: npyz::Deserialize>(
    archive: &mut ZipArchive<File>,
    entry_name: &str,
) -> Result<(Vec<T>, Vec<u64>), String> {
    let file = archive.by_name(entry_name).map_err(|e| e.to_string())?;
    let npy = NpyFile::new(file).map_err(|e| e.to_string())?;
    let shape = npy.shape().to_vec();
    let values = npy.into_vec::<T>().map_err(|e| e.to_string())?;
    Ok((values, shape))
}

fn read_npy_array_bool_like(
    archive: &mut ZipArchive<File>,
    entry_name: &str,
) -> Result<(Vec<u8>, Vec<u64>), String> {
    if let Ok((values, shape)) = read_npy_array::<u8>(archive, entry_name) {
        return Ok((values, shape));
    }

    let (values, shape) = read_npy_array::<bool>(archive, entry_name)?;
    Ok((values.into_iter().map(u8::from).collect(), shape))
}

/// Stream an npy array directly to a zip entry without intermediate buffers.
fn stream_npy_to_zip<T: npyz::Serialize + npyz::AutoSerialize>(
    zip: &mut ZipWriter<File>,
    options: FileOptions,
    name: &str,
    shape: &[u64],
    data: impl Iterator<Item = T>,
) -> Result<(), String> {
    zip.start_file(name, options)
        .map_err(|e: zip::result::ZipError| e.to_string())?;

    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(&mut *zip)
        .begin_nd()
        .map_err(|e: std::io::Error| e.to_string())?;
    writer
        .extend(data)
        .map_err(|e: std::io::Error| e.to_string())?;
    writer.finish().map_err(|e: std::io::Error| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::test_utils;
    use crate::mcts::NUM_ACTIONS;
    use std::fs;
    use std::path::PathBuf;

    fn unique_temp_path(name: &str) -> PathBuf {
        test_utils::unique_temp_path("tetris_core", name)
    }

    fn make_example(move_number: u32, hold_piece: usize, combo_raw: u32) -> TrainingExample {
        use crate::nn::normalize_combo_for_feature;

        let mut board = vec![0u8; BOARD_HEIGHT * BOARD_WIDTH];
        board[0] = 1;
        board[42] = 1;

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[0] = 0.25;
        policy[13] = 0.75;

        let mut action_mask = vec![false; NUM_ACTIONS];
        action_mask[0] = true;
        action_mask[13] = true;

        TrainingExample {
            board,
            current_piece: 2,
            hold_piece,
            hold_available: true,
            next_queue: vec![0, 1, 2, 3, 4],
            move_number,
            placement_count: move_number as f32 / 100.0,
            combo: normalize_combo_for_feature(combo_raw),
            back_to_back: true,
            next_hidden_piece_probs: vec![0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0],
            column_heights: vec![0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            max_column_height: 0.45,
            min_column_height: 0.0,
            row_fill_counts: vec![0.0; BOARD_HEIGHT],
            total_blocks: 0.01,
            bumpiness: 0.0025,
            holes: 0.05,
            policy,
            value: 3.5,
            action_mask,
            overhang_fields: crate::mcts::normalize_overhang_fields(17),
            game_number: 123,
            game_total_attack: 37,
        }
    }

    #[test]
    fn test_npz_round_trip_preserves_examples() {
        let path = unique_temp_path("roundtrip");
        let examples = vec![make_example(0, 7, 3), make_example(88, 5, 3)];

        write_examples_to_npz(&path, &examples).expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded.len(), examples.len());
        for (expected, actual) in examples.iter().zip(loaded.iter()) {
            assert_eq!(actual.board, expected.board);
            assert_eq!(actual.current_piece, expected.current_piece);
            assert_eq!(actual.hold_piece, expected.hold_piece);
            assert_eq!(actual.hold_available, expected.hold_available);
            assert_eq!(actual.next_queue, expected.next_queue);
            assert_eq!(actual.move_number, expected.move_number);
            assert_eq!(actual.placement_count, expected.placement_count);
            assert_eq!(actual.combo, expected.combo);
            assert_eq!(actual.back_to_back, expected.back_to_back);
            assert_eq!(
                actual.next_hidden_piece_probs,
                expected.next_hidden_piece_probs
            );
            assert_eq!(actual.column_heights, expected.column_heights);
            assert_eq!(actual.max_column_height, expected.max_column_height);
            assert_eq!(actual.min_column_height, expected.min_column_height);
            assert_eq!(actual.row_fill_counts, expected.row_fill_counts);
            assert_eq!(actual.total_blocks, expected.total_blocks);
            assert_eq!(actual.bumpiness, expected.bumpiness);
            assert_eq!(actual.holes, expected.holes);
            assert_eq!(actual.value, expected.value);
            assert_eq!(actual.policy.len(), expected.policy.len());
            assert_eq!(actual.action_mask, expected.action_mask);
            assert_eq!(actual.overhang_fields, expected.overhang_fields);
            assert_eq!(actual.game_number, expected.game_number);
            assert_eq!(actual.game_total_attack, expected.game_total_attack);
            assert_eq!(actual.policy[0], expected.policy[0]);
            assert_eq!(actual.policy[13], expected.policy[13]);
        }
    }

    #[test]
    fn test_validate_shape_with_dynamic_batch_rejects_mismatched_dims() {
        let err = validate_shape_with_dynamic_batch("boards", &[2, 19, 10], &[0, 20, 10])
            .expect_err("shape mismatch should error");
        assert!(err.contains("expected [N, 20, 10]"));
    }

    #[test]
    fn test_npz_combo_round_trip_preserves_normalized_value() {
        use crate::nn::normalize_combo_for_feature;

        let path = unique_temp_path("combo_normalized");
        // combo=99 clamps to COMBO_NORMALIZATION_MAX and normalizes to 1.0
        let examples = vec![make_example(0, 7, 99)];

        write_examples_to_npz(&path, &examples).expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded.len(), 1);
        assert!((loaded[0].combo - normalize_combo_for_feature(99)).abs() < 1e-6);
    }

    #[test]
    fn test_npz_slices_round_trip() {
        let path = unique_temp_path("slices_roundtrip");
        let examples_a = vec![make_example(0, 7, 3)];
        let examples_b = vec![make_example(88, 5, 3)];

        write_examples_slices_to_npz(&path, &examples_a, &examples_b)
            .expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].move_number, 0);
        assert_eq!(loaded[1].move_number, 88);
    }
}
