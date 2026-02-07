//! NPZ File Writing
//!
//! Write training examples to NPZ format (compatible with Python numpy).

use std::fs::File;
use std::io::{Cursor, Write};
use std::path::PathBuf;

use npyz::WriterBuilder;
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::mcts::{TrainingExample, NUM_ACTIONS};
use crate::piece::NUM_PIECE_TYPES;

/// Write training examples to NPZ format (compatible with Python numpy).
///
/// Format matches the Python training dataset layout expected by scripts:
/// - boards: (N, 20, 10) bool
/// - current_pieces: (N, 7) float32 one-hot
/// - hold_pieces: (N, 8) float32 one-hot
/// - hold_available: (N,) bool
/// - next_queue: (N, 5, 7) float32 one-hot
/// - move_numbers: (N,) float32 normalized
/// - policy_targets: (N, 734) float32
/// - value_targets: (N,) float32
/// - action_masks: (N, 734) bool
pub fn write_examples_to_npz(
    filepath: &PathBuf,
    examples: &[TrainingExample],
) -> Result<(), String> {
    let n = examples.len();
    if n == 0 {
        return Ok(());
    }

    // Create arrays
    let mut boards: Vec<u8> = Vec::with_capacity(n * BOARD_HEIGHT * BOARD_WIDTH);
    let mut current_pieces: Vec<f32> = vec![0.0; n * NUM_PIECE_TYPES];
    let mut hold_pieces: Vec<f32> = vec![0.0; n * (NUM_PIECE_TYPES + 1)];
    let mut hold_available: Vec<u8> = Vec::with_capacity(n);
    let mut next_queue: Vec<f32> = vec![0.0; n * 5 * NUM_PIECE_TYPES];
    let mut move_numbers: Vec<f32> = Vec::with_capacity(n);
    let mut policy_targets: Vec<f32> = Vec::with_capacity(n * NUM_ACTIONS);
    let mut value_targets: Vec<f32> = Vec::with_capacity(n);
    let mut action_masks: Vec<u8> = Vec::with_capacity(n * NUM_ACTIONS);

    for (i, ex) in examples.iter().enumerate() {
        // Board (flatten from (20, 10) to 200)
        boards.extend(ex.board.iter().copied());

        // Current piece one-hot
        current_pieces[i * NUM_PIECE_TYPES + ex.current_piece] = 1.0;

        // Hold piece one-hot (7 = empty slot)
        if ex.hold_piece < NUM_PIECE_TYPES {
            hold_pieces[i * (NUM_PIECE_TYPES + 1) + ex.hold_piece] = 1.0;
        } else {
            hold_pieces[i * (NUM_PIECE_TYPES + 1) + NUM_PIECE_TYPES] = 1.0; // Empty
        }

        // Hold available
        hold_available.push(ex.hold_available as u8);

        // Next queue one-hot (5 slots x 7 piece types)
        for (j, &piece) in ex.next_queue.iter().take(5).enumerate() {
            next_queue[i * 5 * NUM_PIECE_TYPES + j * NUM_PIECE_TYPES + piece] = 1.0;
        }

        // Move number (normalized)
        move_numbers.push(ex.move_number as f32 / 100.0);

        // Policy targets
        policy_targets.extend(ex.policy.iter().copied());

        // Value target
        value_targets.push(ex.value);

        // Action mask
        action_masks.extend(ex.action_mask.iter().map(|&b| b as u8));
    }

    // Create NPZ file (zip with npy arrays)
    let file = File::create(filepath).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    // Write each array
    write_npy_to_zip(
        &mut zip,
        options,
        "boards.npy",
        &[n as u64, BOARD_HEIGHT as u64, BOARD_WIDTH as u64],
        &boards,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "current_pieces.npy",
        &[n as u64, NUM_PIECE_TYPES as u64],
        &current_pieces,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "hold_pieces.npy",
        &[n as u64, (NUM_PIECE_TYPES + 1) as u64],
        &hold_pieces,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "hold_available.npy",
        &[n as u64],
        &hold_available,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "next_queue.npy",
        &[n as u64, 5, NUM_PIECE_TYPES as u64],
        &next_queue,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "move_numbers.npy",
        &[n as u64],
        &move_numbers,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "policy_targets.npy",
        &[n as u64, NUM_ACTIONS as u64],
        &policy_targets,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "value_targets.npy",
        &[n as u64],
        &value_targets,
    )?;
    write_npy_to_zip(
        &mut zip,
        options,
        "action_masks.npy",
        &[n as u64, NUM_ACTIONS as u64],
        &action_masks,
    )?;

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Helper to write an npy array to a zip file
fn write_npy_to_zip<T: npyz::Serialize + npyz::AutoSerialize + Copy>(
    zip: &mut ZipWriter<File>,
    options: FileOptions,
    name: &str,
    shape: &[u64],
    data: &[T],
) -> Result<(), String> {
    zip.start_file(name, options)
        .map_err(|e: zip::result::ZipError| e.to_string())?;

    // Write NPY format
    let mut buffer = Cursor::new(Vec::new());
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(&mut buffer)
        .begin_nd()
        .map_err(|e: std::io::Error| e.to_string())?;
    writer
        .extend(data.iter().copied())
        .map_err(|e: std::io::Error| e.to_string())?;
    writer.finish().map_err(|e: std::io::Error| e.to_string())?;

    zip.write_all(buffer.get_ref())
        .map_err(|e: std::io::Error| e.to_string())?;
    Ok(())
}
