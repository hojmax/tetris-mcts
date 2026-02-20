//! Tetris Core Library
//!
//! A high-performance Tetris game engine written in Rust with Python bindings.
//!
//! # Modules
//!
//! - `piece`: Tetromino pieces and geometry
//! - `kicks`: SRS (Super Rotation System) wall kick data
//! - `env`: The main Tetris game environment
//! - `scoring`: Attack scoring system (T-spins, combos, back-to-back, perfect clears)
//! - `moves`: Move generation for finding all possible piece placements
//! - `mcts`: Monte Carlo Tree Search for AlphaZero-style play
//! - `nn`: Neural network inference (tract default, optional ONNX Runtime backend)
//! - `generator`: Background game generation and evaluation
#![allow(non_local_definitions)] // PyO3 #[pymethods] triggers this warning with current toolchain.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod constants;
pub mod env;
pub mod generator;
pub mod kicks;
pub mod mcts;
pub mod moves;
pub mod nn;
pub mod piece;
pub mod scoring;

// Re-export main types for convenience
pub use constants::NUM_PIECE_TYPES;
pub use env::TetrisEnv;
pub use generator::{
    evaluate_model, evaluate_model_without_nn, EvalResult, GameGenerator, GameReplay, ReplayMove,
};
pub use kicks::{get_i_kicks, get_jlstz_kicks, get_kicks_for_piece};
pub use mcts::{
    GameResult, MCTSAgent, MCTSConfig, MCTSResult, MCTSTreeExport, TrainingExample, TreeNodeExport,
};
pub use moves::{find_all_placements, Action, Board, Placement};
pub use piece::{get_cells, Piece, TETROMINOS, TETROMINO_CELLS};
pub use scoring::{
    calculate_attack, combo_attack, determine_clear_type, AttackResult, ClearType,
    BACK_TO_BACK_BONUS, PERFECT_CLEAR_ATTACK,
};

#[pyfunction]
fn debug_encode_state(
    env: &TetrisEnv,
    move_number: usize,
    max_placements: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    nn::encode_state_features(env, move_number, max_placements)
        .map_err(|e| PyValueError::new_err(format!("Failed to encode state: {e}")))
}

#[pyfunction]
fn debug_get_action_mask(env: &TetrisEnv) -> Vec<bool> {
    nn::get_action_mask(env)
}

#[pyfunction]
fn debug_masked_softmax(logits: Vec<f32>, mask: Vec<bool>) -> PyResult<Vec<f32>> {
    if logits.len() != mask.len() {
        return Err(PyValueError::new_err(format!(
            "logits and mask length mismatch: logits={}, mask={}",
            logits.len(),
            mask.len()
        )));
    }
    if !mask.iter().any(|&valid| valid) {
        return Err(PyValueError::new_err(
            "mask must contain at least one valid action",
        ));
    }
    Ok(nn::masked_softmax(&logits, &mask))
}

#[pyfunction]
fn debug_predict_masked_from_tensors(
    model_path: &str,
    board_tensor: Vec<f32>,
    aux_tensor: Vec<f32>,
    action_mask: Vec<bool>,
) -> PyResult<(Vec<f32>, f32)> {
    let expected_board = constants::BOARD_HEIGHT * constants::BOARD_WIDTH;
    if board_tensor.len() != expected_board {
        return Err(PyValueError::new_err(format!(
            "board tensor length mismatch: got {}, expected {}",
            board_tensor.len(),
            expected_board
        )));
    }
    if aux_tensor.len() != constants::AUX_FEATURES {
        return Err(PyValueError::new_err(format!(
            "aux tensor length mismatch: got {}, expected {}",
            aux_tensor.len(),
            constants::AUX_FEATURES
        )));
    }
    if action_mask.len() != mcts::NUM_ACTIONS {
        return Err(PyValueError::new_err(format!(
            "action mask length mismatch: got {}, expected {}",
            action_mask.len(),
            mcts::NUM_ACTIONS
        )));
    }
    if !action_mask.iter().any(|&valid| valid) {
        return Err(PyValueError::new_err(
            "action mask must contain at least one valid action",
        ));
    }

    let nn = nn::TetrisNN::load(model_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load model: {e}")))?;
    nn.predict_masked_from_tensors(&board_tensor, &aux_tensor, &action_mask)
        .map_err(|e| PyValueError::new_err(format!("Failed to run inference: {e}")))
}

#[pymodule]
fn tetris_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TetrisEnv>()?;
    m.add_class::<Piece>()?;
    m.add_class::<AttackResult>()?;
    m.add_class::<Placement>()?;
    m.add_class::<MCTSConfig>()?;
    m.add_class::<MCTSAgent>()?;
    m.add_class::<MCTSResult>()?;
    m.add_class::<TrainingExample>()?;
    m.add_class::<GameResult>()?;
    m.add_class::<TreeNodeExport>()?;
    m.add_class::<MCTSTreeExport>()?;
    m.add_class::<GameGenerator>()?;
    m.add_class::<EvalResult>()?;
    m.add_class::<ReplayMove>()?;
    m.add_class::<GameReplay>()?;
    m.add_function(wrap_pyfunction!(evaluate_model, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_model_without_nn, m)?)?;
    m.add_function(wrap_pyfunction!(debug_encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(debug_get_action_mask, m)?)?;
    m.add_function(wrap_pyfunction!(debug_masked_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(debug_predict_masked_from_tensors, m)?)?;
    Ok(())
}
