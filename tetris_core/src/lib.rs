//! Tetris Core Library
//!
//! A high-performance Tetris game engine written in Rust with Python bindings.
//!
//! # Modules
//!
//! - `piece`: Tetromino pieces, shapes, and colors
//! - `kicks`: SRS (Super Rotation System) wall kick data
//! - `env`: The main Tetris game environment
//! - `scoring`: Attack scoring system (T-spins, combos, back-to-back, perfect clears)
//! - `moves`: Move generation for finding all possible piece placements
//! - `mcts`: Monte Carlo Tree Search for AlphaZero-style play
//! - `nn`: Neural network inference using tract-onnx
//! - `generator`: Background game generation and evaluation

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
pub use env::TetrisEnv;
pub use generator::{GameGenerator, EvalResult, evaluate_model};
pub use kicks::{get_i_kicks, get_jlstz_kicks, get_kicks_for_piece};
pub use mcts::{MCTSAgent, MCTSConfig, MCTSResult, MCTSTreeExport, TrainingExample, TreeNodeExport, GameResult};
pub use moves::{find_all_placements, Action, Board, Placement};
pub use piece::{get_cells_for_shape, Piece, COLORS, NUM_PIECE_TYPES, TETROMINOS};
pub use scoring::{
    calculate_attack, combo_attack, determine_clear_type, AttackResult, ClearType,
    BACK_TO_BACK_BONUS, PERFECT_CLEAR_ATTACK,
};

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
    m.add_function(wrap_pyfunction!(evaluate_model, m)?)?;
    Ok(())
}
