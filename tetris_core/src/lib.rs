//! Tetris Core Library
//!
//! A high-performance Tetris game engine written in Rust with Python bindings.
//!
//! # Modules
//!
//! - `piece`: Tetromino pieces, shapes, and colors
//! - `kicks`: SRS (Super Rotation System) wall kick data
//! - `env`: The main Tetris game environment

use pyo3::prelude::*;

pub mod env;
pub mod kicks;
pub mod piece;

// Re-export main types for convenience
pub use env::{generate_bag, TetrisEnv};
pub use kicks::{get_i_kicks, get_jlstz_kicks, get_kicks_for_piece, get_o_kicks, rotate_ccw, rotate_cw};
pub use piece::{get_cells_for_shape, Piece, COLORS, NUM_PIECE_TYPES, TETROMINOS};

#[pymodule]
fn tetris_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TetrisEnv>()?;
    m.add_class::<Piece>()?;
    Ok(())
}
