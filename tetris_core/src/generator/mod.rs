//! Background game generation and evaluation for parallel training.
//!
//! Provides:
//! - `GameGenerator`: Background thread that continuously generates self-play games
//! - `evaluate_model`: Evaluate a model on fixed seeds for consistent benchmarking
//! - `write_examples_to_npz`: Write training data to NPZ format

mod evaluation;
mod game_generator;
pub mod npz;

pub use evaluation::{evaluate_model, EvalResult, GameReplay, ReplayMove};
pub use game_generator::GameGenerator;
pub use npz::write_examples_to_npz;
