//! Replay persistence and serialization primitives.

pub mod npz;
pub mod types;

pub use npz::{read_examples_from_npz, write_examples_to_npz};
pub use types::{GameReplay, ReplayMove};
