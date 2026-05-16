//! Shared replay types used by both evaluation and game generation.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A single move in a game replay.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayMove {
    pub action: usize,
    pub attack: u32,
}

/// A complete game replay that can be saved and replayed.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameReplay {
    pub seed: u64,
    pub moves: Vec<ReplayMove>,
    pub total_attack: u32,
    pub num_moves: u32,
}
