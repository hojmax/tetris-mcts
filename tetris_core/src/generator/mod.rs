//! Background game generation and evaluation for parallel training.
//!
//! Provides:
//! - `GameGenerator`: Background thread that continuously generates self-play games
//! - `evaluate_model`: Evaluate a model on fixed seeds for consistent benchmarking
//! - `write_examples_to_npz` / `read_examples_from_npz`: NPZ replay I/O

pub(crate) mod evaluation;
mod game_generator;
pub mod npz;
pub(crate) mod types;

pub use evaluation::{evaluate_model, evaluate_model_without_nn, EvalResult};
pub use types::{GameReplay, ReplayMove};
pub use game_generator::GameGenerator;
pub use npz::{read_examples_from_npz, write_examples_to_npz};

#[cfg(test)]
pub(crate) mod test_utils {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn unique_temp_path(prefix: &str, name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "{}_{}_{}_{}.npz",
            prefix,
            name,
            std::process::id(),
            nanos
        ))
    }
}
