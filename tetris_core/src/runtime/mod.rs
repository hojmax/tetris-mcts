//! Runtime orchestration for self-play generation and fixed-seed evaluation.

pub(crate) mod evaluation;
mod game_generator;

pub use evaluation::{evaluate_model, evaluate_model_without_nn, EvalResult};
pub use game_generator::GameGenerator;

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
