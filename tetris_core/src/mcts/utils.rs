//! MCTS Utility Functions

use rand::prelude::*;
use rand_distr::{Distribution, Gamma};

/// Sample from Dirichlet distribution.
///
/// # Arguments
/// * `alpha` - Concentration parameter (must be > 0)
/// * `n` - Number of samples
///
/// # Panics
/// Panics if alpha <= 0
pub fn sample_dirichlet(alpha: f32, n: usize) -> Vec<f32> {
    debug_assert!(alpha > 0.0, "Dirichlet alpha must be positive, got {}", alpha);
    debug_assert!(n > 0, "Dirichlet n must be positive");

    let mut rng = thread_rng();
    // Gamma::new only fails if alpha <= 0 or scale <= 0
    let gamma = Gamma::new(alpha as f64, 1.0)
        .expect("Invalid Dirichlet alpha parameter (must be > 0)");

    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();

    if sum == 0.0 {
        // Fallback to uniform if all samples are zero (shouldn't happen with valid alpha)
        return vec![1.0 / n as f32; n];
    }

    samples.into_iter().map(|x| (x / sum) as f32).collect()
}

/// Sample an action index from a policy distribution.
///
/// # Arguments
/// * `policy` - Probability distribution over actions (should sum to ~1.0)
///
/// # Returns
/// Index of sampled action
#[inline]
pub fn sample_action(policy: &[f32]) -> usize {
    let mut rng = thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in policy.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }

    // Fallback to last action (handles floating-point rounding)
    policy.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet() {
        let noise = sample_dirichlet(0.15, 10);
        assert_eq!(noise.len(), 10);

        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
