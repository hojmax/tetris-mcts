//! MCTS Utility Functions

use rand_distr::{Distribution, Gamma};

/// Sample from Dirichlet distribution.
///
/// # Arguments
/// * `alpha` - Concentration parameter (must be > 0)
/// * `n` - Number of samples
///
/// # Panics
/// Panics if alpha <= 0
pub fn sample_dirichlet<R: rand::Rng + ?Sized>(alpha: f32, n: usize, rng: &mut R) -> Vec<f32> {
    debug_assert!(
        alpha > 0.0,
        "Dirichlet alpha must be positive, got {}",
        alpha
    );
    debug_assert!(n > 0, "Dirichlet n must be positive");

    // Gamma::new only fails if alpha <= 0 or scale <= 0
    let gamma =
        Gamma::new(alpha as f64, 1.0).expect("Invalid Dirichlet alpha parameter (must be > 0)");

    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
    let sum: f64 = samples.iter().sum();

    if sum == 0.0 {
        // Fallback to uniform if all samples are zero (shouldn't happen with valid alpha)
        debug_assert!(
            false,
            "Dirichlet sampling produced all zeros with alpha={} n={} - falling back to uniform",
            alpha, n
        );
        return vec![1.0 / n as f32; n];
    }

    samples.into_iter().map(|x| (x / sum) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_dirichlet() {
        let mut rng = thread_rng();
        let noise = sample_dirichlet(0.15, 10, &mut rng);
        assert_eq!(noise.len(), 10);

        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
