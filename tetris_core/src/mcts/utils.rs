//! MCTS Utility Functions

use rand::prelude::*;
use rand_distr::{Distribution, Gamma};

/// Sample from Dirichlet distribution
pub fn sample_dirichlet(alpha: f32, n: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    let gamma = Gamma::new(alpha as f64, 1.0).unwrap();

    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();

    samples.into_iter().map(|x| (x / sum) as f32).collect()
}

/// Sample an action from a policy distribution
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

    // Fallback to last action
    policy.len() - 1
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
