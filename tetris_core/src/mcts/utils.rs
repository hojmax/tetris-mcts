//! MCTS Utility Functions

use rand_distr::{Distribution, Gamma};

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::TetrisEnv;

/// Normalization denominator for overhang fields.
/// For each column, max overhang is `height - 1` (top filled, rest empty).
/// On a 10x20 board: 10 * 19 = 190.
pub const OVERHANG_NORMALIZATION_DENOMINATOR: f32 = (BOARD_WIDTH * (BOARD_HEIGHT - 1)) as f32;

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

/// Compute proportional death penalty based on how early the game ended.
/// Dying early (low move_number) incurs a larger penalty than dying late.
/// Returns 0.0 if death_penalty is 0.0 or move_number >= max_moves.
pub fn compute_death_penalty(move_number: u32, max_moves: u32, death_penalty: f32) -> f32 {
    let remaining = (max_moves as f32 - move_number as f32) / max_moves as f32;
    death_penalty * remaining.max(0.0)
}

/// Count overhang fields (empty cells with at least one filled cell above in the same column).
pub fn count_overhang_fields(env: &TetrisEnv) -> u32 {
    let mut overhang_fields: u32 = 0;
    let board = env.board_cells();

    for x in 0..env.width {
        let mut seen_filled = false;
        for y in 0..env.height {
            let cell = board[y * env.width + x];
            if cell != 0 {
                seen_filled = true;
                continue;
            }
            if seen_filled {
                overhang_fields += 1;
            }
        }
    }

    overhang_fields
}

/// Compute normalized overhang penalty magnitude from raw overhang count.
pub fn compute_overhang_penalty(overhang_fields: u32, overhang_penalty_weight: f32) -> f32 {
    (overhang_fields as f32 / OVERHANG_NORMALIZATION_DENOMINATOR) * overhang_penalty_weight
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

    #[test]
    fn test_count_overhang_fields_empty_board() {
        let mut env = TetrisEnv::with_seed(10, 20, 1);
        env.set_board(vec![vec![0; 10]; 20])
            .expect("set_board should succeed");
        assert_eq!(count_overhang_fields(&env), 0);
    }

    #[test]
    fn test_count_overhang_fields_counts_holes_below_filled_cells() {
        let mut env = TetrisEnv::with_seed(10, 20, 2);
        let mut board = vec![vec![0; 10]; 20];
        board[0][0] = 1;
        board[2][0] = 1;
        board[5][1] = 1;
        env.set_board(board).expect("set_board should succeed");

        // Column 0: y=1,3..19 are overhangs => 18
        // Column 1: y=6..19 are overhangs => 14
        assert_eq!(count_overhang_fields(&env), 32);
    }

    #[test]
    fn test_compute_overhang_penalty_uses_fixed_normalization() {
        let penalty = compute_overhang_penalty(95, 2.0);
        // 95 / 190 = 0.5
        assert!((penalty - 1.0).abs() < 1e-6);
    }
}
