//! MCTS Utility Functions

use rand_distr::{Distribution, Gamma};

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::env::global_cache::{
    build_board_key, get_cached_board_analysis, insert_cached_board_analysis,
};
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
/// Returns 0.0 if death_penalty is 0.0 or move_number >= max_placements.
pub fn compute_death_penalty(move_number: u32, max_placements: u32, death_penalty: f32) -> f32 {
    let remaining = (max_placements as f32 - move_number as f32) / max_placements as f32;
    death_penalty * remaining.max(0.0)
}

/// Count overhang fields and holes in one pass.
///
/// Overhang fields are empty cells with at least one filled cell above in the same column.
/// Holes are overhang fields not reachable from top-row air via 4-neighbor flood-fill.
pub fn count_overhang_fields_and_holes(env: &TetrisEnv) -> (u32, u32) {
    if let Some(cached) = env.get_cached_overhang_fields_and_holes() {
        return cached;
    }

    let board_key = build_board_key(env);
    if let Some(key) = board_key {
        if let Some(cached) = get_cached_board_analysis(key) {
            env.set_cached_overhang_fields_and_holes(cached);
            return cached;
        }
    }

    let board = env.board_cells();
    let width = env.width;
    let height = env.height;
    let cell_count = width * height;

    let mut reachable = vec![false; cell_count];
    let mut queue = vec![0usize; cell_count];
    let mut queue_head = 0usize;
    let mut queue_tail = 0usize;

    // Multi-source flood-fill from all top-row empty cells.
    for x in 0..width {
        let idx = x;
        if board[idx] != 0 || reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        queue[queue_tail] = idx;
        queue_tail += 1;
    }

    while queue_head < queue_tail {
        let idx = queue[queue_head];
        queue_head += 1;
        let x = idx % width;
        let y = idx / width;

        if y > 0 {
            let up = idx - width;
            if board[up] == 0 && !reachable[up] {
                reachable[up] = true;
                queue[queue_tail] = up;
                queue_tail += 1;
            }
        }
        if y + 1 < height {
            let down = idx + width;
            if board[down] == 0 && !reachable[down] {
                reachable[down] = true;
                queue[queue_tail] = down;
                queue_tail += 1;
            }
        }
        if x > 0 {
            let left = idx - 1;
            if board[left] == 0 && !reachable[left] {
                reachable[left] = true;
                queue[queue_tail] = left;
                queue_tail += 1;
            }
        }
        if x + 1 < width {
            let right = idx + 1;
            if board[right] == 0 && !reachable[right] {
                reachable[right] = true;
                queue[queue_tail] = right;
                queue_tail += 1;
            }
        }
    }

    let mut overhang_fields: u32 = 0;
    let mut holes: u32 = 0;
    for x in 0..env.width {
        let mut seen_filled = false;
        for y in 0..env.height {
            let idx = y * env.width + x;
            let cell = board[idx];
            if cell != 0 {
                seen_filled = true;
                continue;
            }
            if seen_filled {
                overhang_fields += 1;
                if !reachable[idx] {
                    holes += 1;
                }
            }
        }
    }

    let result = (overhang_fields, holes);
    env.set_cached_overhang_fields_and_holes(result);
    if let Some(key) = board_key {
        insert_cached_board_analysis(key, result);
    }
    result
}

/// Count overhang fields (empty cells with at least one filled cell above in the same column).
pub fn count_overhang_fields(env: &TetrisEnv) -> u32 {
    count_overhang_fields_and_holes(env).0
}

pub fn normalize_overhang_fields(overhang_fields: u32) -> f32 {
    overhang_fields as f32 / OVERHANG_NORMALIZATION_DENOMINATOR
}

/// Compute normalized overhang penalty magnitude from raw overhang count.
pub fn compute_overhang_penalty(overhang_fields: u32, overhang_penalty_weight: f32) -> f32 {
    normalize_overhang_fields(overhang_fields) * overhang_penalty_weight
}

/// Compute terrain bumpiness as the sum of squared adjacent column-height deltas.
/// For heights h[0..W-1], bumpiness = Σ_{i=0..W-2} (h[i] - h[i+1])^2.
pub fn compute_bumpiness(column_heights: &[u8]) -> u32 {
    if column_heights.len() < 2 {
        return 0;
    }

    let mut bumpiness: u32 = 0;
    for i in 0..(column_heights.len() - 1) {
        let delta = column_heights[i] as i32 - column_heights[i + 1] as i32;
        bumpiness += (delta * delta) as u32;
    }
    bumpiness
}

pub fn normalize_column_heights(column_heights: &[u8], board_height: usize) -> Vec<f32> {
    let denominator = board_height as f32;
    column_heights
        .iter()
        .map(|&height| height as f32 / denominator)
        .collect()
}

pub fn normalize_row_fill_counts(row_fill_counts: &[u8], board_width: usize) -> Vec<f32> {
    let denominator = board_width as f32;
    row_fill_counts
        .iter()
        .map(|&count| count as f32 / denominator)
        .collect()
}

pub fn normalize_total_blocks(total_blocks: u32, board_width: usize, board_height: usize) -> f32 {
    let denominator = (board_width * board_height) as f32;
    total_blocks as f32 / denominator
}

pub fn normalize_bumpiness(raw_bumpiness: u32, board_width: usize, board_height: usize) -> f32 {
    if board_width < 2 {
        return 0.0;
    }
    let max_bumpiness = ((board_width - 1) * board_height * board_height) as f32;
    raw_bumpiness as f32 / max_bumpiness
}

pub fn normalize_holes(holes: u32, board_width: usize, board_height: usize) -> f32 {
    if board_height < 2 || board_width == 0 {
        return 0.0;
    }
    let max_holes = (board_width * (board_height - 1)) as f32;
    holes as f32 / max_holes
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
    fn test_count_holes_ignores_reachable_overhang_air() {
        let mut env = TetrisEnv::with_seed(3, 4, 3);
        let board = vec![vec![0, 1, 0], vec![0, 0, 0], vec![1, 0, 1], vec![1, 1, 1]];
        env.set_board(board).expect("set_board should succeed");

        let (overhang, holes) = count_overhang_fields_and_holes(&env);
        assert_eq!(overhang, 2);
        assert_eq!(holes, 0);
    }

    #[test]
    fn test_count_holes_counts_sealed_overhang_cells() {
        let mut env = TetrisEnv::with_seed(3, 4, 4);
        let board = vec![vec![0, 1, 0], vec![1, 0, 1], vec![1, 0, 1], vec![1, 1, 1]];
        env.set_board(board).expect("set_board should succeed");

        let (overhang, holes) = count_overhang_fields_and_holes(&env);
        assert_eq!(overhang, 2);
        assert_eq!(holes, 2);
    }

    #[test]
    fn test_compute_overhang_penalty_uses_fixed_normalization() {
        let penalty = compute_overhang_penalty(95, 2.0);
        // 95 / 190 = 0.5
        assert!((penalty - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_bumpiness_squared_adjacent_deltas() {
        let column_heights = vec![0, 2, 5, 1];
        // (0-2)^2 + (2-5)^2 + (5-1)^2 = 4 + 9 + 16 = 29
        assert_eq!(compute_bumpiness(&column_heights), 29);
    }

    #[test]
    fn test_compute_bumpiness_short_inputs_are_zero() {
        assert_eq!(compute_bumpiness(&[]), 0);
        assert_eq!(compute_bumpiness(&[3]), 0);
    }

    #[test]
    fn test_normalize_total_blocks_on_standard_board() {
        let normalized = normalize_total_blocks(50, 10, 20);
        assert!((normalized - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_column_heights_divides_by_board_height() {
        let normalized = normalize_column_heights(&[0, 10, 20], 20);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_normalize_row_fill_counts_divides_by_board_width() {
        let normalized = normalize_row_fill_counts(&[0, 5, 10], 10);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_normalize_bumpiness_uses_board_maximum() {
        let normalized = normalize_bumpiness(1800, 10, 20);
        // Max is (10 - 1) * 20^2 = 3600
        assert!((normalized - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_holes_on_standard_board() {
        let normalized = normalize_holes(95, 10, 20);
        // Max is 10 * (20 - 1) = 190
        assert!((normalized - 0.5).abs() < 1e-6);
    }
}
