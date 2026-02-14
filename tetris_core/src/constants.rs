//! Game constants for Tetris
//!
//! Centralized constants to avoid hardcoding values throughout the codebase.

/// Standard Tetris board width
pub const BOARD_WIDTH: usize = 10;

/// Standard Tetris board height
pub const BOARD_HEIGHT: usize = 20;

/// Number of pieces visible in the preview queue
pub const QUEUE_SIZE: usize = 5;

/// Number of tetromino piece types
pub const NUM_PIECE_TYPES: usize = 7;

/// Number of occupied cells in any tetromino
pub const MAX_PIECE_CELLS: usize = 4;

/// Combo normalization divisor (combo >= this value maps to 1.0).
/// Chosen to match the combo-attack table saturation point.
pub const COMBO_NORMALIZATION_MAX: u32 = 12;

/// Auxiliary feature vector size for NN input:
/// current piece (7) + hold piece (8) + hold available (1) + queue (35) + placement count (1)
/// + combo (1) + back-to-back (1) + hidden-piece distribution (7)
/// + column heights (10) + max column height (1) + min column height (1)
/// + row fill counts (20) + total blocks (1) + bumpiness (1)
/// + holes (1) + overhang fields (1).
pub const AUX_FEATURES: usize = NUM_PIECE_TYPES
    + (NUM_PIECE_TYPES + 1)
    + 1
    + (QUEUE_SIZE * NUM_PIECE_TYPES)
    + 1
    + 1
    + 1
    + NUM_PIECE_TYPES
    + BOARD_WIDTH
    + 1
    + 1
    + BOARD_HEIGHT
    + 1
    + 1
    + 1
    + 1;

/// Default lock delay in milliseconds
pub const DEFAULT_LOCK_DELAY_MS: u32 = 500;

/// Default number of moves/rotations allowed during lock delay
pub const DEFAULT_LOCK_MOVES: u32 = 15;

/// Max number of placement-cache entries in each worker thread's global cache.
pub const PLACEMENT_CACHE_MAX_ENTRIES: usize = 10_000;

/// Max number of board-analysis entries in each worker thread's global cache.
pub const BOARD_ANALYSIS_CACHE_MAX_ENTRIES: usize = 500_000;

/// Max number of board-embedding entries in Rust NN inference cache.
pub const BOARD_CACHE_MAX_ENTRIES: usize = 1_000_000;

/// Piece type indices
pub const I_PIECE: usize = 0;
pub const O_PIECE: usize = 1;
pub const T_PIECE: usize = 2;
pub const S_PIECE: usize = 3;
pub const Z_PIECE: usize = 4;
pub const J_PIECE: usize = 5;
pub const L_PIECE: usize = 6;
