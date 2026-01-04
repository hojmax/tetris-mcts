//! Game constants for Tetris
//!
//! Centralized constants to avoid hardcoding values throughout the codebase.

/// Standard Tetris board width
pub const BOARD_WIDTH: usize = 10;

/// Standard Tetris board height
pub const BOARD_HEIGHT: usize = 20;

/// Number of tetromino piece types (I, O, T, S, Z, J, L)
pub const NUM_PIECE_TYPES: usize = 7;

/// Number of pieces visible in the preview queue
pub const QUEUE_SIZE: usize = 5;

/// Number of rotations per piece (0, 90, 180, 270 degrees)
pub const NUM_ROTATIONS: usize = 4;

/// Default lock delay in milliseconds
pub const DEFAULT_LOCK_DELAY_MS: u32 = 500;

/// Default number of moves/rotations allowed during lock delay
pub const DEFAULT_LOCK_MOVES: u32 = 15;
