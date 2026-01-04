//! Game constants for Tetris
//!
//! Centralized constants to avoid hardcoding values throughout the codebase.

/// Standard Tetris board width
pub const BOARD_WIDTH: usize = 10;

/// Standard Tetris board height
pub const BOARD_HEIGHT: usize = 20;

/// Number of pieces visible in the preview queue
pub const QUEUE_SIZE: usize = 5;

/// Default lock delay in milliseconds
pub const DEFAULT_LOCK_DELAY_MS: u32 = 500;

/// Default number of moves/rotations allowed during lock delay
pub const DEFAULT_LOCK_MOVES: u32 = 15;
