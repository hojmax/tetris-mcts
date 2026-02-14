//! TetrisEnv State Definition
//!
//! Core struct definition and initialization.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::constants::{DEFAULT_LOCK_DELAY_MS, DEFAULT_LOCK_MOVES};
use crate::moves::Placement;
use crate::scoring::AttackResult;

#[derive(Clone)]
pub(crate) struct PlacementCache {
    pub placements: Arc<Vec<Placement>>,
    pub action_to_placement_idx: Arc<Vec<Option<usize>>>,
    pub placement_action_indices: Arc<Vec<usize>>,
}

#[pyclass]
#[derive(Clone)]
pub struct TetrisEnv {
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
    #[pyo3(get)]
    pub lines_cleared: u32,
    #[pyo3(get)]
    pub game_over: bool,
    #[pyo3(get)]
    pub attack: u32,
    #[pyo3(get)]
    pub combo: u32,
    #[pyo3(get)]
    pub back_to_back: bool,
    pub(crate) board: Vec<u8>,
    pub(crate) board_piece_types: Vec<Option<usize>>,
    pub(crate) current_piece: Option<crate::piece::Piece>,
    pub(crate) piece_queue: VecDeque<usize>,
    pub(crate) hold_piece: Option<usize>,
    pub(crate) hold_piece_bag_position: Option<u32>,
    pub(crate) hold_used: bool,
    pub(crate) lock_delay_ms: Option<u32>,
    pub(crate) lock_delay_max: u32,
    pub(crate) lock_moves_remaining: u32,
    pub(crate) last_move_was_rotation: bool,
    pub(crate) last_kick_index: usize,
    pub(crate) last_attack_result: Option<AttackResult>,
    pub(crate) pieces_spawned: u32,
    pub(crate) current_piece_bag_position: u32,
    pub(crate) rng: StdRng,
    /// The seed used to initialize this environment's RNG (for determinism tracking)
    pub(crate) seed: u64,
    /// Total number of filled cells on the board. Used for O(1) perfect clear detection.
    pub(crate) total_blocks: u32,
    /// Number of filled cells per row. Used for O(1) line clear detection.
    /// A row is full when `row_fill_counts[y] == width`.
    pub(crate) row_fill_counts: Vec<u8>,
    /// Height per column measured from the bottom (0 for empty, up to board height).
    pub(crate) column_heights: Vec<u8>,
    /// Cached placements for current piece (invalidated when piece or board changes)
    /// Using RefCell for interior mutability to cache with &self
    pub(crate) placements_cache: RefCell<Option<PlacementCache>>,
    /// Cached board diagnostics (overhang_fields, holes), invalidated only when board changes.
    pub(crate) board_analysis_cache: RefCell<Option<(u32, u32)>>,
}

impl TetrisEnv {
    /// Create a new TetrisEnv with a specific random seed.
    pub fn new_with_seed(width: usize, height: usize, seed: u64) -> Self {
        let mut env = TetrisEnv {
            width,
            height,
            attack: 0,
            lines_cleared: 0,
            game_over: false,
            combo: 0,
            back_to_back: false,
            board: vec![0u8; width * height],
            board_piece_types: vec![None; width * height],
            current_piece: None,
            piece_queue: VecDeque::new(),
            hold_piece: None,
            hold_piece_bag_position: None,
            hold_used: false,
            lock_delay_ms: None,
            lock_delay_max: DEFAULT_LOCK_DELAY_MS,
            lock_moves_remaining: DEFAULT_LOCK_MOVES,
            last_move_was_rotation: false,
            last_kick_index: 0,
            last_attack_result: None,
            pieces_spawned: 0,
            current_piece_bag_position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            total_blocks: 0,
            row_fill_counts: vec![0; height],
            column_heights: vec![0; width],
            placements_cache: RefCell::new(None),
            board_analysis_cache: RefCell::new(None),
        };
        env.spawn_piece_internal();
        env
    }

    /// Reset the game with a specific random seed for reproducibility.
    pub fn reset_internal(&mut self, seed: u64) {
        self.board = vec![0u8; self.width * self.height];
        self.board_piece_types = vec![None; self.width * self.height];
        self.attack = 0;
        self.lines_cleared = 0;
        self.game_over = false;
        self.combo = 0;
        self.back_to_back = false;
        self.current_piece = None;
        self.piece_queue.clear();
        self.hold_piece = None;
        self.hold_piece_bag_position = None;
        self.hold_used = false;
        self.lock_delay_ms = None;
        self.lock_moves_remaining = DEFAULT_LOCK_MOVES;
        self.last_move_was_rotation = false;
        self.last_kick_index = 0;
        self.last_attack_result = None;
        self.pieces_spawned = 0;
        self.current_piece_bag_position = 0;
        self.rng = StdRng::seed_from_u64(seed);
        self.seed = seed;
        self.total_blocks = 0;
        self.row_fill_counts = vec![0; self.height];
        self.column_heights = vec![0; self.width];
        *self.placements_cache.borrow_mut() = None;
        *self.board_analysis_cache.borrow_mut() = None;
        self.spawn_piece_internal();
    }

    /// Invalidate the placements cache (call when board or piece changes)
    #[inline]
    pub(crate) fn invalidate_placement_cache(&self) {
        *self.placements_cache.borrow_mut() = None;
    }

    /// Invalidate cached board analysis metrics (overhang fields and holes).
    #[inline]
    pub(crate) fn invalidate_board_analysis_cache(&self) {
        *self.board_analysis_cache.borrow_mut() = None;
    }

    #[inline]
    pub(crate) fn get_cached_overhang_fields_and_holes(&self) -> Option<(u32, u32)> {
        *self.board_analysis_cache.borrow()
    }

    #[inline]
    pub(crate) fn set_cached_overhang_fields_and_holes(&self, value: (u32, u32)) {
        *self.board_analysis_cache.borrow_mut() = Some(value);
    }
}
