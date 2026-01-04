//! TetrisEnv State Definition
//!
//! Core struct definition and initialization.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;

use crate::constants::{DEFAULT_LOCK_DELAY_MS, DEFAULT_LOCK_MOVES};
use crate::scoring::AttackResult;

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
    pub(crate) board: Vec<Vec<u8>>,
    pub(crate) board_colors: Vec<Vec<Option<usize>>>,
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
            board: vec![vec![0; width]; height],
            board_colors: vec![vec![None; width]; height],
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
        };
        env.spawn_piece_internal();
        env
    }

    /// Reset the game with a specific random seed for reproducibility.
    pub fn reset_internal(&mut self, seed: u64) {
        self.board = vec![vec![0; self.width]; self.height];
        self.board_colors = vec![vec![None; self.width]; self.height];
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
        self.spawn_piece_internal();
    }
}
