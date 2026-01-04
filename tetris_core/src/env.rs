//! Tetris game environment
//!
//! This module contains the main TetrisEnv struct which manages the game state,
//! including the board, current piece, piece queue, hold functionality, and scoring.

use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::VecDeque;

use crate::constants::{DEFAULT_LOCK_DELAY_MS, DEFAULT_LOCK_MOVES, T_PIECE};
use crate::kicks::get_kicks_for_piece;
use crate::piece::{get_cells_for_shape, Piece, COLORS, TETROMINOS};
use crate::scoring::{
    calculate_attack, combo_attack, determine_clear_type, AttackResult,
    BACK_TO_BACK_BONUS, PERFECT_CLEAR_ATTACK,
};

/// Generate a new shuffled bag of 7 pieces (7-bag randomizer)
pub fn generate_bag() -> Vec<usize> {
    let mut bag: Vec<usize> = (0..7).collect();
    bag.shuffle(&mut thread_rng());
    bag
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
    /// Total attack/lines sent (replaces old score)
    #[pyo3(get)]
    pub attack: u32,
    /// Current combo count (resets when no lines cleared)
    #[pyo3(get)]
    pub combo: u32,
    /// Whether back-to-back is active
    #[pyo3(get)]
    pub back_to_back: bool,
    board: Vec<Vec<u8>>,
    board_colors: Vec<Vec<Option<usize>>>,
    current_piece: Option<Piece>,
    /// Queue of upcoming piece types (7-bag system)
    piece_queue: VecDeque<usize>,
    /// Held piece type (None if no piece is held)
    hold_piece: Option<usize>,
    /// The bag position of the held piece (for precise 7-bag tracking)
    hold_piece_bag_position: Option<u32>,
    /// Whether hold has been used for the current piece (can only hold once per piece)
    hold_used: bool,
    /// Lock delay timer in milliseconds (None = piece not grounded)
    lock_delay_ms: Option<u32>,
    /// Maximum lock delay before piece locks (ms)
    lock_delay_max: u32,
    /// Number of moves/rotates allowed during lock delay
    lock_moves_remaining: u32,
    /// Whether the last move was a rotation (for T-spin detection)
    last_move_was_rotation: bool,
    /// Which kick was used for the last rotation (0 = no kick, 1-4 = kick index)
    last_kick_index: usize,
    /// Last attack result from line clear
    last_attack_result: Option<AttackResult>,
    /// Total pieces spawned (including current). Used for 7-bag tracking.
    /// This is 1-indexed: after spawning the first piece, pieces_spawned = 1.
    pieces_spawned: u32,
    /// The bag position (0-indexed global position) of the current piece.
    /// For the first piece spawned, this is 0.
    current_piece_bag_position: u32,
}

// Internal helper methods (not exposed to Python)
impl TetrisEnv {
    /// Check if the current piece is grounded (cannot move down)
    fn is_grounded(&self) -> bool {
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.y += 1;
            !self.is_valid_position_for(&test_piece)
        } else {
            false
        }
    }

    /// Reset lock delay timer (called when piece moves/rotates while grounded)
    fn reset_lock_delay(&mut self) {
        if self.lock_moves_remaining > 0 {
            self.lock_delay_ms = Some(0);
            self.lock_moves_remaining -= 1;
        }
    }

    /// Clear lock delay state (called when piece spawns)
    fn clear_lock_delay(&mut self) {
        self.lock_delay_ms = None;
        self.lock_moves_remaining = DEFAULT_LOCK_MOVES;
    }

    fn is_valid_position_for(&self, piece: &Piece) -> bool {
        let shape = &TETROMINOS[piece.piece_type][piece.rotation];
        self.is_valid_position_for_shape(shape, piece.x, piece.y)
    }

    fn is_valid_position_for_shape(&self, shape: &[[u8; 4]; 4], x: i32, y: i32) -> bool {
        for (cx, cy) in get_cells_for_shape(shape, x, y) {
            if cx < 0 || cx >= self.width as i32 || cy >= self.height as i32 {
                return false;
            }
            if cy >= 0 && self.board[cy as usize][cx as usize] != 0 {
                return false;
            }
        }
        true
    }

    /// Ensure the piece queue has at least `count` pieces
    fn fill_queue(&mut self, count: usize) {
        while self.piece_queue.len() < count {
            let bag = generate_bag();
            self.piece_queue.extend(bag);
        }
    }

    fn spawn_piece_internal(&mut self) {
        // Ensure we have enough pieces in the queue (need at least 6 for preview + 1 for current)
        self.fill_queue(7);

        // Take the first piece from the queue (O(1) with VecDeque)
        let piece_type = self.piece_queue.pop_front().expect("Queue should not be empty after fill_queue");
        let mut piece = Piece::new(piece_type);

        // Set spawn position - centrally at top
        piece.x = (self.width as i32 - 4) / 2;
        piece.y = 0;
        piece.rotation = 0;

        // Track bag position BEFORE incrementing pieces_spawned
        // The piece we're spawning was at queue position 0, which is global position pieces_spawned
        self.current_piece_bag_position = self.pieces_spawned;

        // Track pieces spawned for 7-bag state
        self.pieces_spawned += 1;

        // Clear lock delay for new piece
        self.clear_lock_delay();

        // Reset rotation tracking for new piece
        self.last_move_was_rotation = false;
        self.last_kick_index = 0;

        // Set piece and check if spawn position is valid
        let is_valid = self.is_valid_position_for(&piece);
        self.current_piece = Some(piece);
        if !is_valid {
            self.game_over = true;
        }
    }

    fn spawn_piece_from_type(&mut self, piece_type: usize) {
        let mut piece = Piece::new(piece_type);

        // Set spawn position - centrally at top
        piece.x = (self.width as i32 - 4) / 2;
        piece.y = 0;
        piece.rotation = 0;

        // Reset rotation tracking
        self.last_move_was_rotation = false;
        self.last_kick_index = 0;

        // Set piece and check if spawn position is valid
        let is_valid = self.is_valid_position_for(&piece);
        self.current_piece = Some(piece);
        if !is_valid {
            self.game_over = true;
        }
    }

    /// Check if the T piece at the given position has a T-spin
    /// Returns (is_tspin, is_mini)
    fn check_tspin(&self, piece: &Piece) -> (bool, bool) {
        // Only T piece can T-spin
        if piece.piece_type != T_PIECE {
            return (false, false);
        }

        // Must have been a rotation
        if !self.last_move_was_rotation {
            return (false, false);
        }

        // Check the 4 corners around the T piece center
        // T piece center is at (x+1, y+1) in its local coordinate system
        let center_x = piece.x + 1;
        let center_y = piece.y + 1;

        let corners = [
            (center_x - 1, center_y - 1), // Top-left
            (center_x + 1, center_y - 1), // Top-right
            (center_x - 1, center_y + 1), // Bottom-left
            (center_x + 1, center_y + 1), // Bottom-right
        ];

        let mut filled_corners = 0;
        for (cx, cy) in corners.iter() {
            if self.is_cell_filled(*cx, *cy) {
                filled_corners += 1;
            }
        }

        // T-spin requires at least 3 corners filled
        if filled_corners < 3 {
            return (false, false);
        }

        // Determine which corners are "front" based on rotation
        // Front corners are the ones the T is "pointing" towards
        let front_corners = match piece.rotation {
            0 => [(center_x - 1, center_y - 1), (center_x + 1, center_y - 1)], // Pointing up
            1 => [(center_x + 1, center_y - 1), (center_x + 1, center_y + 1)], // Pointing right
            2 => [(center_x - 1, center_y + 1), (center_x + 1, center_y + 1)], // Pointing down
            3 => [(center_x - 1, center_y - 1), (center_x - 1, center_y + 1)], // Pointing left
            _ => return (false, false),
        };

        let front_filled = front_corners
            .iter()
            .filter(|(cx, cy)| self.is_cell_filled(*cx, *cy))
            .count();

        // Full T-spin: both front corners filled, OR used kick 4 (the special T-spin kick)
        // Mini T-spin: only 1 front corner filled (and 3+ total corners)
        if front_filled == 2 || self.last_kick_index == 4 {
            (true, false) // Full T-spin
        } else {
            (true, true) // Mini T-spin
        }
    }

    /// Check if a cell is filled (occupied or out of bounds)
    fn is_cell_filled(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return true; // Out of bounds counts as filled
        }
        self.board[y as usize][x as usize] != 0
    }

    /// Check if the board is completely empty (perfect clear)
    fn is_perfect_clear(&self) -> bool {
        for row in &self.board {
            for &cell in row {
                if cell != 0 {
                    return false;
                }
            }
        }
        true
    }

    fn lock_piece_internal(&mut self) {
        if let Some(piece) = self.current_piece.take() {
            // Check for T-spin before locking
            let (is_tspin, is_mini) = self.check_tspin(&piece);

            let shape = &TETROMINOS[piece.piece_type][piece.rotation];
            for (x, y) in get_cells_for_shape(shape, piece.x, piece.y) {
                if y >= 0 && y < self.height as i32 && x >= 0 && x < self.width as i32 {
                    self.board[y as usize][x as usize] = 1;
                    self.board_colors[y as usize][x as usize] = Some(piece.piece_type);
                }
            }

            // Clear lines and calculate attack
            self.clear_lines_internal(is_tspin, is_mini);

            // Reset hold_used when a new piece spawns after locking
            self.hold_used = false;
            self.spawn_piece_internal();
        }
    }

    fn clear_lines_internal(&mut self, is_tspin: bool, is_mini: bool) {
        let mut lines_to_clear = Vec::new();

        for y in 0..self.height {
            if self.board[y].iter().all(|&cell| cell != 0) {
                lines_to_clear.push(y);
            }
        }

        let num_lines = lines_to_clear.len() as u32;

        // Remove cleared lines
        for &y in lines_to_clear.iter().rev() {
            self.board.remove(y);
            self.board_colors.remove(y);
        }

        // Add new empty lines at top
        for _ in 0..num_lines {
            self.board.insert(0, vec![0; self.width]);
            self.board_colors.insert(0, vec![None; self.width]);
        }

        // Calculate attack
        if num_lines > 0 {
            // Determine clear type
            let clear_type = determine_clear_type(num_lines, is_tspin, is_mini);

            // Check for perfect clear
            let is_pc = self.is_perfect_clear();

            // Calculate attack with current combo and B2B state
            let (attack_value, new_b2b) =
                calculate_attack(clear_type, self.combo, self.back_to_back, is_pc);

            // Build attack result
            let mut result = AttackResult::new();
            result.clear_type = format!("{:?}", clear_type);
            result.lines_cleared = num_lines;
            result.base_attack = clear_type.base_attack();
            result.combo_attack = combo_attack(self.combo);
            result.back_to_back_attack = if self.back_to_back && clear_type.is_difficult() {
                BACK_TO_BACK_BONUS
            } else {
                0
            };
            result.perfect_clear_attack = if is_pc { PERFECT_CLEAR_ATTACK } else { 0 };
            result.total_attack = attack_value;
            result.combo = self.combo + 1;
            result.back_to_back_active = new_b2b;
            result.is_tspin = is_tspin;
            result.is_perfect_clear = is_pc;

            // Update state
            self.attack += attack_value;
            self.combo += 1;
            self.back_to_back = new_b2b;
            self.last_attack_result = Some(result);
        } else {
            // No lines cleared - reset combo
            self.combo = 0;
            self.last_attack_result = None;
        }

        // Update lines cleared
        self.lines_cleared += num_lines;
    }
}

#[pymethods]
impl TetrisEnv {
    #[new]
    #[pyo3(signature = (width=10, height=20))]
    pub fn new(width: usize, height: usize) -> Self {
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
        };
        env.spawn_piece_internal();
        env
    }

    pub fn reset(&mut self) {
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
        self.spawn_piece_internal();
    }

    pub fn get_board(&self) -> Vec<Vec<u8>> {
        self.board.clone()
    }

    pub fn get_board_colors(&self) -> Vec<Vec<Option<usize>>> {
        self.board_colors.clone()
    }

    pub fn get_current_piece(&self) -> Option<Piece> {
        self.current_piece.clone()
    }

    /// Get the next piece (first in queue)
    pub fn get_next_piece(&self) -> Option<Piece> {
        self.piece_queue.front().map(|&pt| Piece::new(pt))
    }

    /// Get multiple next pieces from the queue
    #[pyo3(signature = (count=5))]
    pub fn get_next_pieces(&self, count: usize) -> Vec<Piece> {
        self.piece_queue
            .iter()
            .take(count)
            .map(|&pt| Piece::new(pt))
            .collect()
    }

    /// Get the piece types in the queue (just the indices)
    #[pyo3(signature = (count=5))]
    pub fn get_queue(&self, count: usize) -> Vec<usize> {
        self.piece_queue.iter().take(count).cloned().collect()
    }

    /// Get the held piece (if any)
    pub fn get_hold_piece(&self) -> Option<Piece> {
        self.hold_piece.map(|pt| Piece::new(pt))
    }

    /// Check if hold has been used for the current piece
    pub fn is_hold_used(&self) -> bool {
        self.hold_used
    }

    /// Get the last attack result (from the most recent line clear)
    pub fn get_last_attack_result(&self) -> Option<AttackResult> {
        self.last_attack_result.clone()
    }

    /// Get the number of pieces spawned (for 7-bag tracking)
    pub fn get_pieces_spawned(&self) -> u32 {
        self.pieces_spawned
    }

    /// Get the possible piece types that could appear as the next unseen piece.
    ///
    /// This is used by MCTS for chance nodes. The possible pieces depend on
    /// what's already in the queue (7-bag system).
    ///
    /// Returns a vector of piece type indices (0-6) that could appear next.
    pub fn get_possible_next_pieces(&self) -> Vec<usize> {
        // The queue contains pieces from consecutive bags.
        // To find what pieces could appear at position queue.len(),
        // we need to look at which bag that position falls into and what's already used.
        //
        // Key insight: pieces at positions 0-6 are from bag 0, 7-13 from bag 1, etc.
        // Queue positions are: pieces_spawned, pieces_spawned+1, ..., pieces_spawned+queue_len-1
        // The next piece to be added would be at position: pieces_spawned + queue_len

        let next_position = self.pieces_spawned as usize + self.piece_queue.len();
        let bag_number = next_position / 7;
        let position_in_bag = next_position % 7;

        // If position_in_bag is 0, this is the start of a new bag - all pieces possible
        if position_in_bag == 0 {
            return (0..7).collect();
        }

        // Find which pieces are already used in this bag
        // The bag starts at position: bag_number * 7
        let bag_start = bag_number * 7;
        let bag_end = bag_start + 7;

        // Pieces in current bag so far are those from bag_start to next_position-1
        let mut used_in_bag: Vec<usize> = Vec::new();

        // Check current piece if it's in this bag (using precise bag position)
        let current_bag_pos = self.current_piece_bag_position as usize;
        if current_bag_pos >= bag_start && current_bag_pos < bag_end {
            if let Some(ref piece) = self.current_piece {
                used_in_bag.push(piece.piece_type);
            }
        }

        // Check held piece if it's in this bag (using precise bag position)
        if let (Some(hold_type), Some(hold_bag_pos)) = (self.hold_piece, self.hold_piece_bag_position) {
            let hold_pos = hold_bag_pos as usize;
            if hold_pos >= bag_start && hold_pos < bag_end {
                used_in_bag.push(hold_type);
            }
        }

        // Check queue pieces
        for (i, &piece_type) in self.piece_queue.iter().enumerate() {
            let piece_pos = self.pieces_spawned as usize + i;
            if piece_pos >= bag_start && piece_pos < bag_end {
                used_in_bag.push(piece_type);
            }
        }

        // Return pieces NOT in used_in_bag
        (0..7).filter(|&p| !used_in_bag.contains(&p)).collect()
    }

    /// Push a specific piece type to the end of the queue.
    ///
    /// This is used by MCTS to simulate chance outcomes. After selecting which
    /// piece appears (from get_possible_next_pieces), call this to add it.
    ///
    /// WARNING: This bypasses normal 7-bag generation. Only use for MCTS simulation.
    pub fn push_queue_piece(&mut self, piece_type: usize) {
        if piece_type < 7 {
            self.piece_queue.push_back(piece_type);
        }
    }

    /// Hold the current piece (swap with hold slot)
    /// Returns true if hold was successful, false if already used this turn
    pub fn hold(&mut self) -> bool {
        if self.game_over || self.hold_used {
            return false;
        }

        if let Some(ref current) = self.current_piece {
            let current_type = current.piece_type;
            let current_bag_pos = self.current_piece_bag_position;

            if let Some(held_type) = self.hold_piece {
                // Swap: put current in hold, spawn held piece
                // The held piece's bag position becomes the current piece's bag position
                let held_bag_pos = self.hold_piece_bag_position
                    .expect("hold_piece_bag_position should be set when hold_piece is Some");

                self.hold_piece = Some(current_type);
                self.hold_piece_bag_position = Some(current_bag_pos);
                self.current_piece_bag_position = held_bag_pos;
                self.hold_used = true;
                self.spawn_piece_from_type(held_type);
            } else {
                // No piece in hold: put current in hold, spawn next from queue
                self.hold_piece = Some(current_type);
                self.hold_piece_bag_position = Some(current_bag_pos);
                self.hold_used = true;
                self.spawn_piece_internal();
            }
            return true;
        }
        false
    }

    pub fn get_color_for_type(&self, piece_type: usize) -> (u8, u8, u8) {
        COLORS[piece_type]
    }

    /// Internal horizontal movement logic
    fn move_horizontal(&mut self, dx: i32) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x += dx;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                // Reset lock delay if we were grounded and moved
                if was_grounded && self.lock_delay_ms.is_some() {
                    self.reset_lock_delay();
                }
                // Movement clears rotation flag
                self.last_move_was_rotation = false;
                return true;
            }
        }
        false
    }

    pub fn move_left(&mut self) -> bool {
        self.move_horizontal(-1)
    }

    pub fn move_right(&mut self) -> bool {
        self.move_horizontal(1)
    }

    pub fn move_down(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.y += 1;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                // If we moved down, check if now grounded
                if self.is_grounded() {
                    // Start lock delay if not already started
                    if self.lock_delay_ms.is_none() {
                        self.lock_delay_ms = Some(0);
                    }
                } else {
                    // Not grounded anymore, clear lock delay
                    self.lock_delay_ms = None;
                }
                // Movement clears rotation flag
                self.last_move_was_rotation = false;
                return true;
            }
        }
        // Could not move down - piece is grounded, start lock delay if not started
        if self.lock_delay_ms.is_none() {
            self.lock_delay_ms = Some(0);
        }
        false
    }

    pub fn hard_drop(&mut self) -> u32 {
        if self.game_over {
            return 0;
        }
        let mut drop_distance = 0;
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            while {
                test_piece.y += 1;
                self.is_valid_position_for(&test_piece)
            } {
                drop_distance += 1;
            }
            test_piece.y -= 1; // Go back to last valid position
            // Only clear rotation flag if piece actually moved (preserves T-spin detection for 0-distance drops)
            if drop_distance > 0 {
                self.last_move_was_rotation = false;
            }
            self.current_piece = Some(test_piece);
        }
        // Hard drop locks immediately (no score for drop distance in attack mode)
        self.lock_piece_internal();
        drop_distance
    }

    /// Internal rotation logic using SRS wall kicks
    fn rotate(&mut self, clockwise: bool) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let from_state = piece.rotation;
            let to_state = if clockwise {
                (piece.rotation + 1) % 4
            } else {
                (piece.rotation + 3) % 4 // +3 is same as -1 mod 4
            };
            let new_shape = &TETROMINOS[piece.piece_type][to_state];
            let kicks = get_kicks_for_piece(piece.piece_type, from_state, to_state);

            // Try each kick
            for (kick_idx, (dx, dy)) in kicks.iter().enumerate() {
                let new_x = piece.x + dx;
                let new_y = piece.y + dy;
                if self.is_valid_position_for_shape(new_shape, new_x, new_y) {
                    let mut new_piece = piece.clone();
                    new_piece.x = new_x;
                    new_piece.y = new_y;
                    new_piece.rotation = to_state;
                    self.current_piece = Some(new_piece);
                    // Reset lock delay if we were grounded and rotated
                    if was_grounded && self.lock_delay_ms.is_some() {
                        self.reset_lock_delay();
                    }
                    // Track that last move was a rotation
                    self.last_move_was_rotation = true;
                    self.last_kick_index = kick_idx;
                    return true;
                }
            }
        }
        false
    }

    /// Rotate clockwise using SRS wall kicks
    pub fn rotate_cw(&mut self) -> bool {
        self.rotate(true)
    }

    /// Rotate counter-clockwise using SRS wall kicks
    pub fn rotate_ccw(&mut self) -> bool {
        self.rotate(false)
    }

    pub fn step(&mut self, action: u8) -> (u32, bool) {
        // Actions: 0=nothing, 1=left, 2=right, 3=down, 4=rotate_cw, 5=rotate_ccw, 6=hard_drop, 7=hold
        let old_attack = self.attack;

        match action {
            1 => {
                self.move_left();
            }
            2 => {
                self.move_right();
            }
            3 => {
                self.move_down();
            }
            4 => {
                self.rotate_cw();
            }
            5 => {
                self.rotate_ccw();
            }
            6 => {
                self.hard_drop();
            }
            7 => {
                self.hold();
            }
            _ => {}
        }

        let reward = self.attack - old_attack;
        (reward, self.game_over)
    }

    pub fn tick(&mut self) -> bool {
        self.move_down()
    }

    /// Update lock delay timer. Call this every frame with delta time in ms.
    /// Returns true if piece was locked.
    pub fn update_lock_delay(&mut self, delta_ms: u32) -> bool {
        if self.game_over {
            return false;
        }

        // Check if piece is grounded
        if self.is_grounded() {
            // Start or update lock delay timer
            if let Some(current_delay) = self.lock_delay_ms {
                let new_delay = current_delay + delta_ms;
                if new_delay >= self.lock_delay_max || self.lock_moves_remaining == 0 {
                    // Lock the piece
                    self.lock_piece_internal();
                    return true;
                }
                self.lock_delay_ms = Some(new_delay);
            } else {
                self.lock_delay_ms = Some(0);
            }
        } else {
            // Not grounded, clear lock delay
            self.lock_delay_ms = None;
        }
        false
    }

    /// Check if piece is currently grounded
    pub fn is_piece_grounded(&self) -> bool {
        self.is_grounded()
    }

    /// Get current lock delay progress (0.0 to 1.0)
    pub fn get_lock_delay_progress(&self) -> f32 {
        if let Some(delay) = self.lock_delay_ms {
            (delay as f32) / (self.lock_delay_max as f32)
        } else {
            0.0
        }
    }

    pub fn get_ghost_piece(&self) -> Option<Piece> {
        if let Some(ref piece) = self.current_piece {
            let mut ghost = piece.clone();
            let shape = &TETROMINOS[ghost.piece_type][ghost.rotation];
            while self.is_valid_position_for_shape(shape, ghost.x, ghost.y + 1) {
                ghost.y += 1;
            }
            Some(ghost)
        } else {
            None
        }
    }

    pub fn clone_state(&self) -> TetrisEnv {
        self.clone()
    }

    /// Directly place the current piece at the specified position and lock it.
    ///
    /// This is more efficient than stepping through individual moves when you
    /// already know the final placement from get_possible_placements().
    ///
    /// Args:
    ///     x: The x position (column) for the piece
    ///     y: The y position (row) for the piece
    ///     rotation: The rotation state (0-3)
    ///
    /// Returns:
    ///     The attack gained from this placement (including line clears)
    ///
    /// Note: For proper T-spin detection including mini vs proper distinction,
    /// use execute_placement() with the full Placement object instead.
    pub fn place_piece(&mut self, x: i32, y: i32, rotation: usize) -> u32 {
        // Simple placement without T-spin detection
        self.place_piece_internal_with_kick(x, y, rotation, false, 0)
    }

    /// Execute a placement from get_possible_placements() with full T-spin detection.
    ///
    /// This uses the kick index stored in the Placement (computed during move generation)
    /// for accurate T-spin detection including mini vs proper distinction.
    ///
    /// Args:
    ///     placement: A Placement object from get_possible_placements()
    ///
    /// Returns:
    ///     The attack gained from this placement (including line clears)
    pub fn execute_placement(&mut self, placement: &crate::moves::Placement) -> u32 {
        let x = placement.piece.x;
        let y = placement.piece.y;
        let rotation = placement.piece.rotation;

        // Use the actual kick index and rotation info from the Placement
        self.place_piece_internal_with_kick(
            x, y, rotation,
            placement.last_move_was_rotation,
            placement.last_kick_index,
        )
    }

    /// Internal placement logic with explicit T-spin detection info
    fn place_piece_internal_with_kick(
        &mut self,
        x: i32, y: i32, rotation: usize,
        was_rotation: bool,
        kick_index: usize,
    ) -> u32 {
        if self.game_over {
            return 0;
        }

        if let Some(ref piece) = self.current_piece {
            debug_assert!(rotation < 4, "Invalid rotation: {}", rotation);

            let piece_type = piece.piece_type;
            let shape = &TETROMINOS[piece_type][rotation];

            // Verify the position is valid
            if !self.is_valid_position_for_shape(shape, x, y) {
                return 0;
            }

            // Set the piece to the target position
            let mut new_piece = piece.clone();
            new_piece.x = x;
            new_piece.y = y;
            new_piece.rotation = rotation;
            self.current_piece = Some(new_piece);

            // Use the actual kick info from move generation
            self.last_move_was_rotation = was_rotation;
            self.last_kick_index = kick_index;

            // Lock the piece
            let old_attack = self.attack;
            self.lock_piece_internal();
            self.attack - old_attack
        } else {
            0
        }
    }

    /// Set the current piece to a specific type.
    ///
    /// This is used by MCTS to explore different possible next pieces
    /// at chance nodes. The piece spawns at the standard spawn position.
    ///
    /// Args:
    ///     piece_type: The piece type (0-6: I, O, T, S, Z, J, L)
    pub fn set_current_piece_type(&mut self, piece_type: usize) {
        if piece_type < 7 && !self.game_over {
            let spawn_x = (self.width as i32 - 4) / 2;
            let spawn_y = 0;
            self.current_piece = Some(Piece {
                piece_type,
                x: spawn_x,
                y: spawn_y,
                rotation: 0,
            });
        }
    }

    /// Get all possible placements for the current piece
    ///
    /// Returns a list of Placement objects, each containing:
    /// - piece: The final piece position after hard drop
    /// - moves: The sequence of action codes to reach this placement
    /// - column: The x position (column) of the placement
    /// - rotation: The rotation state (0-3)
    ///
    /// The move sequence uses these action codes:
    /// 1=left, 2=right, 3=down, 4=rotate_cw, 5=rotate_ccw, 6=hard_drop
    pub fn get_possible_placements(&self) -> Vec<crate::moves::Placement> {
        use crate::moves::{find_all_placements, Board};

        if let Some(ref piece) = self.current_piece {
            let board = Board::new(self.width, self.height, self.board.clone());
            find_all_placements(&board, piece.piece_type, piece.x, piece.y)
        } else {
            Vec::new()
        }
    }

    /// Get all possible placements for a specific piece type
    ///
    /// This allows querying placements for any piece, not just the current one.
    /// Useful for planning ahead with known upcoming pieces.
    pub fn get_placements_for_piece(&self, piece_type: usize) -> Vec<crate::moves::Placement> {
        use crate::moves::{find_all_placements, Board};

        if piece_type >= 7 {
            return Vec::new();
        }

        let board = Board::new(self.width, self.height, self.board.clone());
        let spawn_x = (self.width as i32 - 4) / 2;
        let spawn_y = 0;
        find_all_placements(&board, piece_type, spawn_x, spawn_y)
    }

    /// Get all possible placements for both current piece and after using hold
    ///
    /// Returns a tuple of (current_piece_placements, hold_piece_placements)
    /// If hold has already been used this turn, hold_piece_placements will be empty.
    pub fn get_possible_placements_with_hold(&self) -> (Vec<crate::moves::Placement>, Vec<crate::moves::Placement>) {
        use crate::moves::{find_all_placements_with_hold, Board};

        if self.hold_used {
            // Can't use hold again this turn
            return (self.get_possible_placements(), Vec::new());
        }

        if let Some(ref piece) = self.current_piece {
            let board = Board::new(self.width, self.height, self.board.clone());
            let next_piece = self.piece_queue.front().copied().unwrap_or(0);
            find_all_placements_with_hold(
                &board,
                piece.piece_type,
                self.hold_piece,
                next_piece,
                piece.x,
                piece.y,
            )
        } else {
            (Vec::new(), Vec::new())
        }
    }
}

// Additional methods for testing (not exposed to Python)
impl TetrisEnv {
    /// Set the board state directly (for testing)
    #[cfg(test)]
    pub fn set_board(&mut self, board: Vec<Vec<u8>>) {
        self.board = board;
    }

    /// Set a specific cell on the board (for testing)
    #[cfg(test)]
    pub fn set_cell(&mut self, x: usize, y: usize, value: u8) {
        if y < self.height && x < self.width {
            self.board[y][x] = value;
        }
    }

    /// Get the current piece queue (for testing)
    #[cfg(test)]
    pub fn get_piece_queue(&self) -> &VecDeque<usize> {
        &self.piece_queue
    }

    /// Force set the current piece (for testing)
    #[cfg(test)]
    pub fn set_current_piece(&mut self, piece: Piece) {
        self.current_piece = Some(piece);
    }

    /// Set rotation tracking (for testing)
    #[cfg(test)]
    pub fn set_last_rotation(&mut self, was_rotation: bool, kick_index: usize) {
        self.last_move_was_rotation = was_rotation;
        self.last_kick_index = kick_index;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_creation() {
        let env = TetrisEnv::new(10, 20);
        assert_eq!(env.width, 10);
        assert_eq!(env.height, 20);
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.combo, 0);
        assert!(!env.back_to_back);
        assert!(!env.game_over);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_custom_board_size() {
        let env = TetrisEnv::new(12, 24);
        assert_eq!(env.width, 12);
        assert_eq!(env.height, 24);
    }

    #[test]
    fn test_env_reset() {
        let mut env = TetrisEnv::new(10, 20);
        env.attack = 100;
        env.lines_cleared = 10;
        env.combo = 5;
        env.back_to_back = true;
        env.reset();
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.combo, 0);
        assert!(!env.back_to_back);
        assert!(!env.game_over);
    }

    #[test]
    fn test_move_left() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let moved = env.move_left();
        if moved {
            let new_x = env.current_piece.as_ref().unwrap().x;
            assert_eq!(new_x, initial_x - 1);
        }
    }

    #[test]
    fn test_move_right() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let moved = env.move_right();
        if moved {
            let new_x = env.current_piece.as_ref().unwrap().x;
            assert_eq!(new_x, initial_x + 1);
        }
    }

    #[test]
    fn test_move_down() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let moved = env.move_down();
        if moved {
            let new_y = env.current_piece.as_ref().unwrap().y;
            assert_eq!(new_y, initial_y + 1);
        }
    }

    #[test]
    fn test_rotate_cw() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_cw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 1) % 4);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_rotate_ccw() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_ccw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 3) % 4);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_movement_clears_rotation_flag() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.rotate_cw();
        assert!(env.last_move_was_rotation);
        env.move_left();
        assert!(!env.last_move_was_rotation);
    }

    #[test]
    fn test_hard_drop() {
        let mut env = TetrisEnv::new(10, 20);
        let drop_distance = env.hard_drop();
        assert!(drop_distance > 0);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_ghost_piece() {
        let env = TetrisEnv::new(10, 20);
        let ghost = env.get_ghost_piece();
        assert!(ghost.is_some());
        let ghost = ghost.unwrap();
        let current = env.current_piece.as_ref().unwrap();
        assert!(ghost.y >= current.y);
        assert_eq!(ghost.piece_type, current.piece_type);
        assert_eq!(ghost.rotation, current.rotation);
    }

    #[test]
    fn test_7bag_randomizer() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.piece_queue.len() >= 6);
        for &pt in &env.piece_queue {
            assert!(pt < 7);
        }
    }

    #[test]
    fn test_7bag_contains_all_pieces() {
        let bag = generate_bag();
        assert_eq!(bag.len(), 7);
        let mut sorted = bag.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_hold_piece() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(env.hold_piece.is_none());

        let current_type = env.current_piece.as_ref().unwrap().piece_type;
        let result = env.hold();

        assert!(result);
        assert_eq!(env.hold_piece, Some(current_type));
        assert!(env.hold_used);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_hold_twice_fails() {
        let mut env = TetrisEnv::new(10, 20);
        env.hold();
        let result = env.hold();
        assert!(!result);
    }

    #[test]
    fn test_hold_swap() {
        let mut env = TetrisEnv::new(10, 20);
        let first_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();
        let held_type = env.hold_piece.unwrap();
        assert_eq!(held_type, first_type);

        env.hard_drop();

        let second_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();

        assert_eq!(env.current_piece.as_ref().unwrap().piece_type, first_type);
        assert_eq!(env.hold_piece, Some(second_type));
    }

    #[test]
    fn test_hold_when_game_over_fails() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let result = env.hold();
        assert!(!result);
    }

    #[test]
    fn test_lock_delay_initialization() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.lock_delay_ms.is_none());
        assert_eq!(env.lock_moves_remaining, 15);
    }

    #[test]
    fn test_lock_delay_starts_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..25 {
            env.move_down();
        }
        assert!(env.lock_delay_ms.is_some() || env.is_grounded());
    }

    #[test]
    fn test_lock_delay_progress() {
        let env = TetrisEnv::new(10, 20);
        assert_eq!(env.get_lock_delay_progress(), 0.0);
    }

    #[test]
    fn test_get_next_pieces() {
        let env = TetrisEnv::new(10, 20);
        let next_pieces = env.get_next_pieces(5);
        assert_eq!(next_pieces.len(), 5);
        for piece in next_pieces {
            assert!(piece.piece_type < 7);
        }
    }

    #[test]
    fn test_get_queue() {
        let env = TetrisEnv::new(10, 20);
        let queue = env.get_queue(5);
        assert_eq!(queue.len(), 5);
        for pt in queue {
            assert!(pt < 7);
        }
    }

    #[test]
    fn test_board_initially_empty() {
        let env = TetrisEnv::new(10, 20);
        let board = env.get_board();
        for row in board {
            for cell in row {
                assert_eq!(cell, 0);
            }
        }
    }

    #[test]
    fn test_board_colors_initially_none() {
        let env = TetrisEnv::new(10, 20);
        let colors = env.get_board_colors();
        for row in colors {
            for cell in row {
                assert!(cell.is_none());
            }
        }
    }

    #[test]
    fn test_clone_state() {
        let env = TetrisEnv::new(10, 20);
        let cloned = env.clone_state();
        assert_eq!(env.attack, cloned.attack);
        assert_eq!(env.width, cloned.width);
        assert_eq!(env.height, cloned.height);
        assert_eq!(env.lines_cleared, cloned.lines_cleared);
        assert_eq!(env.combo, cloned.combo);
        assert_eq!(env.back_to_back, cloned.back_to_back);
    }

    #[test]
    fn test_step_actions() {
        let mut env = TetrisEnv::new(10, 20);
        env.step(0); // noop
        env.step(1); // left
        env.step(2); // right
        env.step(3); // down
        env.step(4); // rotate_cw
        env.step(5); // rotate_ccw
        env.step(7); // hold
        assert!(!env.game_over);
    }

    #[test]
    fn test_step_returns_attack() {
        let mut env = TetrisEnv::new(10, 20);
        let (reward, game_over) = env.step(6); // hard_drop
        // May or may not have attack depending on line clears
        assert!(!game_over);
        // reward is the attack gained this step
        assert!(reward >= 0);
    }

    #[test]
    fn test_wall_collision_left() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..10 {
            env.move_left();
        }
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x >= 0, "Piece should not go past left wall");
        }
    }

    #[test]
    fn test_wall_collision_right() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..10 {
            env.move_right();
        }
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x < env.width as i32, "Piece should not go past right wall");
        }
    }

    #[test]
    fn test_tick() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let moved = env.tick();
        if moved {
            let new_y = env.current_piece.as_ref().unwrap().y;
            assert_eq!(new_y, initial_y + 1);
        }
    }

    #[test]
    fn test_get_color_for_type() {
        let env = TetrisEnv::new(10, 20);
        for i in 0..7 {
            let _color = env.get_color_for_type(i);
            // Just ensure it doesn't panic
        }
    }

    #[test]
    fn test_move_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        assert!(!env.move_left());
        assert!(!env.move_right());
        assert!(!env.move_down());
        assert!(!env.rotate_cw());
        assert!(!env.rotate_ccw());
        assert_eq!(env.hard_drop(), 0);
    }

    #[test]
    fn test_update_lock_delay_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let locked = env.update_lock_delay(100);
        assert!(!locked);
    }

    #[test]
    fn test_is_piece_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(!env.is_piece_grounded());

        for _ in 0..25 {
            env.move_down();
        }
        assert!(env.is_piece_grounded());
    }

    #[test]
    fn test_get_next_piece() {
        let env = TetrisEnv::new(10, 20);
        let next = env.get_next_piece();
        assert!(next.is_some());
        assert!(next.unwrap().piece_type < 7);
    }

    #[test]
    fn test_is_hold_used() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(!env.is_hold_used());
        env.hold();
        assert!(env.is_hold_used());
    }

    #[test]
    fn test_get_hold_piece() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(env.get_hold_piece().is_none());
        env.hold();
        assert!(env.get_hold_piece().is_some());
    }

    #[test]
    fn test_board_dimensions() {
        let env = TetrisEnv::new(10, 20);
        let board = env.get_board();
        assert_eq!(board.len(), 20);
        for row in board {
            assert_eq!(row.len(), 10);
        }
    }

    #[test]
    fn test_perfect_clear_detection() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.is_perfect_clear()); // Empty board is perfect clear
    }

    #[test]
    fn test_perfect_clear_with_cells() {
        let mut env = TetrisEnv::new(10, 20);
        env.set_cell(0, 19, 1);
        assert!(!env.is_perfect_clear());
    }

    #[test]
    fn test_tspin_requires_t_piece() {
        let mut env = TetrisEnv::new(10, 20);
        // Create an I piece
        let piece = Piece::with_position(0, 3, 10, 0);
        env.set_current_piece(piece);
        env.set_last_rotation(true, 0);

        let (is_tspin, _) = env.check_tspin(env.current_piece.as_ref().unwrap());
        assert!(!is_tspin);
    }

    #[test]
    fn test_tspin_requires_rotation() {
        let mut env = TetrisEnv::new(10, 20);
        // Create a T piece
        let piece = Piece::with_position(T_PIECE, 3, 10, 0);
        env.set_current_piece(piece);
        env.set_last_rotation(false, 0); // No rotation

        let (is_tspin, _) = env.check_tspin(env.current_piece.as_ref().unwrap());
        assert!(!is_tspin);
    }

    #[test]
    fn test_combo_resets_on_no_clear() {
        let mut env = TetrisEnv::new(10, 20);
        env.combo = 5;
        // Lock a piece without clearing lines
        env.clear_lines_internal(false, false);
        assert_eq!(env.combo, 0);
    }

    #[test]
    fn test_get_last_attack_result_initially_none() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.get_last_attack_result().is_none());
    }
}
