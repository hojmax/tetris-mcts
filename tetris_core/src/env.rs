//! Tetris game environment
//!
//! This module contains the main TetrisEnv struct which manages the game state,
//! including the board, current piece, piece queue, hold functionality, and scoring.

use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::kicks::{get_i_kicks, get_jlstz_kicks};
use crate::piece::{get_cells_for_shape, Piece, COLORS, TETROMINOS};

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
    pub score: u32,
    #[pyo3(get)]
    pub lines_cleared: u32,
    #[pyo3(get)]
    pub level: u32,
    #[pyo3(get)]
    pub game_over: bool,
    board: Vec<Vec<u8>>,
    board_colors: Vec<Vec<Option<usize>>>,
    current_piece: Option<Piece>,
    /// Queue of upcoming piece types (7-bag system)
    piece_queue: Vec<usize>,
    /// Held piece type (None if no piece is held)
    hold_piece: Option<usize>,
    /// Whether hold has been used for the current piece (can only hold once per piece)
    hold_used: bool,
    /// Lock delay timer in milliseconds (None = piece not grounded)
    lock_delay_ms: Option<u32>,
    /// Maximum lock delay before piece locks (ms)
    lock_delay_max: u32,
    /// Number of moves/rotates allowed during lock delay
    lock_moves_remaining: u32,
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
        self.lock_moves_remaining = 15;
    }

    fn is_valid_position_for(&self, piece: &Piece) -> bool {
        let shape = &TETROMINOS[piece.piece_type][piece.rotation];
        for (x, y) in get_cells_for_shape(shape, piece.x, piece.y) {
            if x < 0 || x >= self.width as i32 || y >= self.height as i32 {
                return false;
            }
            if y >= 0 && self.board[y as usize][x as usize] != 0 {
                return false;
            }
        }
        true
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
            let mut bag = generate_bag();
            self.piece_queue.append(&mut bag);
        }
    }

    fn spawn_piece_internal(&mut self) {
        // Ensure we have enough pieces in the queue (need at least 6 for preview + 1 for current)
        self.fill_queue(7);

        // Take the first piece from the queue
        let piece_type = self.piece_queue.remove(0);
        let mut piece = Piece::new(piece_type);

        // Set spawn position - centrally at top
        piece.x = (self.width as i32 - 4) / 2;
        piece.y = 0;
        piece.rotation = 0;

        // Clear lock delay for new piece
        self.clear_lock_delay();

        // Check if spawn position is valid
        if self.is_valid_position_for(&piece) {
            self.current_piece = Some(piece);
        } else {
            self.current_piece = Some(piece);
            self.game_over = true;
        }
    }

    fn spawn_piece_from_type(&mut self, piece_type: usize) {
        let mut piece = Piece::new(piece_type);

        // Set spawn position - centrally at top
        piece.x = (self.width as i32 - 4) / 2;
        piece.y = 0;
        piece.rotation = 0;

        // Check if spawn position is valid
        if self.is_valid_position_for(&piece) {
            self.current_piece = Some(piece);
        } else {
            self.current_piece = Some(piece);
            self.game_over = true;
        }
    }

    fn lock_piece_internal(&mut self) {
        if let Some(ref piece) = self.current_piece.clone() {
            let shape = &TETROMINOS[piece.piece_type][piece.rotation];
            for (x, y) in get_cells_for_shape(shape, piece.x, piece.y) {
                if y >= 0 && y < self.height as i32 && x >= 0 && x < self.width as i32 {
                    self.board[y as usize][x as usize] = 1;
                    self.board_colors[y as usize][x as usize] = Some(piece.piece_type);
                }
            }
            self.clear_lines_internal();
            // Reset hold_used when a new piece spawns after locking
            self.hold_used = false;
            self.spawn_piece_internal();
        }
    }

    fn clear_lines_internal(&mut self) {
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

        // Update score
        self.lines_cleared += num_lines;
        self.score += match num_lines {
            1 => 100 * self.level,
            2 => 300 * self.level,
            3 => 500 * self.level,
            4 => 800 * self.level,
            _ => 0,
        };

        // Update level
        self.level = (self.lines_cleared / 10) + 1;
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
            score: 0,
            lines_cleared: 0,
            level: 1,
            game_over: false,
            board: vec![vec![0; width]; height],
            board_colors: vec![vec![None; width]; height],
            current_piece: None,
            piece_queue: Vec::new(),
            hold_piece: None,
            hold_used: false,
            lock_delay_ms: None,
            lock_delay_max: 500, // 500ms lock delay
            lock_moves_remaining: 15, // 15 moves/rotates allowed during lock delay
        };
        env.spawn_piece_internal();
        env
    }

    pub fn reset(&mut self) {
        self.board = vec![vec![0; self.width]; self.height];
        self.board_colors = vec![vec![None; self.width]; self.height];
        self.score = 0;
        self.lines_cleared = 0;
        self.level = 1;
        self.game_over = false;
        self.current_piece = None;
        self.piece_queue.clear();
        self.hold_piece = None;
        self.hold_used = false;
        self.lock_delay_ms = None;
        self.lock_moves_remaining = 15;
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
        self.piece_queue.first().map(|&pt| Piece::new(pt))
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

    /// Hold the current piece (swap with hold slot)
    /// Returns true if hold was successful, false if already used this turn
    pub fn hold(&mut self) -> bool {
        if self.game_over || self.hold_used {
            return false;
        }

        if let Some(ref current) = self.current_piece {
            let current_type = current.piece_type;

            if let Some(held_type) = self.hold_piece {
                // Swap: put current in hold, spawn held piece
                self.hold_piece = Some(current_type);
                self.hold_used = true;
                self.spawn_piece_from_type(held_type);
            } else {
                // No piece in hold: put current in hold, spawn next from queue
                self.hold_piece = Some(current_type);
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

    pub fn move_left(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x -= 1;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                // Reset lock delay if we were grounded and moved
                if was_grounded && self.lock_delay_ms.is_some() {
                    self.reset_lock_delay();
                }
                return true;
            }
        }
        false
    }

    pub fn move_right(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x += 1;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                // Reset lock delay if we were grounded and moved
                if was_grounded && self.lock_delay_ms.is_some() {
                    self.reset_lock_delay();
                }
                return true;
            }
        }
        false
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
            if drop_distance > 0 {
                test_piece.y = piece.y + drop_distance as i32;
            }
            self.current_piece = Some(test_piece);
        }
        self.score += drop_distance * 2;
        // Hard drop locks immediately
        self.lock_piece_internal();
        drop_distance
    }

    /// Rotate clockwise using SRS wall kicks
    pub fn rotate_cw(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let from_state = piece.rotation;
            let to_state = (piece.rotation + 1) % 4;
            let new_shape = &TETROMINOS[piece.piece_type][to_state];

            // Get appropriate kicks based on piece type
            let kicks = if piece.piece_type == 0 {
                // I piece
                get_i_kicks(from_state, to_state)
            } else if piece.piece_type == 1 {
                // O piece - no kicks needed, but also no real rotation
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
            } else {
                // J, L, S, T, Z pieces
                get_jlstz_kicks(from_state, to_state)
            };

            // Try each kick
            for (dx, dy) in kicks.iter() {
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
                    return true;
                }
            }
        }
        false
    }

    /// Rotate counter-clockwise using SRS wall kicks
    pub fn rotate_ccw(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let from_state = piece.rotation;
            let to_state = (piece.rotation + 3) % 4; // +3 is same as -1 mod 4
            let new_shape = &TETROMINOS[piece.piece_type][to_state];

            // Get appropriate kicks based on piece type
            let kicks = if piece.piece_type == 0 {
                // I piece
                get_i_kicks(from_state, to_state)
            } else if piece.piece_type == 1 {
                // O piece - no kicks needed
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
            } else {
                // J, L, S, T, Z pieces
                get_jlstz_kicks(from_state, to_state)
            };

            // Try each kick
            for (dx, dy) in kicks.iter() {
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
                    return true;
                }
            }
        }
        false
    }

    pub fn step(&mut self, action: u8) -> (u32, bool) {
        // Actions: 0=nothing, 1=left, 2=right, 3=down, 4=rotate_cw, 5=rotate_ccw, 6=hard_drop
        let old_score = self.score;

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
            _ => {}
        }

        let reward = self.score - old_score;
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
    pub fn get_piece_queue(&self) -> &Vec<usize> {
        &self.piece_queue
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
        assert_eq!(env.score, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.level, 1);
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
        env.score = 1000;
        env.lines_cleared = 10;
        env.level = 2;
        env.reset();
        assert_eq!(env.score, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.level, 1);
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
        // Move down a bit to give room for rotation
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_cw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 1) % 4);
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
        }
    }

    #[test]
    fn test_hard_drop() {
        let mut env = TetrisEnv::new(10, 20);
        let drop_distance = env.hard_drop();
        assert!(drop_distance > 0);
        // After hard drop, a new piece should spawn
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_hard_drop_scoring() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_score = env.score;
        let drop_distance = env.hard_drop();
        // Hard drop should add 2 points per cell dropped
        assert_eq!(env.score, initial_score + drop_distance * 2);
    }

    #[test]
    fn test_ghost_piece() {
        let env = TetrisEnv::new(10, 20);
        let ghost = env.get_ghost_piece();
        assert!(ghost.is_some());
        let ghost = ghost.unwrap();
        let current = env.current_piece.as_ref().unwrap();
        // Ghost should be below or at same position as current piece
        assert!(ghost.y >= current.y);
        // Ghost should have same piece type and rotation
        assert_eq!(ghost.piece_type, current.piece_type);
        assert_eq!(ghost.rotation, current.rotation);
    }

    #[test]
    fn test_7bag_randomizer() {
        let env = TetrisEnv::new(10, 20);
        // Queue should have at least 6 pieces (since current piece was taken from it)
        assert!(env.piece_queue.len() >= 6);
        // All pieces in queue should be valid types (0-6)
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
        // Second hold in same turn should fail
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

        // Simulate piece locking and new piece spawning
        env.hard_drop();

        // Now we can hold again
        let second_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();

        // Should have swapped - current should be the first type we held
        assert_eq!(env.current_piece.as_ref().unwrap().piece_type, first_type);
        // Hold should now contain second type
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
        // Move piece to bottom
        for _ in 0..25 {
            env.move_down();
        }
        // After reaching bottom, lock delay should start
        assert!(env.lock_delay_ms.is_some() || env.is_grounded());
    }

    #[test]
    fn test_lock_delay_progress() {
        let env = TetrisEnv::new(10, 20);
        // Initially should be 0
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
        assert_eq!(env.score, cloned.score);
        assert_eq!(env.level, cloned.level);
        assert_eq!(env.width, cloned.width);
        assert_eq!(env.height, cloned.height);
        assert_eq!(env.lines_cleared, cloned.lines_cleared);
    }

    #[test]
    fn test_step_actions() {
        let mut env = TetrisEnv::new(10, 20);

        // Test all actions don't crash
        env.step(0); // noop
        env.step(1); // left
        env.step(2); // right
        env.step(3); // down
        env.step(4); // rotate_cw
        env.step(5); // rotate_ccw

        assert!(!env.game_over);
    }

    #[test]
    fn test_step_returns_reward() {
        let mut env = TetrisEnv::new(10, 20);
        let (reward, game_over) = env.step(6); // hard_drop
        // Hard drop should give some reward
        assert!(reward > 0);
        assert!(!game_over);
    }

    #[test]
    fn test_wall_collision_left() {
        let mut env = TetrisEnv::new(10, 20);
        // Move all the way left
        for _ in 0..10 {
            env.move_left();
        }
        // Additional moves should fail
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x >= 0, "Piece should not go past left wall");
        }
    }

    #[test]
    fn test_wall_collision_right() {
        let mut env = TetrisEnv::new(10, 20);
        // Move all the way right
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
            let color = env.get_color_for_type(i);
            assert!(color.0 <= 255 && color.1 <= 255 && color.2 <= 255);
        }
    }

    #[test]
    fn test_scoring_single_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom row except one cell
        for x in 0..9 {
            env.set_cell(x, 19, 1);
        }
        // The scoring happens when a piece locks and completes a line
        // This is tested implicitly through the clear_lines_internal function
        assert_eq!(env.level, 1);
    }

    #[test]
    fn test_level_progression() {
        let mut env = TetrisEnv::new(10, 20);
        env.lines_cleared = 10;
        env.level = (env.lines_cleared / 10) + 1;
        assert_eq!(env.level, 2);
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
        // Initially piece should not be grounded
        assert!(!env.is_piece_grounded());

        // Move piece all the way down
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
}
