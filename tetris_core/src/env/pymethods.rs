//! Python-exposed methods for TetrisEnv
//!
//! All #[pymethods] consolidated in one file due to PyO3 limitation.

use pyo3::prelude::*;
use std::collections::HashSet;

use crate::mcts::get_action_space;
use crate::moves::{find_all_placements, find_all_placements_with_hold, Board, Placement};
use crate::piece::{Piece, COLORS};
use crate::scoring::AttackResult;

use super::piece_management::{spawn_x, spawn_y_offset};
use super::TetrisEnv;

#[pymethods]
impl TetrisEnv {
    // === State (state.rs) ===

    #[new]
    #[pyo3(signature = (width=10, height=20))]
    pub fn new(width: usize, height: usize) -> Self {
        use rand::{thread_rng, RngCore};
        Self::new_with_seed(width, height, thread_rng().next_u64())
    }

    #[staticmethod]
    #[pyo3(signature = (width, height, seed))]
    pub fn with_seed(width: usize, height: usize, seed: u64) -> Self {
        Self::new_with_seed(width, height, seed)
    }

    pub fn reset(&mut self) {
        use rand::{thread_rng, RngCore};
        self.reset_internal(thread_rng().next_u64());
    }

    #[pyo3(signature = (seed))]
    pub fn reset_with_seed(&mut self, seed: u64) {
        self.reset_internal(seed);
    }

    pub fn clone_state(&self) -> TetrisEnv {
        self.clone()
    }

    // === Board (board.rs) ===

    pub fn get_board(&self) -> Vec<Vec<u8>> {
        self.board.clone()
    }

    pub fn get_board_colors(&self) -> Vec<Vec<Option<usize>>> {
        self.board_colors.clone()
    }

    // === Piece Management (piece_management.rs) ===

    pub fn get_current_piece(&self) -> Option<Piece> {
        self.current_piece.clone()
    }

    pub fn get_next_piece(&self) -> Option<Piece> {
        self.piece_queue.front().map(|&pt| Piece::new(pt))
    }

    #[pyo3(signature = (count=5))]
    pub fn get_next_pieces(&self, count: usize) -> Vec<Piece> {
        self.piece_queue
            .iter()
            .take(count)
            .map(|&pt| Piece::new(pt))
            .collect()
    }

    #[pyo3(signature = (count=5))]
    pub fn get_queue(&self, count: usize) -> Vec<usize> {
        self.piece_queue.iter().take(count).cloned().collect()
    }

    pub fn get_queue_len(&self) -> usize {
        self.piece_queue.len()
    }

    pub fn get_hold_piece(&self) -> Option<Piece> {
        self.hold_piece.map(|pt| Piece::new(pt))
    }

    pub fn is_hold_used(&self) -> bool {
        self.hold_used
    }

    pub fn get_pieces_spawned(&self) -> u32 {
        self.pieces_spawned
    }

    pub fn get_possible_next_pieces(&self) -> Vec<usize> {
        let next_position = self.pieces_spawned as usize + self.piece_queue.len();
        let bag_number = next_position / 7;
        let position_in_bag = next_position % 7;

        if position_in_bag == 0 {
            return (0..7).collect();
        }

        let bag_start = bag_number * 7;
        let bag_end = bag_start + 7;
        let mut used_in_bag: HashSet<usize> = HashSet::with_capacity(7);

        let current_bag_pos = self.current_piece_bag_position as usize;
        if current_bag_pos >= bag_start && current_bag_pos < bag_end {
            if let Some(ref piece) = self.current_piece {
                used_in_bag.insert(piece.piece_type);
            }
        }

        if let (Some(hold_type), Some(hold_bag_pos)) =
            (self.hold_piece, self.hold_piece_bag_position)
        {
            let hold_pos = hold_bag_pos as usize;
            if hold_pos >= bag_start && hold_pos < bag_end {
                used_in_bag.insert(hold_type);
            }
        }

        for (i, &piece_type) in self.piece_queue.iter().enumerate() {
            let piece_pos = self.pieces_spawned as usize + i;
            if piece_pos >= bag_start && piece_pos < bag_end {
                used_in_bag.insert(piece_type);
            }
        }

        (0..7).filter(|p| !used_in_bag.contains(p)).collect()
    }

    pub fn push_queue_piece(&mut self, piece_type: usize) {
        if piece_type < 7 {
            self.piece_queue.push_back(piece_type);
        }
    }

    /// Truncate the queue to at most `max_len` pieces.
    /// Used by MCTS to limit queue to visible pieces before creating chance nodes.
    pub fn truncate_queue(&mut self, max_len: usize) {
        while self.piece_queue.len() > max_len {
            self.piece_queue.pop_back();
        }
    }

    pub fn hold(&mut self) -> bool {
        if self.game_over || self.hold_used {
            return false;
        }

        if let Some(ref current) = self.current_piece {
            let current_type = current.piece_type;
            let current_bag_pos = self.current_piece_bag_position;

            if let Some(held_type) = self.hold_piece {
                let held_bag_pos = self
                    .hold_piece_bag_position
                    .expect("hold_piece_bag_position should be set when hold_piece is Some");

                self.hold_piece = Some(current_type);
                self.hold_piece_bag_position = Some(current_bag_pos);
                self.current_piece_bag_position = held_bag_pos;
                self.hold_used = true;
                self.spawn_piece_from_type(held_type);
            } else {
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

    pub fn set_current_piece_type(&mut self, piece_type: usize) {
        if piece_type < 7 && !self.game_over {
            self.current_piece = Some(Piece {
                piece_type,
                x: spawn_x(self.width),
                y: spawn_y_offset(piece_type),
                rotation: 0,
            });
        }
    }

    // === Lock Delay (lock_delay.rs) ===

    pub fn update_lock_delay(&mut self, delta_ms: u32) -> bool {
        if self.game_over {
            return false;
        }

        if self.is_grounded() {
            if let Some(current_delay) = self.lock_delay_ms {
                let new_delay = current_delay + delta_ms;
                if new_delay >= self.lock_delay_max || self.lock_moves_remaining == 0 {
                    self.lock_piece_internal();
                    return true;
                }
                self.lock_delay_ms = Some(new_delay);
            } else {
                self.lock_delay_ms = Some(0);
            }
        } else {
            self.lock_delay_ms = None;
        }
        false
    }

    pub fn is_piece_grounded(&self) -> bool {
        self.is_grounded()
    }

    pub fn get_lock_delay_progress(&self) -> f32 {
        if let Some(delay) = self.lock_delay_ms {
            (delay as f32) / (self.lock_delay_max as f32)
        } else {
            0.0
        }
    }

    // === Movement (movement.rs) ===

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
                if self.is_grounded() {
                    if self.lock_delay_ms.is_none() {
                        self.lock_delay_ms = Some(0);
                    }
                } else {
                    self.lock_delay_ms = None;
                }
                self.last_move_was_rotation = false;
                return true;
            }
        }
        if self.lock_delay_ms.is_none() {
            self.lock_delay_ms = Some(0);
        }
        false
    }

    pub fn hard_drop(&mut self) -> u32 {
        if self.game_over {
            return 0;
        }
        let drop_distance = if let Some(ref piece) = self.current_piece {
            let dist = self.compute_drop_distance(piece);
            if dist > 0 {
                let mut dropped = piece.clone();
                dropped.y += dist;
                self.current_piece = Some(dropped);
                self.last_move_was_rotation = false;
            }
            dist as u32
        } else {
            0
        };
        self.lock_piece_internal();
        drop_distance
    }

    pub fn rotate_cw(&mut self) -> bool {
        self.rotate(true)
    }

    pub fn rotate_ccw(&mut self) -> bool {
        self.rotate(false)
    }

    pub fn step(&mut self, action: u8) -> (u32, bool) {
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

    pub fn get_ghost_piece(&self) -> Option<Piece> {
        if let Some(ref piece) = self.current_piece {
            let drop = self.compute_drop_distance(piece);
            let mut ghost = piece.clone();
            ghost.y += drop;
            Some(ghost)
        } else {
            None
        }
    }

    // === Clearing (clearing.rs) ===

    pub fn get_last_attack_result(&self) -> Option<AttackResult> {
        self.last_attack_result.clone()
    }

    // === Placement (placement.rs) ===

    pub fn place_piece(&mut self, x: i32, y: i32, rotation: usize) -> u32 {
        self.place_piece_internal_with_kick(x, y, rotation, false, 0)
    }

    pub fn execute_placement(&mut self, placement: &Placement) -> u32 {
        let x = placement.piece.x;
        let y = placement.piece.y;
        let rotation = placement.piece.rotation;

        self.place_piece_internal_with_kick(
            x,
            y,
            rotation,
            placement.last_move_was_rotation,
            placement.last_kick_index,
        )
    }

    pub fn get_possible_placements(&self) -> Vec<Placement> {
        // Check cache first
        if let Some(ref cached) = *self.placements_cache.borrow() {
            return cached.clone();
        }

        // Cache miss - compute placements
        let placements = if let Some(ref piece) = self.current_piece {
            let board = Board::new(self.width, self.height, &self.board);
            find_all_placements(&board, piece.piece_type, piece.x, piece.y)
        } else {
            Vec::new()
        };

        // Store in cache and return
        *self.placements_cache.borrow_mut() = Some(placements.clone());
        placements
    }

    pub fn get_placements_for_piece(&self, piece_type: usize) -> Vec<Placement> {
        if piece_type >= 7 {
            return Vec::new();
        }

        let board = Board::new(self.width, self.height, &self.board);
        find_all_placements(
            &board,
            piece_type,
            spawn_x(self.width),
            spawn_y_offset(piece_type),
        )
    }

    pub fn get_possible_placements_with_hold(&self) -> (Vec<Placement>, Vec<Placement>) {
        if self.hold_used {
            return (self.get_possible_placements(), Vec::new());
        }

        if let Some(ref piece) = self.current_piece {
            let board = Board::new(self.width, self.height, &self.board);
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

    /// Execute an action by its index (0-733) in the action space.
    ///
    /// Converts the action index to (x, y, rotation), finds the matching
    /// placement from valid placements, and executes it.
    ///
    /// Args:
    ///     action_idx: Action index from MCTS (0-733)
    ///
    /// Returns:
    ///     Attack sent if successful, or None if action is invalid
    pub fn execute_action_index(&mut self, action_idx: usize) -> Option<u32> {
        let action_space = get_action_space();
        let (x, y, rot) = action_space.index_to_placement(action_idx)?;
        let piece = self.current_piece.as_ref()?;
        let board = Board::new(self.width, self.height, &self.board);
        let placements = find_all_placements(&board, piece.piece_type, piece.x, piece.y);
        let placement = placements
            .iter()
            .find(|p| p.piece.x == x && p.piece.y == y && p.piece.rotation == rot)?;

        Some(self.execute_placement(placement))
    }
}
