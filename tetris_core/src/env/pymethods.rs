//! Python-exposed methods for TetrisEnv
//!
//! All #[pymethods] consolidated in one file due to PyO3 limitation.

use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::mcts::HOLD_ACTION_INDEX;
use crate::moves::{find_all_placements, Board, Placement};
use crate::piece::{spawn_x, spawn_y, Piece};
use crate::scoring::AttackResult;

use super::global_cache::{
    build_placement_lookup_key, get_cached_placements, insert_cached_placements,
};
use super::state::PlacementCache;
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

    pub fn set_board(&mut self, board: Vec<Vec<u8>>) -> PyResult<()> {
        if board.len() != self.height {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Board height mismatch: got {}, expected {}",
                board.len(),
                self.height
            )));
        }
        for (y, row) in board.iter().enumerate() {
            if row.len() != self.width {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Board width mismatch at row {}: got {}, expected {}",
                    y,
                    row.len(),
                    self.width
                )));
            }
            for (x, &cell) in row.iter().enumerate() {
                if cell != 0 && cell != 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid board value at ({}, {}): {} (expected 0 or 1)",
                        x, y, cell
                    )));
                }
            }
        }

        // Flatten nested Vec into flat Vec
        self.board = board.into_iter().flatten().collect();
        self.board_piece_types = vec![None; self.width * self.height];
        self.sync_board_stats();
        self.invalidate_placement_cache();
        Ok(())
    }

    pub fn set_board_piece_types(
        &mut self,
        board_piece_types: Vec<Vec<Option<usize>>>,
    ) -> PyResult<()> {
        if board_piece_types.len() != self.height {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "board_piece_types height mismatch: got {}, expected {}",
                board_piece_types.len(),
                self.height
            )));
        }
        for (y, row) in board_piece_types.iter().enumerate() {
            if row.len() != self.width {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "board_piece_types width mismatch at row {}: got {}, expected {}",
                    y,
                    row.len(),
                    self.width
                )));
            }
            for (x, &piece_type) in row.iter().enumerate() {
                if let Some(pt) = piece_type {
                    if pt >= 7 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid piece type at ({}, {}): {} (expected 0-6)",
                            x, y, pt
                        )));
                    }
                    if self.board[y * self.width + x] == 0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Piece type set for empty board cell at ({}, {})",
                            x, y
                        )));
                    }
                } else if self.board[y * self.width + x] != 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Missing piece type for filled board cell at ({}, {})",
                        x, y
                    )));
                }
            }
        }

        // Flatten nested Vec into flat Vec
        self.board_piece_types = board_piece_types.into_iter().flatten().collect();
        Ok(())
    }

    pub fn set_queue(&mut self, queue: Vec<usize>) -> PyResult<()> {
        for (idx, piece_type) in queue.iter().enumerate() {
            if *piece_type >= 7 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid queue piece type at index {}: {} (expected 0-6)",
                    idx, piece_type
                )));
            }
        }

        self.piece_queue = VecDeque::from(queue);
        Ok(())
    }

    pub fn set_hold_piece_type(&mut self, hold_piece_type: Option<usize>) -> PyResult<()> {
        if let Some(pt) = hold_piece_type {
            if pt >= 7 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid hold piece type: {} (expected 0-6)",
                    pt
                )));
            }
        }
        self.hold_piece = hold_piece_type;
        self.hold_piece_bag_position = hold_piece_type.map(|_| 0);
        Ok(())
    }

    pub fn set_hold_used(&mut self, hold_used: bool) {
        self.hold_used = hold_used;
    }

    // === Board (board.rs) ===

    pub fn get_board(&self) -> Vec<Vec<u8>> {
        self.board
            .chunks(self.width)
            .map(|row| row.to_vec())
            .collect()
    }

    pub fn get_board_piece_types(&self) -> Vec<Vec<Option<usize>>> {
        self.board_piece_types
            .chunks(self.width)
            .map(|row| row.to_vec())
            .collect()
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
        let mut used_in_bag = [false; 7];

        let current_bag_pos = self.current_piece_bag_position as usize;
        if current_bag_pos >= bag_start && current_bag_pos < bag_end {
            if let Some(ref piece) = self.current_piece {
                used_in_bag[piece.piece_type] = true;
            }
        }

        if let (Some(hold_type), Some(hold_bag_pos)) =
            (self.hold_piece, self.hold_piece_bag_position)
        {
            let hold_pos = hold_bag_pos as usize;
            if hold_pos >= bag_start && hold_pos < bag_end {
                used_in_bag[hold_type] = true;
            }
        }

        for (i, &piece_type) in self.piece_queue.iter().enumerate() {
            let piece_pos = self.pieces_spawned as usize + i;
            if piece_pos >= bag_start && piece_pos < bag_end {
                used_in_bag[piece_type] = true;
            }
        }

        (0..7)
            .filter(|piece_type| !used_in_bag[*piece_type])
            .collect()
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

    pub fn set_current_piece_type(&mut self, piece_type: usize) {
        if piece_type < 7 && !self.game_over {
            self.current_piece = Some(Piece::spawn(piece_type, self.width));
            self.clear_lock_delay();
            self.last_move_was_rotation = false;
            self.last_kick_index = 0;
            self.invalidate_placement_cache();
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
        self.ensure_placements_cache();
        let cache_ref = self.placements_cache.borrow();
        cache_ref
            .as_ref()
            .expect("placements cache should exist after ensure_placements_cache")
            .placements
            .as_ref()
            .clone()
    }

    pub fn get_placements_for_piece(&self, piece_type: usize) -> Vec<Placement> {
        if piece_type >= 7 {
            return Vec::new();
        }

        let board = Board::new(self.width, self.height, self.board_cells());
        find_all_placements(&board, piece_type, spawn_x(self.width), spawn_y(piece_type))
    }

    /// Execute an action by its index in the action space.
    ///
    /// Placement actions execute immediately. The dedicated hold action executes
    /// hold as a standalone action and returns zero attack.
    ///
    /// Args:
    ///     action_idx: Action index from MCTS
    ///
    /// Returns:
    ///     Attack sent if successful, or None if action is invalid
    pub fn execute_action_index(&mut self, action_idx: usize) -> Option<u32> {
        if action_idx == HOLD_ACTION_INDEX {
            return if self.hold() { Some(0) } else { None };
        }

        let (x, y, rotation, last_move_was_rotation, last_kick_index) =
            self.cached_placement_params_for_action(action_idx)?;

        Some(self.place_piece_internal_with_kick(
            x,
            y,
            rotation,
            last_move_was_rotation,
            last_kick_index,
        ))
    }
}

impl TetrisEnv {
    fn ensure_placements_cache(&self) {
        if self.placements_cache.borrow().is_some() {
            return;
        }

        let Some(piece) = self.current_piece.as_ref() else {
            *self.placements_cache.borrow_mut() = Some(PlacementCache {
                placements: Arc::new(Vec::new()),
                action_to_placement_idx: Arc::new(vec![None; crate::mcts::NUM_ACTIONS]),
            });
            return;
        };
        let piece = piece.clone();
        let global_cache_key = build_placement_lookup_key(self, &piece);

        if let Some(cache_key) = global_cache_key {
            if let Some(global_cached) = get_cached_placements(cache_key) {
                *self.placements_cache.borrow_mut() = Some(global_cached);
                return;
            }
        }

        let board = Board::new(self.width, self.height, self.board_cells());
        let placements = find_all_placements(&board, piece.piece_type, piece.x, piece.y);

        let mut action_to_placement_idx = vec![None; crate::mcts::NUM_ACTIONS];
        for (placement_idx, placement) in placements.iter().enumerate() {
            debug_assert!(placement.action_index < action_to_placement_idx.len());
            action_to_placement_idx[placement.action_index] = Some(placement_idx);
        }

        let cache_entry = PlacementCache {
            placements: Arc::new(placements),
            action_to_placement_idx: Arc::new(action_to_placement_idx),
        };

        if let Some(cache_key) = global_cache_key {
            insert_cached_placements(cache_key, cache_entry.clone());
        }
        *self.placements_cache.borrow_mut() = Some(cache_entry);
    }

    fn cached_placement_params_for_action(
        &self,
        action_idx: usize,
    ) -> Option<(i32, i32, usize, bool, usize)> {
        self.ensure_placements_cache();

        let cache_ref = self.placements_cache.borrow();
        let cache = cache_ref
            .as_ref()
            .expect("placements cache should exist after ensure_placements_cache");
        let placement_idx = cache
            .action_to_placement_idx
            .get(action_idx)
            .copied()
            .flatten()?;
        let placement = &cache.placements[placement_idx];
        Some((
            placement.piece.x,
            placement.piece.y,
            placement.piece.rotation,
            placement.last_move_was_rotation,
            placement.last_kick_index,
        ))
    }

    pub(crate) fn fill_cached_action_mask(&self, mask: &mut [bool]) {
        debug_assert_eq!(mask.len(), crate::mcts::NUM_ACTIONS);
        mask.fill(false);

        if self.game_over {
            return;
        }

        self.ensure_placements_cache();

        let cache_ref = self.placements_cache.borrow();
        let cache = cache_ref
            .as_ref()
            .expect("placements cache should exist after ensure_placements_cache");
        for (action_idx, placement_idx) in cache.action_to_placement_idx.iter().enumerate() {
            if placement_idx.is_some() {
                mask[action_idx] = true;
            }
        }

        let hold_is_available =
            !self.game_over && !self.is_hold_used() && self.current_piece.is_some();
        if hold_is_available {
            mask[HOLD_ACTION_INDEX] = true;
        }
    }
}
