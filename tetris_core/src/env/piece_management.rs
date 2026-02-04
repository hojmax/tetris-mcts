//! Piece Queue and Hold Management
//!
//! 7-bag randomizer, queue management, and hold functionality (internal implementations).

use rand::seq::SliceRandom;

use crate::constants::{I_PIECE, O_PIECE};
use crate::piece::Piece;

use super::TetrisEnv;

/// Get spawn X position for a piece (centers the 4-wide bounding box).
pub(super) fn spawn_x(width: usize) -> i32 {
    (width as i32 - 4) / 2
}

/// Get spawn Y offset for a piece type.
/// I and O pieces have cells at y=1 in their matrix, so they spawn at y=-1 to align with other pieces.
pub(super) fn spawn_y_offset(piece_type: usize) -> i32 {
    if piece_type == I_PIECE || piece_type == O_PIECE {
        -1
    } else {
        0
    }
}

impl TetrisEnv {
    /// Ensure the piece queue has at least `count` pieces
    pub(crate) fn fill_queue(&mut self, count: usize) {
        while self.piece_queue.len() < count {
            let mut bag: Vec<usize> = (0..7).collect();
            bag.shuffle(&mut self.rng);
            self.piece_queue.extend(bag);
        }
    }

    pub(crate) fn spawn_piece_internal(&mut self) {
        // Ensure we have enough pieces in the queue
        self.fill_queue(7);

        let piece_type = self.piece_queue.pop_front().expect("Queue should not be empty after fill_queue");
        let mut piece = Piece::new(piece_type);

        piece.x = spawn_x(self.width);
        piece.y = spawn_y_offset(piece_type);
        piece.rotation = 0;

        self.current_piece_bag_position = self.pieces_spawned;
        self.pieces_spawned += 1;

        self.clear_lock_delay();
        self.last_move_was_rotation = false;
        self.last_kick_index = 0;

        let is_valid = self.is_valid_position_for(&piece);
        self.current_piece = Some(piece);
        if !is_valid {
            self.game_over = true;
        }
        // Invalidate placements cache when piece changes
        self.invalidate_placement_cache();
    }

    pub(crate) fn spawn_piece_from_type(&mut self, piece_type: usize) {
        let mut piece = Piece::new(piece_type);

        piece.x = spawn_x(self.width);
        piece.y = spawn_y_offset(piece_type);
        piece.rotation = 0;

        self.last_move_was_rotation = false;
        self.last_kick_index = 0;

        let is_valid = self.is_valid_position_for(&piece);
        self.current_piece = Some(piece);
        if !is_valid {
            self.game_over = true;
        }
    }
}
