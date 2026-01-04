//! Piece Movement
//!
//! Horizontal movement and rotation (internal implementations).

use crate::kicks::get_kicks_for_piece;
use crate::piece::TETROMINOS;

use super::TetrisEnv;

impl TetrisEnv {
    /// Internal horizontal movement logic
    pub(crate) fn move_horizontal(&mut self, dx: i32) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x += dx;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                if was_grounded && self.lock_delay_ms.is_some() {
                    self.reset_lock_delay();
                }
                self.last_move_was_rotation = false;
                return true;
            }
        }
        false
    }

    /// Internal rotation logic using SRS wall kicks
    pub(crate) fn rotate(&mut self, clockwise: bool) -> bool {
        if self.game_over {
            return false;
        }
        let was_grounded = self.is_grounded();
        if let Some(ref piece) = self.current_piece {
            let from_state = piece.rotation;
            let to_state = if clockwise {
                (piece.rotation + 1) % 4
            } else {
                (piece.rotation + 3) % 4
            };
            let new_shape = &TETROMINOS[piece.piece_type][to_state];
            let kicks = get_kicks_for_piece(piece.piece_type, from_state, to_state);

            for (kick_idx, (dx, dy)) in kicks.iter().enumerate() {
                let new_x = piece.x + dx;
                let new_y = piece.y + dy;
                if self.is_valid_position_for_shape(new_shape, new_x, new_y) {
                    let mut new_piece = piece.clone();
                    new_piece.x = new_x;
                    new_piece.y = new_y;
                    new_piece.rotation = to_state;
                    self.current_piece = Some(new_piece);
                    if was_grounded && self.lock_delay_ms.is_some() {
                        self.reset_lock_delay();
                    }
                    self.last_move_was_rotation = true;
                    self.last_kick_index = kick_idx;
                    return true;
                }
            }
        }
        false
    }
}
