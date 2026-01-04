//! Placement Execution
//!
//! Internal placement logic.

use crate::piece::TETROMINOS;

use super::TetrisEnv;

impl TetrisEnv {
    /// Internal placement logic with explicit T-spin detection info
    pub(crate) fn place_piece_internal_with_kick(
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

            if !self.is_valid_position_for_shape(shape, x, y) {
                return 0;
            }

            let mut new_piece = piece.clone();
            new_piece.x = x;
            new_piece.y = y;
            new_piece.rotation = rotation;
            self.current_piece = Some(new_piece);

            self.last_move_was_rotation = was_rotation;
            self.last_kick_index = kick_index;

            let old_attack = self.attack;
            self.lock_piece_internal();
            self.attack - old_attack
        } else {
            0
        }
    }
}
