//! Lock Delay Mechanics
//!
//! Grounded detection and lock delay timer management.

use crate::game::constants::DEFAULT_LOCK_MOVES;

use super::TetrisEnv;

impl TetrisEnv {
    /// Check if the current piece is grounded (cannot move down)
    pub(crate) fn is_grounded(&self) -> bool {
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.y += 1;
            !self.is_valid_position_for(&test_piece)
        } else {
            false
        }
    }

    /// Reset lock delay timer (called when piece moves/rotates while grounded)
    pub(crate) fn reset_lock_delay(&mut self) {
        if self.lock_moves_remaining > 0 {
            self.lock_delay_ms = Some(0);
            self.lock_moves_remaining -= 1;
        }
    }

    /// Clear lock delay state (called when piece spawns)
    pub(crate) fn clear_lock_delay(&mut self) {
        self.lock_delay_ms = None;
        self.lock_moves_remaining = DEFAULT_LOCK_MOVES;
    }
}
