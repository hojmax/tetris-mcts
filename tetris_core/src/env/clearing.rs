//! Line Clearing and T-Spin Detection
//!
//! Locking pieces, clearing lines, and T-spin mechanics.

use crate::constants::T_PIECE;
use crate::piece::{get_cells, Piece};
use crate::scoring::{
    calculate_attack, combo_attack, determine_clear_type, AttackResult,
    BACK_TO_BACK_BONUS, PERFECT_CLEAR_ATTACK,
};

use super::TetrisEnv;

impl TetrisEnv {
    /// Check if the T piece at the given position has a T-spin
    /// Returns (is_tspin, is_mini)
    pub(crate) fn check_tspin(&self, piece: &Piece) -> (bool, bool) {
        if piece.piece_type != T_PIECE {
            return (false, false);
        }

        if !self.last_move_was_rotation {
            return (false, false);
        }

        let center_x = piece.x + 1;
        let center_y = piece.y + 1;

        let corners = [
            (center_x - 1, center_y - 1),
            (center_x + 1, center_y - 1),
            (center_x - 1, center_y + 1),
            (center_x + 1, center_y + 1),
        ];

        let mut filled_corners = 0;
        for (cx, cy) in corners.iter() {
            if self.is_cell_filled(*cx, *cy) {
                filled_corners += 1;
            }
        }

        if filled_corners < 3 {
            return (false, false);
        }

        let front_corners = match piece.rotation {
            0 => [(center_x - 1, center_y - 1), (center_x + 1, center_y - 1)],
            1 => [(center_x + 1, center_y - 1), (center_x + 1, center_y + 1)],
            2 => [(center_x - 1, center_y + 1), (center_x + 1, center_y + 1)],
            3 => [(center_x - 1, center_y - 1), (center_x - 1, center_y + 1)],
            _ => return (false, false),
        };

        let front_filled = front_corners
            .iter()
            .filter(|(cx, cy)| self.is_cell_filled(*cx, *cy))
            .count();

        if front_filled == 2 || self.last_kick_index == 4 {
            (true, false)
        } else {
            (true, true)
        }
    }

    pub(crate) fn lock_piece_internal(&mut self) {
        if let Some(piece) = self.current_piece.take() {
            let (is_tspin, is_mini) = self.check_tspin(&piece);

            for (x, y) in get_cells(piece.piece_type, piece.rotation, piece.x, piece.y) {
                if y >= 0 && y < self.height as i32 && x >= 0 && x < self.width as i32 {
                    self.board[y as usize][x as usize] = 1;
                    self.board_colors[y as usize][x as usize] = Some(piece.piece_type);
                }
            }

            self.clear_lines_internal(is_tspin, is_mini);
            self.hold_used = false;
            self.spawn_piece_internal();
        }
    }

    pub(crate) fn clear_lines_internal(&mut self, is_tspin: bool, is_mini: bool) {
        // Count cleared lines and build new board in one pass (O(n) instead of O(n²))
        let mut new_board = Vec::with_capacity(self.height);
        let mut new_colors = Vec::with_capacity(self.height);
        let mut num_lines = 0u32;

        for y in 0..self.height {
            if self.board[y].iter().all(|&cell| cell != 0) {
                // Line is full - count it but don't add to new board
                num_lines += 1;
            } else {
                // Line not full - keep it
                new_board.push(std::mem::take(&mut self.board[y]));
                new_colors.push(std::mem::take(&mut self.board_colors[y]));
            }
        }

        // Add empty rows at the top
        for _ in 0..num_lines {
            new_board.insert(0, vec![0; self.width]);
            new_colors.insert(0, vec![None; self.width]);
        }

        self.board = new_board;
        self.board_colors = new_colors;

        if num_lines > 0 {
            let clear_type = determine_clear_type(num_lines, is_tspin, is_mini);
            let is_pc = self.is_perfect_clear();
            let (attack_value, new_b2b) =
                calculate_attack(clear_type, self.combo, self.back_to_back, is_pc);

            let mut result = AttackResult::new();
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

            self.attack += attack_value;
            self.combo += 1;
            self.back_to_back = new_b2b;
            self.last_attack_result = Some(result);
        } else {
            self.combo = 0;
            self.last_attack_result = None;
        }

        self.lines_cleared += num_lines;
    }
}
