//! Board Operations
//!
//! Board state queries and validation.

use crate::piece::{get_cells, Piece};

use super::TetrisEnv;

impl TetrisEnv {
    /// Check if a position is valid for a piece
    pub(crate) fn is_valid_position_for(&self, piece: &Piece) -> bool {
        self.is_valid_position(piece.piece_type, piece.rotation, piece.x, piece.y)
    }

    pub(crate) fn is_valid_position(&self, piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        for (cx, cy) in get_cells(piece_type, rotation, x, y) {
            if cx < 0 || cx >= self.width as i32 || cy >= self.height as i32 {
                return false;
            }
            if cy >= 0 && self.board[cy as usize][cx as usize] != 0 {
                return false;
            }
        }
        true
    }

    /// Check if a cell is filled (occupied or out of bounds)
    pub(crate) fn is_cell_filled(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return true;
        }
        self.board[y as usize][x as usize] != 0
    }

    /// Check if the board is completely empty (perfect clear)
    pub(crate) fn is_perfect_clear(&self) -> bool {
        for row in &self.board {
            for &cell in row {
                if cell != 0 {
                    return false;
                }
            }
        }
        true
    }
}
