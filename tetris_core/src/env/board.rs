//! Board Operations
//!
//! Board state queries and validation.

use crate::piece::{get_cells, Piece};

use super::TetrisEnv;

impl TetrisEnv {
    #[inline]
    pub(crate) fn board_cells(&self) -> &[u8] {
        &self.board
    }

    /// Compute how far a piece can drop using exact collision checks.
    /// Returns the drop distance (0 if already at bottom).
    pub(crate) fn compute_drop_distance(&self, piece: &Piece) -> i32 {
        let mut drop_distance = 0;
        while self.is_valid_position(
            piece.piece_type,
            piece.rotation,
            piece.x,
            piece.y + drop_distance + 1,
        ) {
            drop_distance += 1;
        }
        drop_distance
    }
}

impl TetrisEnv {
    /// Check if a position is valid for a piece
    pub(crate) fn is_valid_position_for(&self, piece: &Piece) -> bool {
        self.is_valid_position(piece.piece_type, piece.rotation, piece.x, piece.y)
    }

    pub(crate) fn is_valid_position(
        &self,
        piece_type: usize,
        rotation: usize,
        x: i32,
        y: i32,
    ) -> bool {
        for (cx, cy) in get_cells(piece_type, rotation, x, y) {
            if cx < 0 || cx >= self.width as i32 || cy < 0 || cy >= self.height as i32 {
                return false;
            }
            if self.board[cy as usize * self.width + cx as usize] != 0 {
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
        self.board[y as usize * self.width + x as usize] != 0
    }

    /// Check if the board is completely empty (perfect clear)
    /// O(1) using total_blocks counter
    pub(crate) fn is_perfect_clear(&self) -> bool {
        self.total_blocks == 0
    }

    /// Recalculate total_blocks and row_fill_counts from the board state.
    /// Call this after directly modifying the board.
    pub(crate) fn sync_board_stats(&mut self) {
        self.total_blocks = 0;
        self.row_fill_counts = vec![0; self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                if self.board[y * self.width + x] == 0 {
                    continue;
                }
                self.total_blocks += 1;
                self.row_fill_counts[y] += 1;
            }
        }
    }
}
