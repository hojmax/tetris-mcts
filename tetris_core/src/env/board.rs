//! Board Operations
//!
//! Board state queries and validation.

use crate::piece::{get_cells, Piece};

use super::TetrisEnv;

impl TetrisEnv {
    #[inline]
    pub(crate) fn board_cells(&self) -> &[Vec<u8>] {
        &self.board
    }

    /// Compute how far a piece can drop using column_heights for O(1) per cell.
    /// Returns the drop distance (0 if already at bottom).
    pub(crate) fn compute_drop_distance(&self, piece: &Piece) -> i32 {
        let mut min_drop = i32::MAX;
        for (cx, cy) in get_cells(piece.piece_type, piece.rotation, piece.x, piece.y) {
            if cx >= 0 && cx < self.width as i32 {
                // Maximum y this cell can reach is column_heights[cx] - 1
                let max_y = self.column_heights[cx as usize] - 1;
                let drop = max_y - cy;
                min_drop = min_drop.min(drop);
            }
        }
        min_drop.max(0)
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
            if self.board[cy as usize][cx as usize] != 0 {
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
    /// O(1) using total_blocks counter
    pub(crate) fn is_perfect_clear(&self) -> bool {
        self.total_blocks == 0
    }

    /// Recalculate column_heights, total_blocks, and row_fill_counts from the board state.
    /// Call this after directly modifying the board.
    pub(crate) fn sync_board_stats(&mut self) {
        self.total_blocks = 0;
        self.row_fill_counts = vec![0; self.height];
        for x in 0..self.width {
            self.column_heights[x] = self.height as i32; // Assume empty
            for y in 0..self.height {
                if self.board[y][x] != 0 {
                    self.total_blocks += 1;
                    self.row_fill_counts[y] += 1;
                    if (y as i32) < self.column_heights[x] {
                        self.column_heights[x] = y as i32;
                    }
                }
            }
        }
    }
}
