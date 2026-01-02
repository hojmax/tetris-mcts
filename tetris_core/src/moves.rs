//! Move Generation Module
//!
//! This module provides functionality to find all possible piece placements
//! and the move sequences required to reach them. This is useful for AI/MCTS
//! planning and move evaluation.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::piece::{get_cells_for_shape, Piece, TETROMINOS};

/// Actions that can be taken during piece movement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Left = 1,
    Right = 2,
    Down = 3,
    RotateCW = 4,
    RotateCCW = 5,
    HardDrop = 6,
}

impl Action {
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// A piece state during pathfinding (position + rotation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PieceState {
    x: i32,
    y: i32,
    rotation: usize,
}

/// A possible placement for a piece, including the move sequence to reach it
#[pyclass]
#[derive(Debug, Clone)]
pub struct Placement {
    /// The final piece position (after hard drop)
    #[pyo3(get)]
    pub piece: Piece,
    /// The sequence of moves to reach this placement (action codes)
    #[pyo3(get)]
    pub moves: Vec<u8>,
    /// The column where the piece lands (leftmost cell x coordinate)
    #[pyo3(get)]
    pub column: i32,
    /// The rotation state (0-3)
    #[pyo3(get)]
    pub rotation: usize,
}

#[pymethods]
impl Placement {
    fn __repr__(&self) -> String {
        format!(
            "Placement(col={}, rot={}, moves={:?})",
            self.column, self.rotation, self.moves
        )
    }
}

/// Board representation for collision checking
pub struct Board {
    width: usize,
    height: usize,
    cells: Vec<Vec<u8>>,
}

impl Board {
    pub fn new(width: usize, height: usize, cells: Vec<Vec<u8>>) -> Self {
        Board {
            width,
            height,
            cells,
        }
    }

    /// Check if a piece at the given state is in a valid position
    fn is_valid_position(&self, piece_type: usize, state: &PieceState) -> bool {
        let shape = &TETROMINOS[piece_type][state.rotation];
        for (x, y) in get_cells_for_shape(shape, state.x, state.y) {
            if x < 0 || x >= self.width as i32 || y >= self.height as i32 {
                return false;
            }
            if y >= 0 && self.cells[y as usize][x as usize] != 0 {
                return false;
            }
        }
        true
    }

    /// Check if a piece at the given state is grounded (cannot move down)
    fn is_grounded(&self, piece_type: usize, state: &PieceState) -> bool {
        let below = PieceState {
            x: state.x,
            y: state.y + 1,
            rotation: state.rotation,
        };
        !self.is_valid_position(piece_type, &below)
    }

    /// Get the final y position after hard dropping from a state
    fn get_drop_y(&self, piece_type: usize, state: &PieceState) -> i32 {
        let mut y = state.y;
        let shape = &TETROMINOS[piece_type][state.rotation];
        while self.is_valid_position_for_shape(shape, state.x, y + 1) {
            y += 1;
        }
        y
    }

    fn is_valid_position_for_shape(&self, shape: &[[u8; 4]; 4], x: i32, y: i32) -> bool {
        for (cx, cy) in get_cells_for_shape(shape, x, y) {
            if cx < 0 || cx >= self.width as i32 || cy >= self.height as i32 {
                return false;
            }
            if cy >= 0 && self.cells[cy as usize][cx as usize] != 0 {
                return false;
            }
        }
        true
    }
}

/// SRS Wall kick data for J, L, S, T, Z pieces
fn get_jlstz_kicks(from_state: usize, to_state: usize) -> [(i32, i32); 5] {
    match (from_state, to_state) {
        (0, 1) => [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (1, 0) => [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (1, 2) => [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (2, 1) => [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (2, 3) => [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (3, 2) => [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (3, 0) => [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (0, 3) => [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        _ => [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    }
}

/// SRS Wall kick data for I piece
fn get_i_kicks(from_state: usize, to_state: usize) -> [(i32, i32); 5] {
    match (from_state, to_state) {
        (0, 1) => [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        (1, 0) => [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        (1, 2) => [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
        (2, 1) => [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        (2, 3) => [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        (3, 2) => [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        (3, 0) => [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        (0, 3) => [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
        _ => [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    }
}

/// Try to rotate a piece and return the new state if successful
fn try_rotate(
    board: &Board,
    piece_type: usize,
    state: &PieceState,
    clockwise: bool,
) -> Option<PieceState> {
    let from_state = state.rotation;
    let to_state = if clockwise {
        (state.rotation + 1) % 4
    } else {
        (state.rotation + 3) % 4
    };

    let kicks = if piece_type == 0 {
        get_i_kicks(from_state, to_state)
    } else if piece_type == 1 {
        // O piece - no real rotation change
        [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    } else {
        get_jlstz_kicks(from_state, to_state)
    };

    let new_shape = &TETROMINOS[piece_type][to_state];

    for (dx, dy) in kicks.iter() {
        let new_x = state.x + dx;
        let new_y = state.y + dy;
        if board.is_valid_position_for_shape(new_shape, new_x, new_y) {
            return Some(PieceState {
                x: new_x,
                y: new_y,
                rotation: to_state,
            });
        }
    }
    None
}

/// Find all possible placements for a piece on the given board
///
/// Uses BFS to explore all reachable states from the spawn position,
/// then returns all unique final positions (after hard drop) with
/// the shortest move sequence to reach each.
pub fn find_all_placements(
    board: &Board,
    piece_type: usize,
    spawn_x: i32,
    spawn_y: i32,
) -> Vec<Placement> {
    let start_state = PieceState {
        x: spawn_x,
        y: spawn_y,
        rotation: 0,
    };

    // Check if spawn position is valid
    if !board.is_valid_position(piece_type, &start_state) {
        return Vec::new();
    }

    // BFS to find all reachable states
    let mut visited: HashSet<PieceState> = HashSet::new();
    let mut queue: VecDeque<(PieceState, Vec<u8>)> = VecDeque::new();

    // Track final positions (after hard drop) -> shortest path
    // Key: (final_x, final_y, rotation)
    let mut final_positions: HashMap<(i32, i32, usize), Vec<u8>> = HashMap::new();

    visited.insert(start_state);
    queue.push_back((start_state, Vec::new()));

    while let Some((state, moves)) = queue.pop_front() {
        // Record this state's final position (after hard drop)
        let final_y = board.get_drop_y(piece_type, &state);
        let final_key = (state.x, final_y, state.rotation);

        // Only keep the shortest path to each final position
        if !final_positions.contains_key(&final_key)
            || moves.len() < final_positions[&final_key].len()
        {
            final_positions.insert(final_key, moves.clone());
        }

        // Try all possible moves
        let transitions = [
            (Action::Left, -1, 0, None),
            (Action::Right, 1, 0, None),
            (Action::Down, 0, 1, None),
            (Action::RotateCW, 0, 0, Some(true)),
            (Action::RotateCCW, 0, 0, Some(false)),
        ];

        for (action, dx, dy, rotate) in transitions.iter() {
            let new_state = if let Some(clockwise) = rotate {
                // Rotation with wall kicks
                try_rotate(board, piece_type, &state, *clockwise)
            } else {
                // Simple translation
                let candidate = PieceState {
                    x: state.x + dx,
                    y: state.y + dy,
                    rotation: state.rotation,
                };
                if board.is_valid_position(piece_type, &candidate) {
                    Some(candidate)
                } else {
                    None
                }
            };

            if let Some(new_state) = new_state {
                if !visited.contains(&new_state) {
                    visited.insert(new_state);
                    let mut new_moves = moves.clone();
                    new_moves.push(action.to_u8());
                    queue.push_back((new_state, new_moves));
                }
            }
        }
    }

    // Deduplicate by actual cells occupied (not just x, y, rotation)
    // This handles cases like O piece where different rotations look identical
    let mut seen_cells: HashSet<Vec<(i32, i32)>> = HashSet::new();
    let mut placements: Vec<Placement> = Vec::new();

    for ((x, y, rotation), mut moves) in final_positions {
        let shape = &TETROMINOS[piece_type][rotation];
        let mut cells: Vec<(i32, i32)> = get_cells_for_shape(shape, x, y);
        cells.sort(); // Normalize for comparison

        if seen_cells.insert(cells) {
            // Add hard drop to complete the move sequence
            moves.push(Action::HardDrop.to_u8());

            placements.push(Placement {
                piece: Piece::with_position(piece_type, x, y, rotation),
                moves,
                column: x,
                rotation,
            });
        }
    }

    // Sort by rotation first, then column
    placements.sort_by(|a, b| {
        a.rotation
            .cmp(&b.rotation)
            .then_with(|| a.column.cmp(&b.column))
    });

    placements
}

/// Find all placements including hold piece option
/// Returns (current_piece_placements, hold_piece_placements)
pub fn find_all_placements_with_hold(
    board: &Board,
    current_piece_type: usize,
    hold_piece_type: Option<usize>,
    next_piece_type: usize,
    spawn_x: i32,
    spawn_y: i32,
) -> (Vec<Placement>, Vec<Placement>) {
    let current_placements = find_all_placements(board, current_piece_type, spawn_x, spawn_y);

    let hold_placements = if let Some(hold_type) = hold_piece_type {
        // If we have a held piece, using hold gives us that piece
        find_all_placements(board, hold_type, spawn_x, spawn_y)
    } else {
        // If no held piece, using hold gives us the next piece
        find_all_placements(board, next_piece_type, spawn_x, spawn_y)
    };

    (current_placements, hold_placements)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_board(width: usize, height: usize) -> Board {
        Board::new(width, height, vec![vec![0; width]; height])
    }

    #[test]
    fn test_find_placements_empty_board() {
        let board = empty_board(10, 20);
        // I piece (type 0)
        let placements = find_all_placements(&board, 0, 3, 0);
        assert!(!placements.is_empty());

        // Should have placements in different columns and rotations
        let columns: HashSet<i32> = placements.iter().map(|p| p.column).collect();
        assert!(columns.len() > 1);
    }

    #[test]
    fn test_find_placements_all_pieces() {
        let board = empty_board(10, 20);
        for piece_type in 0..7 {
            let placements = find_all_placements(&board, piece_type, 3, 0);
            assert!(
                !placements.is_empty(),
                "Piece type {} should have placements",
                piece_type
            );
        }
    }

    #[test]
    fn test_placements_have_moves() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        for placement in &placements {
            // Every placement should end with hard drop
            assert!(
                !placement.moves.is_empty(),
                "Placement should have moves"
            );
            assert_eq!(
                *placement.moves.last().unwrap(),
                Action::HardDrop.to_u8(),
                "Last move should be hard drop"
            );
        }
    }

    #[test]
    fn test_placements_reach_bottom() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 0, 3, 0); // I piece

        for placement in &placements {
            // On empty board, pieces should reach near bottom
            // I piece horizontal is at row 1 in its 4x4 grid, so y=18 means cells at y=19
            assert!(
                placement.piece.y >= 16,
                "Piece should be near bottom, got y={}",
                placement.piece.y
            );
        }
    }

    #[test]
    fn test_placements_unique() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        let mut positions: HashSet<(i32, i32, usize)> = HashSet::new();
        for placement in &placements {
            let key = (
                placement.piece.x,
                placement.piece.y,
                placement.piece.rotation,
            );
            assert!(
                positions.insert(key),
                "Duplicate placement found: {:?}",
                key
            );
        }
    }

    #[test]
    fn test_i_piece_horizontal_positions() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 0, 3, 0); // I piece

        // I piece horizontal (rotation 0 and 2) can be at columns -1 to 6 (piece x position)
        // because the I piece extends 4 cells from x
        let horizontal_placements: Vec<_> = placements
            .iter()
            .filter(|p| p.rotation == 0 || p.rotation == 2)
            .collect();

        assert!(!horizontal_placements.is_empty());
    }

    #[test]
    fn test_o_piece_no_rotation_change() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 1, 3, 0); // O piece

        // O piece looks the same in all rotations, so we should have
        // placements that effectively only differ by column
        let unique_columns: HashSet<i32> = placements.iter().map(|p| p.column).collect();
        assert!(!unique_columns.is_empty());
    }

    #[test]
    fn test_blocked_board() {
        // Create a board with a wall blocking most positions
        let mut cells = vec![vec![0; 10]; 20];
        // Fill rows 10-19 except column 0
        for y in 10..20 {
            for x in 1..10 {
                cells[y][x] = 1;
            }
        }
        let board = Board::new(10, 20, cells);

        // I piece can only fit in column 0 when vertical
        let placements = find_all_placements(&board, 0, 3, 0);

        // Should still find some placements (vertical I in column 0)
        // or placements above the filled area
        assert!(!placements.is_empty());
    }

    #[test]
    fn test_spawn_blocked() {
        // Create a board where spawn is blocked
        let mut cells = vec![vec![0; 10]; 20];
        for x in 0..10 {
            cells[0][x] = 1;
            cells[1][x] = 1;
        }
        let board = Board::new(10, 20, cells);

        let placements = find_all_placements(&board, 0, 3, 0);
        assert!(placements.is_empty(), "Should have no placements when spawn is blocked");
    }

    #[test]
    fn test_move_sequence_valid() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        for placement in &placements {
            for &action in &placement.moves {
                assert!(
                    action >= 1 && action <= 6,
                    "Invalid action code: {}",
                    action
                );
            }
        }
    }

    #[test]
    fn test_action_enum() {
        assert_eq!(Action::Left.to_u8(), 1);
        assert_eq!(Action::Right.to_u8(), 2);
        assert_eq!(Action::Down.to_u8(), 3);
        assert_eq!(Action::RotateCW.to_u8(), 4);
        assert_eq!(Action::RotateCCW.to_u8(), 5);
        assert_eq!(Action::HardDrop.to_u8(), 6);
    }

    #[test]
    fn test_placements_sorted() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        // Check that placements are sorted by rotation first, then column
        for i in 1..placements.len() {
            let prev = &placements[i - 1];
            let curr = &placements[i];
            assert!(
                prev.rotation < curr.rotation
                    || (prev.rotation == curr.rotation && prev.column <= curr.column),
                "Placements not sorted: {:?} should come before {:?}",
                prev,
                curr
            );
        }
    }

    #[test]
    fn test_t_piece_rotations() {
        let board = empty_board(10, 20);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        // T piece should have placements in all 4 rotations
        let rotations: HashSet<usize> = placements.iter().map(|p| p.rotation).collect();
        assert_eq!(rotations.len(), 4, "T piece should have all 4 rotations");
    }

    #[test]
    fn test_find_placements_with_hold() {
        let board = empty_board(10, 20);

        // With no hold piece, hold gives next piece
        let (current, hold) = find_all_placements_with_hold(&board, 0, None, 2, 3, 0);
        assert!(!current.is_empty());
        assert!(!hold.is_empty());

        // Current should be I piece (type 0) placements
        assert!(current.iter().all(|p| p.piece.piece_type == 0));
        // Hold should be T piece (type 2) placements (next piece)
        assert!(hold.iter().all(|p| p.piece.piece_type == 2));
    }

    #[test]
    fn test_find_placements_with_existing_hold() {
        let board = empty_board(10, 20);

        // With existing hold piece, hold gives that piece
        let (current, hold) = find_all_placements_with_hold(&board, 0, Some(5), 2, 3, 0);

        // Current should be I piece (type 0) placements
        assert!(current.iter().all(|p| p.piece.piece_type == 0));
        // Hold should be J piece (type 5) placements (held piece)
        assert!(hold.iter().all(|p| p.piece.piece_type == 5));
    }
}
