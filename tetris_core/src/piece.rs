use pyo3::prelude::*;

/// Rotation states: 0=spawn, 1=R (CW from spawn), 2=180°, 3=L (CCW from spawn)
/// All tetromino shapes in all 4 rotation states
/// Shape format: [piece_type][rotation_state][row][col]
pub const TETROMINOS: [[[[u8; 4]; 4]; 4]; 7] = [
    // I piece - index 0
    [
        // State 0 (spawn)
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        // State R (CW from spawn)
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        // State 2 (180°)
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        // State L (CCW from spawn)
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    ],
    // O piece - index 1
    [
        // All states are the same for O (centered in matrix for rotation stability)
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    ],
    // T piece - index 2
    [
        // State 0 (spawn)
        [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // State R
        [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        // State 2
        [[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        // State L
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
    // S piece - index 3
    [
        // State 0 (spawn)
        [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // State R
        [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        // State 2
        [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
        // State L
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
    // Z piece - index 4
    [
        // State 0 (spawn)
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // State R
        [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        // State 2
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        // State L
        [[0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    ],
    // J piece - index 5
    [
        // State 0 (spawn)
        [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // State R
        [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        // State 2
        [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        // State L
        [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    ],
    // L piece - index 6
    [
        // State 0 (spawn)
        [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // State R
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        // State 2
        [[0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        // State L
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
];

/// Precomputed cell offsets for each tetromino [piece_type][rotation] -> [(dx, dy); 4]
/// This avoids iterating over the 4x4 shape array at runtime.
pub const TETROMINO_CELLS: [[[(i8, i8); 4]; 4]; 7] = [
    // I piece - index 0
    [
        [(0, 1), (1, 1), (2, 1), (3, 1)], // State 0
        [(2, 0), (2, 1), (2, 2), (2, 3)], // State R
        [(0, 2), (1, 2), (2, 2), (3, 2)], // State 2
        [(1, 0), (1, 1), (1, 2), (1, 3)], // State L
    ],
    // O piece - index 1
    [
        [(1, 1), (2, 1), (1, 2), (2, 2)], // All states same
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
    ],
    // T piece - index 2
    [
        [(1, 0), (0, 1), (1, 1), (2, 1)], // State 0
        [(1, 0), (1, 1), (2, 1), (1, 2)], // State R
        [(0, 1), (1, 1), (2, 1), (1, 2)], // State 2
        [(1, 0), (0, 1), (1, 1), (1, 2)], // State L
    ],
    // S piece - index 3
    [
        [(1, 0), (2, 0), (0, 1), (1, 1)], // State 0
        [(1, 0), (1, 1), (2, 1), (2, 2)], // State R
        [(1, 1), (2, 1), (0, 2), (1, 2)], // State 2
        [(0, 0), (0, 1), (1, 1), (1, 2)], // State L
    ],
    // Z piece - index 4
    [
        [(0, 0), (1, 0), (1, 1), (2, 1)], // State 0
        [(2, 0), (1, 1), (2, 1), (1, 2)], // State R
        [(0, 1), (1, 1), (1, 2), (2, 2)], // State 2
        [(1, 0), (0, 1), (1, 1), (0, 2)], // State L
    ],
    // J piece - index 5
    [
        [(0, 0), (0, 1), (1, 1), (2, 1)], // State 0
        [(1, 0), (2, 0), (1, 1), (1, 2)], // State R
        [(0, 1), (1, 1), (2, 1), (2, 2)], // State 2
        [(1, 0), (1, 1), (0, 2), (1, 2)], // State L
    ],
    // L piece - index 6
    [
        [(2, 0), (0, 1), (1, 1), (2, 1)], // State 0
        [(1, 0), (1, 1), (1, 2), (2, 2)], // State R
        [(0, 1), (1, 1), (2, 1), (0, 2)], // State 2
        [(0, 0), (1, 0), (1, 1), (1, 2)], // State L
    ],
];

/// Colors for each tetromino (RGB) - matching Jstris style
pub const COLORS: [(u8, u8, u8); 7] = [
    (93, 173, 212), // I - Light blue/Cyan
    (219, 174, 63), // O - Golden yellow
    (178, 74, 156), // T - Magenta
    (114, 184, 65), // S - Green
    (204, 65, 65),  // Z - Red
    (59, 84, 165),  // J - Blue
    (227, 127, 59), // L - Orange
];

/// Number of piece types (tetromino variants)
pub const NUM_PIECE_TYPES: usize = 7;

/// Maximum number of cells in a tetromino (always 4)
pub const MAX_PIECE_CELLS: usize = 4;

/// Get the cells occupied by a piece at a given position using precomputed offsets.
/// Returns a fixed-size array of exactly 4 cells (all tetrominos have 4 cells).
#[inline]
pub fn get_cells(
    piece_type: usize,
    rotation: usize,
    x: i32,
    y: i32,
) -> [(i32, i32); MAX_PIECE_CELLS] {
    let offsets = &TETROMINO_CELLS[piece_type][rotation];
    [
        (x + offsets[0].0 as i32, y + offsets[0].1 as i32),
        (x + offsets[1].0 as i32, y + offsets[1].1 as i32),
        (x + offsets[2].0 as i32, y + offsets[2].1 as i32),
        (x + offsets[3].0 as i32, y + offsets[3].1 as i32),
    ]
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    #[pyo3(get)]
    pub piece_type: usize,
    #[pyo3(get)]
    pub x: i32,
    #[pyo3(get)]
    pub y: i32,
    #[pyo3(get)]
    pub rotation: usize,
}

#[pymethods]
impl Piece {
    #[new]
    pub fn new(piece_type: usize) -> Self {
        Piece {
            piece_type,
            x: 3,
            y: 0,
            rotation: 0,
        }
    }

    pub fn get_color(&self) -> (u8, u8, u8) {
        COLORS[self.piece_type]
    }

    pub fn get_cells(&self) -> Vec<(i32, i32)> {
        get_cells(self.piece_type, self.rotation, self.x, self.y).to_vec()
    }
}

impl Piece {
    /// Create a piece with specific position and rotation
    pub fn with_position(piece_type: usize, x: i32, y: i32, rotation: usize) -> Self {
        Piece {
            piece_type,
            x,
            y,
            rotation,
        }
    }

    /// Get the raw shape array for this piece
    pub fn get_shape_array(&self) -> &[[u8; 4]; 4] {
        &TETROMINOS[self.piece_type][self.rotation]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_creation() {
        let piece = Piece::new(0); // I piece
        assert_eq!(piece.piece_type, 0);
        assert_eq!(piece.x, 3);
        assert_eq!(piece.y, 0);
        assert_eq!(piece.rotation, 0);
    }

    #[test]
    fn test_piece_with_position() {
        let piece = Piece::with_position(2, 5, 10, 1);
        assert_eq!(piece.piece_type, 2);
        assert_eq!(piece.x, 5);
        assert_eq!(piece.y, 10);
        assert_eq!(piece.rotation, 1);
    }

    #[test]
    fn test_piece_colors() {
        // Piece color lookup should succeed for each piece type.
        for i in 0..NUM_PIECE_TYPES {
            let piece = Piece::new(i);
            let _color = piece.get_color();
        }
    }

    #[test]
    fn test_colors_are_unique() {
        let mut colors = COLORS.to_vec();
        colors.sort();
        colors.dedup();
        assert_eq!(colors.len(), NUM_PIECE_TYPES);
    }

    #[test]
    fn test_piece_cells_count() {
        // All tetrominos have exactly 4 cells
        for piece_type in 0..NUM_PIECE_TYPES {
            for rotation in 0..4 {
                let piece = Piece::with_position(piece_type, 0, 0, rotation);
                let cells = piece.get_cells();
                assert_eq!(
                    cells.len(),
                    4,
                    "Piece type {} rotation {} should have 4 cells",
                    piece_type,
                    rotation
                );
            }
        }
    }

    #[test]
    fn test_i_piece_cells() {
        let piece = Piece::new(0); // I piece, spawn state
        let cells = piece.get_cells();
        assert_eq!(cells.len(), 4);
        // I piece in spawn state is horizontal on row 1
        // With x=3, y=0, cells should be at (3,1), (4,1), (5,1), (6,1)
        assert!(cells.contains(&(3, 1)));
        assert!(cells.contains(&(4, 1)));
        assert!(cells.contains(&(5, 1)));
        assert!(cells.contains(&(6, 1)));
    }

    #[test]
    fn test_o_piece_symmetry() {
        // O piece should be the same in all rotations
        for rotation in 0..4 {
            let cells = get_cells(1, rotation, 0, 0);
            let base_cells = get_cells(1, 0, 0, 0);
            assert_eq!(
                cells, base_cells,
                "O piece rotation {} should match spawn state",
                rotation
            );
        }
    }

    #[test]
    fn test_get_shape_array() {
        let piece = Piece::new(0);
        let shape = piece.get_shape_array();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[0].len(), 4);
    }

    #[test]
    fn test_piece_clone() {
        let piece = Piece::new(3);
        let cloned = piece.clone();
        assert_eq!(piece, cloned);
    }

    #[test]
    fn test_get_cells() {
        // Test with I piece horizontal
        let cells = get_cells(0, 0, 0, 0);
        assert_eq!(cells.len(), MAX_PIECE_CELLS);
    }

    #[test]
    fn test_rotation_states_valid() {
        // All pieces should have valid shapes in all 4 rotation states
        for piece_type in 0..NUM_PIECE_TYPES {
            for rotation in 0..4 {
                let shape = &TETROMINOS[piece_type][rotation];
                let mut cell_count = 0;
                for row in shape.iter() {
                    for &cell in row.iter() {
                        if cell == 1 {
                            cell_count += 1;
                        }
                    }
                }
                assert_eq!(
                    cell_count, 4,
                    "Piece {} rotation {} should have exactly 4 cells",
                    piece_type, rotation
                );
            }
        }
    }

    #[test]
    fn test_i_piece_rotations() {
        // I piece should be vertical in states 1 and 3
        let piece_r = Piece::with_position(0, 0, 0, 1);
        let cells_r = piece_r.get_cells();
        // All cells should have same x coordinate in R state
        let x_coords: Vec<i32> = cells_r.iter().map(|(x, _)| *x).collect();
        assert!(
            x_coords.iter().all(|&x| x == x_coords[0]),
            "I piece in R state should be vertical"
        );

        let piece_l = Piece::with_position(0, 0, 0, 3);
        let cells_l = piece_l.get_cells();
        let x_coords: Vec<i32> = cells_l.iter().map(|(x, _)| *x).collect();
        assert!(
            x_coords.iter().all(|&x| x == x_coords[0]),
            "I piece in L state should be vertical"
        );
    }

    #[test]
    fn test_t_piece_spawn_shape() {
        // T piece spawn: looks like ⊥
        let piece = Piece::new(2);
        let cells = piece.get_cells();
        // With x=3, y=0:
        // Row 0: (4, 0) - top of T
        // Row 1: (3, 1), (4, 1), (5, 1) - bottom of T
        assert!(cells.contains(&(4, 0)));
        assert!(cells.contains(&(3, 1)));
        assert!(cells.contains(&(4, 1)));
        assert!(cells.contains(&(5, 1)));
    }

    #[test]
    fn test_num_piece_types_constant() {
        assert_eq!(NUM_PIECE_TYPES, 7);
        assert_eq!(TETROMINOS.len(), NUM_PIECE_TYPES);
        assert_eq!(COLORS.len(), NUM_PIECE_TYPES);
    }

    #[test]
    fn test_tetromino_cells_matches_tetrominos() {
        // Verify that TETROMINO_CELLS matches what we'd compute from TETROMINOS
        for piece_type in 0..NUM_PIECE_TYPES {
            for rotation in 0..4 {
                // Compute cells the slow way from TETROMINOS
                let shape = &TETROMINOS[piece_type][rotation];
                let mut expected: Vec<(i8, i8)> = Vec::new();
                for dy in 0..4 {
                    for dx in 0..4 {
                        if shape[dy][dx] == 1 {
                            expected.push((dx as i8, dy as i8));
                        }
                    }
                }
                expected.sort();

                // Get cells from precomputed table
                let mut precomputed: Vec<(i8, i8)> = TETROMINO_CELLS[piece_type][rotation].to_vec();
                precomputed.sort();

                assert_eq!(
                    expected, precomputed,
                    "TETROMINO_CELLS mismatch for piece {} rotation {}: expected {:?}, got {:?}",
                    piece_type, rotation, expected, precomputed
                );
            }
        }
    }
}
