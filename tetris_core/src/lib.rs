use pyo3::prelude::*;
use rand::Rng;

// Tetromino shapes (4 rotations each)
const TETROMINOS: [[[u8; 4]; 4]; 7] = [
    // I
    [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    // O
    [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ],
    // T
    [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ],
    // S
    [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ],
    // Z
    [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ],
    // J
    [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ],
    // L
    [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ],
];

// Colors for each tetromino (RGB)
const COLORS: [(u8, u8, u8); 7] = [
    (0, 255, 255),   // I - Cyan
    (255, 255, 0),   // O - Yellow
    (128, 0, 128),   // T - Purple
    (0, 255, 0),     // S - Green
    (255, 0, 0),     // Z - Red
    (0, 0, 255),     // J - Blue
    (255, 165, 0),   // L - Orange
];

fn rotate_shape(shape: [[u8; 4]; 4], clockwise: bool) -> [[u8; 4]; 4] {
    let mut rotated = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            if clockwise {
                rotated[x][3 - y] = shape[y][x];
            } else {
                rotated[3 - x][y] = shape[y][x];
            }
        }
    }
    rotated
}

fn get_cells_for_piece(piece: &Piece) -> Vec<(i32, i32)> {
    let mut cells = Vec::new();
    for dy in 0..4 {
        for dx in 0..4 {
            if piece.shape[dy][dx] == 1 {
                cells.push((piece.x + dx as i32, piece.y + dy as i32));
            }
        }
    }
    cells
}

#[pyclass]
#[derive(Clone)]
pub struct Piece {
    #[pyo3(get)]
    pub piece_type: usize,
    #[pyo3(get)]
    pub x: i32,
    #[pyo3(get)]
    pub y: i32,
    #[pyo3(get)]
    pub rotation: usize,
    shape: [[u8; 4]; 4],
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
            shape: TETROMINOS[piece_type],
        }
    }

    pub fn get_shape(&self) -> Vec<Vec<u8>> {
        self.shape.iter().map(|row| row.to_vec()).collect()
    }

    pub fn get_color(&self) -> (u8, u8, u8) {
        COLORS[self.piece_type]
    }

    pub fn get_cells(&self) -> Vec<(i32, i32)> {
        get_cells_for_piece(self)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct TetrisEnv {
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
    #[pyo3(get)]
    pub score: u32,
    #[pyo3(get)]
    pub lines_cleared: u32,
    #[pyo3(get)]
    pub level: u32,
    #[pyo3(get)]
    pub game_over: bool,
    board: Vec<Vec<u8>>,
    board_colors: Vec<Vec<Option<usize>>>,
    current_piece: Option<Piece>,
    next_piece: Option<Piece>,
}

#[pymethods]
impl TetrisEnv {
    #[new]
    #[pyo3(signature = (width=10, height=20))]
    pub fn new(width: usize, height: usize) -> Self {
        let mut env = TetrisEnv {
            width,
            height,
            score: 0,
            lines_cleared: 0,
            level: 1,
            game_over: false,
            board: vec![vec![0; width]; height],
            board_colors: vec![vec![None; width]; height],
            current_piece: None,
            next_piece: None,
        };
        env.spawn_piece();
        env
    }

    pub fn reset(&mut self) {
        self.board = vec![vec![0; self.width]; self.height];
        self.board_colors = vec![vec![None; self.width]; self.height];
        self.score = 0;
        self.lines_cleared = 0;
        self.level = 1;
        self.game_over = false;
        self.current_piece = None;
        self.next_piece = None;
        self.spawn_piece();
    }

    pub fn get_board(&self) -> Vec<Vec<u8>> {
        self.board.clone()
    }

    pub fn get_board_colors(&self) -> Vec<Vec<Option<usize>>> {
        self.board_colors.clone()
    }

    pub fn get_current_piece(&self) -> Option<Piece> {
        self.current_piece.clone()
    }

    pub fn get_next_piece(&self) -> Option<Piece> {
        self.next_piece.clone()
    }

    pub fn get_color_for_type(&self, piece_type: usize) -> (u8, u8, u8) {
        COLORS[piece_type]
    }

    fn spawn_piece(&mut self) {
        let mut rng = rand::thread_rng();

        if self.next_piece.is_none() {
            self.next_piece = Some(Piece::new(rng.gen_range(0..7)));
        }

        self.current_piece = self.next_piece.take();
        self.next_piece = Some(Piece::new(rng.gen_range(0..7)));

        // Reset position
        if let Some(ref mut piece) = self.current_piece {
            piece.x = (self.width as i32 - 4) / 2;
            piece.y = 0;
        }

        // Check if spawn position is valid
        if let Some(ref piece) = self.current_piece {
            if !self.is_valid_position_for(piece) {
                self.game_over = true;
            }
        }
    }

    fn is_valid_position_for(&self, piece: &Piece) -> bool {
        for (x, y) in get_cells_for_piece(piece) {
            if x < 0 || x >= self.width as i32 || y >= self.height as i32 {
                return false;
            }
            if y >= 0 && self.board[y as usize][x as usize] != 0 {
                return false;
            }
        }
        true
    }

    pub fn move_left(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x -= 1;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                return true;
            }
        }
        false
    }

    pub fn move_right(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.x += 1;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                return true;
            }
        }
        false
    }

    pub fn move_down(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.y += 1;
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                return true;
            }
        }
        // Could not move down, lock the piece
        self.lock_piece();
        false
    }

    pub fn hard_drop(&mut self) -> u32 {
        if self.game_over {
            return 0;
        }
        let mut drop_distance = 0;
        while self.move_down() {
            drop_distance += 1;
        }
        self.score += drop_distance * 2;
        drop_distance
    }

    pub fn rotate_cw(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.shape = rotate_shape(test_piece.shape, true);
            test_piece.rotation = (test_piece.rotation + 1) % 4;

            // Try original position first
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                return true;
            }

            // Try wall kicks
            let kicks: [(i32, i32); 6] = [(-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, -2)];
            for (dx, dy) in kicks.iter() {
                let mut kicked = test_piece.clone();
                kicked.x += dx;
                kicked.y += dy;
                if self.is_valid_position_for(&kicked) {
                    self.current_piece = Some(kicked);
                    return true;
                }
            }
        }
        false
    }

    pub fn rotate_ccw(&mut self) -> bool {
        if self.game_over {
            return false;
        }
        if let Some(ref piece) = self.current_piece {
            let mut test_piece = piece.clone();
            test_piece.shape = rotate_shape(test_piece.shape, false);
            test_piece.rotation = (test_piece.rotation + 3) % 4;

            // Try original position first
            if self.is_valid_position_for(&test_piece) {
                self.current_piece = Some(test_piece);
                return true;
            }

            // Try wall kicks
            let kicks: [(i32, i32); 6] = [(1, 0), (-1, 0), (2, 0), (-2, 0), (0, -1), (0, -2)];
            for (dx, dy) in kicks.iter() {
                let mut kicked = test_piece.clone();
                kicked.x += dx;
                kicked.y += dy;
                if self.is_valid_position_for(&kicked) {
                    self.current_piece = Some(kicked);
                    return true;
                }
            }
        }
        false
    }

    fn lock_piece(&mut self) {
        if let Some(ref piece) = self.current_piece.clone() {
            for (x, y) in get_cells_for_piece(&piece) {
                if y >= 0 && y < self.height as i32 && x >= 0 && x < self.width as i32 {
                    self.board[y as usize][x as usize] = 1;
                    self.board_colors[y as usize][x as usize] = Some(piece.piece_type);
                }
            }
            self.clear_lines();
            self.spawn_piece();
        }
    }

    fn clear_lines(&mut self) {
        let mut lines_to_clear = Vec::new();

        for y in 0..self.height {
            if self.board[y].iter().all(|&cell| cell != 0) {
                lines_to_clear.push(y);
            }
        }

        let num_lines = lines_to_clear.len() as u32;

        // Remove cleared lines
        for &y in lines_to_clear.iter().rev() {
            self.board.remove(y);
            self.board_colors.remove(y);
        }

        // Add new empty lines at top
        for _ in 0..num_lines {
            self.board.insert(0, vec![0; self.width]);
            self.board_colors.insert(0, vec![None; self.width]);
        }

        // Update score
        self.lines_cleared += num_lines;
        self.score += match num_lines {
            1 => 100 * self.level,
            2 => 300 * self.level,
            3 => 500 * self.level,
            4 => 800 * self.level,
            _ => 0,
        };

        // Update level
        self.level = (self.lines_cleared / 10) + 1;
    }

    pub fn step(&mut self, action: u8) -> (u32, bool) {
        // Actions: 0=nothing, 1=left, 2=right, 3=down, 4=rotate_cw, 5=rotate_ccw, 6=hard_drop
        let old_score = self.score;

        match action {
            1 => { self.move_left(); }
            2 => { self.move_right(); }
            3 => { self.move_down(); }
            4 => { self.rotate_cw(); }
            5 => { self.rotate_ccw(); }
            6 => { self.hard_drop(); }
            _ => {}
        }

        let reward = self.score - old_score;
        (reward, self.game_over)
    }

    pub fn tick(&mut self) -> bool {
        self.move_down()
    }

    pub fn get_ghost_piece(&self) -> Option<Piece> {
        if let Some(ref piece) = self.current_piece {
            let mut ghost = piece.clone();
            while self.is_valid_position_for(&ghost) {
                ghost.y += 1;
            }
            ghost.y -= 1;
            Some(ghost)
        } else {
            None
        }
    }

    pub fn clone_state(&self) -> TetrisEnv {
        self.clone()
    }
}

#[pymodule]
fn tetris_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TetrisEnv>()?;
    m.add_class::<Piece>()?;
    Ok(())
}
