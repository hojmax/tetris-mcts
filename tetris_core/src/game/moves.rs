//! Move Generation Module
//!
//! This module provides functionality to find all possible piece placements
//! and the move sequences required to reach them. This is useful for AI/MCTS
//! planning and move evaluation.

use pyo3::prelude::*;
use std::collections::{HashSet, VecDeque};

use crate::game::action_space::get_action_space;
use crate::game::kicks::{get_i_kicks, get_jlstz_kicks};
use crate::game::piece::{get_cells, Piece, TETROMINO_CELLS};

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

const X_MIN: i32 = -3;
const X_STATES: usize = 16; // x in [-3, 12]
const Y_MIN: i32 = -3;
const Y_STATES: usize = 26; // y in [-3, 22]
const ROT_STATES: usize = 4;
const X_STRIDE: usize = Y_STATES * ROT_STATES;
const Y_STRIDE: usize = ROT_STATES;
const NUM_STATE_INDICES: usize = X_STATES * Y_STATES * ROT_STATES; // 1,664
const VISITED_WORDS: usize = (NUM_STATE_INDICES + 63) / 64; // 26

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
    /// The kick index used for the final rotation (0 = no kick, 1-4 = kick used)
    /// This is needed for proper T-spin detection (kick 4 = always full T-spin)
    #[pyo3(get)]
    pub last_kick_index: usize,
    /// Whether the last move before hard drop was a rotation
    #[pyo3(get)]
    pub last_move_was_rotation: bool,
    /// Pre-computed placement action index for fast lookup (0..NUM_PLACEMENT_ACTIONS-1)
    #[pyo3(get)]
    pub action_index: usize,
}

/// Compact placement metadata used by MCTS action execution/cache paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementParams {
    pub x: i32,
    pub y: i32,
    pub rotation: usize,
    pub last_kick_index: usize,
    pub last_move_was_rotation: bool,
    pub action_index: usize,
}

/// Board representation for collision checking (borrows cells to avoid cloning)
pub struct Board<'a> {
    width: usize,
    width_u32: u32,
    height_u32: u32,
    cells: &'a [u8],
    column_masks: Option<Vec<u32>>,
}

impl<'a> Board<'a> {
    pub fn new(width: usize, height: usize, cells: &'a [u8]) -> Self {
        debug_assert!(width <= u32::MAX as usize);
        debug_assert!(height <= u32::MAX as usize);
        let column_masks = if height <= u32::BITS as usize {
            let mut masks = vec![0u32; width];
            for y in 0..height {
                let row = &cells[y * width..(y + 1) * width];
                let row_bit = 1u32 << (y as u32);
                for (x, &cell) in row.iter().enumerate() {
                    if cell != 0 {
                        masks[x] |= row_bit;
                    }
                }
            }
            Some(masks)
        } else {
            None
        };
        Board {
            width,
            width_u32: width as u32,
            height_u32: height as u32,
            cells,
            column_masks,
        }
    }

    /// Check if a piece at the given state is in a valid position
    fn is_valid_position(&self, piece_type: usize, state: &PieceState) -> bool {
        self.is_valid_position_at(piece_type, state.rotation, state.x, state.y)
    }

    /// Get the final y position after hard dropping from a state
    fn get_drop_y(&self, piece_type: usize, state: &PieceState) -> i32 {
        if let Some(column_masks) = &self.column_masks {
            let offsets = &TETROMINO_CELLS[piece_type][state.rotation];
            let mut min_drop = i32::MAX;
            for &(dx, dy) in offsets.iter() {
                let cx = state.x + dx as i32;
                let cy = state.y + dy as i32;
                debug_assert!((cx as u32) < self.width_u32);
                debug_assert!((cy as u32) < self.height_u32);

                let search_start = cy as u32 + 1;
                let next_occupied_row = if search_start >= self.height_u32 {
                    self.height_u32 as i32
                } else {
                    let shifted = column_masks[cx as usize] >> search_start;
                    if shifted == 0 {
                        self.height_u32 as i32
                    } else {
                        (search_start + shifted.trailing_zeros()) as i32
                    }
                };
                let cell_drop = next_occupied_row - 1 - cy;
                min_drop = min_drop.min(cell_drop);
            }
            debug_assert!(min_drop >= 0);
            return state.y + min_drop;
        }

        // Intentionally step downward with exact collision checks.
        // In move-generation BFS we evaluate arbitrary intermediate states
        // (including kick/slide positions under overhangs), where a simple
        // column-height profile shortcut is not always correct.
        let mut y = state.y;
        while self.is_valid_position_at(piece_type, state.rotation, state.x, y + 1) {
            y += 1;
        }
        y
    }

    #[inline]
    fn is_valid_position_at(&self, piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        let [(dx0, dy0), (dx1, dy1), (dx2, dy2), (dx3, dy3)] =
            TETROMINO_CELLS[piece_type][rotation];
        let cells_ptr = self.cells.as_ptr();

        let cx0 = x + dx0 as i32;
        let cy0 = y + dy0 as i32;
        if (cx0 as u32) >= self.width_u32 || (cy0 as u32) >= self.height_u32 {
            return false;
        }
        let idx0 = cy0 as usize * self.width + cx0 as usize;
        if unsafe { *cells_ptr.add(idx0) } != 0 {
            return false;
        }

        let cx1 = x + dx1 as i32;
        let cy1 = y + dy1 as i32;
        if (cx1 as u32) >= self.width_u32 || (cy1 as u32) >= self.height_u32 {
            return false;
        }
        let idx1 = cy1 as usize * self.width + cx1 as usize;
        if unsafe { *cells_ptr.add(idx1) } != 0 {
            return false;
        }

        let cx2 = x + dx2 as i32;
        let cy2 = y + dy2 as i32;
        if (cx2 as u32) >= self.width_u32 || (cy2 as u32) >= self.height_u32 {
            return false;
        }
        let idx2 = cy2 as usize * self.width + cx2 as usize;
        if unsafe { *cells_ptr.add(idx2) } != 0 {
            return false;
        }

        let cx3 = x + dx3 as i32;
        let cy3 = y + dy3 as i32;
        if (cx3 as u32) >= self.width_u32 || (cy3 as u32) >= self.height_u32 {
            return false;
        }
        let idx3 = cy3 as usize * self.width + cx3 as usize;
        if unsafe { *cells_ptr.add(idx3) } != 0 {
            return false;
        }

        true
    }
}

/// Try to rotate a piece and return the new state if successful
/// Returns (new_state, kick_index) where kick_index is 0-4 indicating which kick was used
fn try_rotate(
    board: &Board<'_>,
    piece_type: usize,
    state: &PieceState,
    clockwise: bool,
) -> Option<(PieceState, usize)> {
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

    for (kick_index, (dx, dy)) in kicks.iter().enumerate() {
        let new_x = state.x + dx;
        let new_y = state.y + dy;
        if board.is_valid_position_at(piece_type, to_state, new_x, new_y) {
            return Some((
                PieceState {
                    x: new_x,
                    y: new_y,
                    rotation: to_state,
                },
                kick_index,
            ));
        }
    }
    None
}

/// Convert piece state to bitset index
#[inline(always)]
fn state_to_index(state: &PieceState) -> usize {
    debug_assert!(state.x >= X_MIN && state.x < X_MIN + X_STATES as i32);
    debug_assert!(state.y >= Y_MIN && state.y < Y_MIN + Y_STATES as i32);
    debug_assert!(state.rotation < ROT_STATES);

    let x_offset = (state.x - X_MIN) as usize;
    let y_offset = (state.y - Y_MIN) as usize;
    x_offset * X_STRIDE + y_offset * Y_STRIDE + state.rotation
}

#[inline]
fn index_to_state(idx: usize) -> PieceState {
    let x_offset = idx / X_STRIDE;
    let rem = idx % X_STRIDE;
    let y_offset = rem / ROT_STATES;
    let rotation = rem % ROT_STATES;
    PieceState {
        x: X_MIN + x_offset as i32,
        y: Y_MIN + y_offset as i32,
        rotation,
    }
}

#[inline]
fn prefer_tspin_safe_path(
    candidate_last_move_was_rotation: bool,
    existing_last_move_was_rotation: bool,
) -> bool {
    if candidate_last_move_was_rotation != existing_last_move_was_rotation {
        // Prefer equivalent paths that don't end on rotation to avoid
        // accidental T-spin classification when both paths are equally short.
        return !candidate_last_move_was_rotation;
    }
    false
}

/// Find all possible placements for a piece on the given board
///
/// Uses BFS to explore all reachable states from the spawn position,
/// then returns all unique final positions (after hard drop) with
/// the shortest move sequence to reach each.
pub fn find_all_placements(
    board: &Board<'_>,
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

    let mut visited = [0u64; VISITED_WORDS];
    let mut queue: VecDeque<PieceState> = VecDeque::with_capacity(128);

    // Parent-pointer BFS metadata per reachable state.
    // This avoids cloning move sequences at each edge expansion.
    let mut parents = [usize::MAX; NUM_STATE_INDICES];
    let mut action_from_parent = [0u8; NUM_STATE_INDICES];
    let mut depth = [u16::MAX; NUM_STATE_INDICES];
    let mut last_kick_index = [0u8; NUM_STATE_INDICES];
    let mut last_move_was_rotation = [false; NUM_STATE_INDICES];

    // For each final dropped state index, track the source BFS state index
    // that reaches it with the shortest path.
    let mut final_best_source = [None; NUM_STATE_INDICES];
    let mut final_best_depth = [u16::MAX; NUM_STATE_INDICES];

    // Mark start state as visited
    let start_idx = state_to_index(&start_state);
    visited[start_idx / 64] |= 1u64 << (start_idx % 64);
    parents[start_idx] = start_idx;
    depth[start_idx] = 0;
    queue.push_back(start_state);

    let transitions = [
        (Action::Left, -1, 0, None),
        (Action::Right, 1, 0, None),
        (Action::Down, 0, 1, None),
        (Action::RotateCW, 0, 0, Some(true)),
        (Action::RotateCCW, 0, 0, Some(false)),
    ];

    while let Some(state) = queue.pop_front() {
        let state_idx = state_to_index(&state);

        // Record this state's final position (after hard drop), keeping shortest source path.
        let final_y = board.get_drop_y(piece_type, &state);
        let final_state = PieceState {
            x: state.x,
            y: final_y,
            rotation: state.rotation,
        };
        let final_idx = state_to_index(&final_state);
        if depth[state_idx] < final_best_depth[final_idx] {
            final_best_depth[final_idx] = depth[state_idx];
            final_best_source[final_idx] = Some(state_idx);
        } else if depth[state_idx] == final_best_depth[final_idx] {
            if let Some(existing_source_idx) = final_best_source[final_idx] {
                if prefer_tspin_safe_path(
                    last_move_was_rotation[state_idx],
                    last_move_was_rotation[existing_source_idx],
                ) {
                    final_best_source[final_idx] = Some(state_idx);
                }
            }
        }

        // Try all possible moves
        for &(action, dx, dy, rotate) in &transitions {
            let (new_state, kick_index, is_rotation) = if let Some(clockwise) = rotate {
                // Rotation with wall kicks - returns state and kick index
                match try_rotate(board, piece_type, &state, clockwise) {
                    Some((state, kick)) => (Some(state), kick as u8, true),
                    None => (None, 0, false),
                }
            } else {
                // Simple translation - no kick
                let candidate = PieceState {
                    x: state.x + dx,
                    y: state.y + dy,
                    rotation: state.rotation,
                };
                if board.is_valid_position(piece_type, &candidate) {
                    (Some(candidate), 0, false)
                } else {
                    (None, 0, false)
                }
            };

            let Some(new_state) = new_state else {
                continue;
            };

            // Check visited using bitset
            let idx = state_to_index(&new_state);
            let word = idx / 64;
            let bit = idx % 64;
            let is_visited = (visited[word] & (1u64 << bit)) != 0;
            let candidate_depth = depth[state_idx] + 1;
            let candidate_last_kick_index = if is_rotation {
                kick_index
            } else {
                last_kick_index[state_idx]
            };

            if !is_visited {
                // Mark as visited and record BFS metadata
                visited[word] |= 1u64 << bit;
                parents[idx] = state_idx;
                action_from_parent[idx] = action.to_u8();
                depth[idx] = candidate_depth;
                last_kick_index[idx] = candidate_last_kick_index;
                last_move_was_rotation[idx] = is_rotation;
                queue.push_back(new_state);
            } else if depth[idx] == candidate_depth
                && prefer_tspin_safe_path(is_rotation, last_move_was_rotation[idx])
            {
                parents[idx] = state_idx;
                action_from_parent[idx] = action.to_u8();
                last_kick_index[idx] = candidate_last_kick_index;
                last_move_was_rotation[idx] = is_rotation;
            }
        }
    }

    // Deduplicate by actual cells occupied (not just x, y, rotation).
    // This handles cases like O piece where different rotations look identical.
    // Use a full coordinate key (including negative values) to avoid collisions.
    let mut seen_cells: HashSet<u64> = HashSet::new();
    let mut placements: Vec<Placement> = Vec::new();
    let action_space = get_action_space();

    for (final_idx, source_idx) in final_best_source.iter().enumerate() {
        let Some(source_idx) = source_idx else {
            continue;
        };

        let final_state = index_to_state(final_idx);
        let x = final_state.x;
        let y = final_state.y;
        let rotation = final_state.rotation;

        let cells = get_cells(piece_type, rotation, x, y);
        debug_assert_eq!(cells.len(), 4, "Tetromino should have exactly 4 cells");
        let mut packed_cells = [0u16; 4];
        for (cell_idx, (cx, cy)) in cells.into_iter().enumerate() {
            let packed = cy as usize * board.width + cx as usize;
            debug_assert!(packed <= u16::MAX as usize);
            packed_cells[cell_idx] = packed as u16;
        }
        packed_cells.sort_unstable();
        let cell_key = ((packed_cells[0] as u64) << 48)
            | ((packed_cells[1] as u64) << 32)
            | ((packed_cells[2] as u64) << 16)
            | packed_cells[3] as u64;

        if !seen_cells.insert(cell_key) {
            continue;
        }

        // Reconstruct shortest path from BFS parent pointers without reverse pass.
        let path_len = depth[*source_idx] as usize;
        let mut moves = vec![0u8; path_len + 1];
        moves[path_len] = Action::HardDrop.to_u8();
        let mut cursor = *source_idx;
        let mut write_idx = path_len;
        while cursor != start_idx {
            write_idx -= 1;
            moves[write_idx] = action_from_parent[cursor];
            cursor = parents[cursor];
        }

        let Some(action_index) = action_space.placement_to_index(x, y, rotation) else {
            debug_assert!(
                false,
                "BUG: valid placement ({}, {}, {}) missing from action space",
                x, y, rotation
            );
            continue;
        };

        placements.push(Placement {
            piece: Piece::with_position(piece_type, x, y, rotation),
            moves,
            column: x,
            rotation,
            last_kick_index: last_kick_index[*source_idx] as usize,
            last_move_was_rotation: last_move_was_rotation[*source_idx],
            action_index,
        });
    }

    // Sort by rotation, column, then y for fully deterministic ordering
    placements.sort_unstable_by(|a, b| {
        a.rotation
            .cmp(&b.rotation)
            .then_with(|| a.column.cmp(&b.column))
            .then_with(|| a.piece.y.cmp(&b.piece.y))
    });

    placements
}

/// Find all possible placements needed by MCTS.
///
/// This variant skips move-sequence reconstruction and returns only placement
/// parameters used by action execution and valid-action masking.
pub fn find_all_placement_params(
    board: &Board<'_>,
    piece_type: usize,
    spawn_x: i32,
    spawn_y: i32,
) -> Vec<PlacementParams> {
    let start_state = PieceState {
        x: spawn_x,
        y: spawn_y,
        rotation: 0,
    };

    if !board.is_valid_position(piece_type, &start_state) {
        return Vec::new();
    }

    let mut visited = [0u64; VISITED_WORDS];
    let mut queue: Vec<(PieceState, usize)> = Vec::with_capacity(128);
    let mut queue_head = 0usize;
    let mut depth = [u16::MAX; NUM_STATE_INDICES];
    let mut last_kick_index = [0u8; NUM_STATE_INDICES];
    let mut last_move_was_rotation = [false; NUM_STATE_INDICES];
    let mut final_best_source = [None; NUM_STATE_INDICES];
    let mut final_best_depth = [u16::MAX; NUM_STATE_INDICES];
    let mut final_candidate_indices = Vec::with_capacity(128);

    let start_idx = state_to_index(&start_state);
    visited[start_idx / 64] |= 1u64 << (start_idx % 64);
    depth[start_idx] = 0;
    queue.push((start_state, start_idx));

    let transitions = [
        (-1, 0, None),
        (1, 0, None),
        (0, 1, None),
        (0, 0, Some(true)),
        (0, 0, Some(false)),
    ];

    while queue_head < queue.len() {
        let (state, state_idx) = queue[queue_head];
        queue_head += 1;

        let final_y = board.get_drop_y(piece_type, &state);
        let final_idx = ((state.x - X_MIN) as usize) * X_STRIDE
            + ((final_y - Y_MIN) as usize) * Y_STRIDE
            + state.rotation;
        if final_best_source[final_idx].is_none() {
            final_candidate_indices.push(final_idx);
        }
        if depth[state_idx] < final_best_depth[final_idx] {
            final_best_depth[final_idx] = depth[state_idx];
            final_best_source[final_idx] = Some(state_idx);
        } else if depth[state_idx] == final_best_depth[final_idx] {
            if let Some(existing_source_idx) = final_best_source[final_idx] {
                if prefer_tspin_safe_path(
                    last_move_was_rotation[state_idx],
                    last_move_was_rotation[existing_source_idx],
                ) {
                    final_best_source[final_idx] = Some(state_idx);
                }
            }
        }

        for &(dx, dy, rotate) in &transitions {
            let (new_state, kick_index, is_rotation) = if let Some(clockwise) = rotate {
                match try_rotate(board, piece_type, &state, clockwise) {
                    Some((state, kick)) => (Some(state), kick as u8, true),
                    None => (None, 0, false),
                }
            } else {
                let candidate = PieceState {
                    x: state.x + dx,
                    y: state.y + dy,
                    rotation: state.rotation,
                };
                if board.is_valid_position(piece_type, &candidate) {
                    (Some(candidate), 0, false)
                } else {
                    (None, 0, false)
                }
            };

            let Some(new_state) = new_state else {
                continue;
            };

            let idx = if is_rotation {
                state_to_index(&new_state)
            } else {
                match (dx, dy) {
                    (-1, 0) => {
                        debug_assert!(state_idx >= X_STRIDE);
                        state_idx - X_STRIDE
                    }
                    (1, 0) => state_idx + X_STRIDE,
                    (0, 1) => state_idx + Y_STRIDE,
                    _ => state_to_index(&new_state),
                }
            };
            let word = idx / 64;
            let bit = idx % 64;
            let is_visited = (visited[word] & (1u64 << bit)) != 0;
            let candidate_depth = depth[state_idx] + 1;
            let candidate_last_kick_index = if is_rotation {
                kick_index
            } else {
                last_kick_index[state_idx]
            };

            if !is_visited {
                visited[word] |= 1u64 << bit;
                depth[idx] = candidate_depth;
                last_kick_index[idx] = candidate_last_kick_index;
                last_move_was_rotation[idx] = is_rotation;
                queue.push((new_state, idx));
            } else if depth[idx] == candidate_depth
                && prefer_tspin_safe_path(is_rotation, last_move_was_rotation[idx])
            {
                last_kick_index[idx] = candidate_last_kick_index;
                last_move_was_rotation[idx] = is_rotation;
            }
        }
    }

    final_candidate_indices.sort_unstable();

    let mut seen_cells = [0u64; crate::search::NUM_PLACEMENT_ACTIONS];
    let mut seen_cells_len = 0usize;
    let mut placements_by_action: [Option<PlacementParams>;
        crate::game::action_space::NUM_ACTIONS] = [None; crate::game::action_space::NUM_ACTIONS];
    let action_space = get_action_space();

    for &final_idx in &final_candidate_indices {
        let Some(source_idx) = final_best_source[final_idx] else {
            continue;
        };

        let final_state = index_to_state(final_idx);
        let x = final_state.x;
        let y = final_state.y;
        let rotation = final_state.rotation;

        let cells = get_cells(piece_type, rotation, x, y);
        debug_assert_eq!(cells.len(), 4, "Tetromino should have exactly 4 cells");
        let mut packed_cells = [0u16; 4];
        for (cell_idx, (cx, cy)) in cells.into_iter().enumerate() {
            let packed = cy as usize * board.width + cx as usize;
            debug_assert!(packed <= u16::MAX as usize);
            packed_cells[cell_idx] = packed as u16;
        }
        packed_cells.sort_unstable();
        let cell_key = ((packed_cells[0] as u64) << 48)
            | ((packed_cells[1] as u64) << 32)
            | ((packed_cells[2] as u64) << 16)
            | packed_cells[3] as u64;

        if seen_cells[..seen_cells_len]
            .iter()
            .any(|&existing_key| existing_key == cell_key)
        {
            continue;
        }
        debug_assert!(
            seen_cells_len < seen_cells.len(),
            "Seen-cell cache overflow while deduplicating placements"
        );
        if seen_cells_len >= seen_cells.len() {
            continue;
        }
        seen_cells[seen_cells_len] = cell_key;
        seen_cells_len += 1;

        let Some(action_index) = action_space.placement_to_index(x, y, rotation) else {
            debug_assert!(
                false,
                "BUG: valid placement ({}, {}, {}) missing from action space",
                x, y, rotation
            );
            continue;
        };
        debug_assert!(
            placements_by_action[action_index].is_none(),
            "Duplicate action index {} in placement params",
            action_index
        );
        placements_by_action[action_index] = Some(PlacementParams {
            x,
            y,
            rotation,
            last_kick_index: last_kick_index[source_idx] as usize,
            last_move_was_rotation: last_move_was_rotation[source_idx],
            action_index,
        });
    }

    let mut placements = Vec::with_capacity(seen_cells_len);
    for placement in placements_by_action.into_iter().flatten() {
        placements.push(placement);
    }
    placements
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn empty_cells(width: usize, height: usize) -> Vec<u8> {
        vec![0u8; width * height]
    }

    #[test]
    fn test_find_placements_empty_board() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        // I piece (type 0)
        let placements = find_all_placements(&board, 0, 3, 0);
        assert!(!placements.is_empty());

        // Should have placements in different columns and rotations
        let columns: HashSet<i32> = placements.iter().map(|p| p.column).collect();
        assert!(columns.len() > 1);
    }

    #[test]
    fn test_find_placements_all_pieces() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
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
    fn test_find_placement_params_matches_full_placements() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);

        for piece_type in 0..7 {
            let full = find_all_placements(&board, piece_type, 3, 0);
            let params = find_all_placement_params(&board, piece_type, 3, 0);

            let full_by_action: HashMap<usize, &Placement> = full
                .iter()
                .map(|placement| (placement.action_index, placement))
                .collect();
            assert_eq!(
                full_by_action.len(),
                full.len(),
                "Full placements should not contain duplicate action indices"
            );
            assert_eq!(
                params.len(),
                full.len(),
                "Fast params count should match full placement count for piece {}",
                piece_type
            );

            for param in params {
                let full_match = full_by_action
                    .get(&param.action_index)
                    .copied()
                    .expect("Fast params produced an unknown action index");
                assert_eq!(param.x, full_match.piece.x);
                assert_eq!(param.y, full_match.piece.y);
                assert_eq!(param.rotation, full_match.piece.rotation);
                assert_eq!(param.last_kick_index, full_match.last_kick_index);
                assert_eq!(
                    param.last_move_was_rotation,
                    full_match.last_move_was_rotation
                );
            }
        }
    }

    #[test]
    fn test_placements_have_moves() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        for placement in &placements {
            // Every placement should end with hard drop
            assert!(!placement.moves.is_empty(), "Placement should have moves");
            assert_eq!(
                *placement.moves.last().unwrap(),
                Action::HardDrop.to_u8(),
                "Last move should be hard drop"
            );
        }
    }

    #[test]
    fn test_placements_reach_bottom() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
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
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
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
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
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
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 1, 3, 0); // O piece

        // O piece looks the same in all rotations, so we should have
        // placements that effectively only differ by column
        let unique_columns: HashSet<i32> = placements.iter().map(|p| p.column).collect();
        assert!(!unique_columns.is_empty());
    }

    #[test]
    fn test_blocked_board() {
        // Create a board with a wall blocking most positions
        let mut cells = vec![0u8; 10 * 20];
        // Fill rows 10-19 except column 0
        for y in 10..20 {
            for x in 1..10 {
                cells[y * 10 + x] = 1;
            }
        }
        let board = Board::new(10, 20, &cells);

        // I piece can only fit in column 0 when vertical
        let placements = find_all_placements(&board, 0, 3, 0);

        // Should still find some placements (vertical I in column 0)
        // or placements above the filled area
        assert!(!placements.is_empty());
    }

    #[test]
    fn test_spawn_blocked() {
        // Create a board where spawn is blocked
        let mut cells = vec![0u8; 10 * 20];
        for x in 0..10 {
            cells[0 * 10 + x] = 1;
            cells[1 * 10 + x] = 1;
        }
        let board = Board::new(10, 20, &cells);

        let placements = find_all_placements(&board, 0, 3, 0);
        assert!(
            placements.is_empty(),
            "Should have no placements when spawn is blocked"
        );
    }

    #[test]
    fn test_move_sequence_valid() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
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
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
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
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        // T piece should have placements in all 4 rotations
        let rotations: HashSet<usize> = placements.iter().map(|p| p.rotation).collect();
        assert_eq!(rotations.len(), 4, "T piece should have all 4 rotations");
    }

    #[test]
    fn test_translated_rotated_placement_prefers_non_rotation_last_move() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        let translated_rotated = placements
            .iter()
            .find(|p| p.rotation == 1 && p.column != 3)
            .expect("Expected a rotated placement that also translates");

        assert!(
            !translated_rotated.last_move_was_rotation,
            "Translated rotated placement should prefer non-rotation final move"
        );
    }

    #[test]
    fn test_pure_rotation_placement_keeps_rotation_last_move() {
        let cells = empty_cells(10, 20);
        let board = Board::new(10, 20, &cells);
        let placements = find_all_placements(&board, 2, 3, 0); // T piece

        let pure_rotation = placements
            .iter()
            .find(|p| p.rotation == 1 && p.column == 3)
            .expect("Expected a pure-rotation placement at spawn column");

        assert!(
            pure_rotation.last_move_was_rotation,
            "Pure rotation placement should keep rotation as last move"
        );
    }
}
