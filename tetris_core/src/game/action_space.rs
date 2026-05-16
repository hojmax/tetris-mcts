//! Canonical action space for Tetris MCTS.
//!
//! Placement actions are indexed by canonical `(rotation, grid_x, grid_y)` cells that
//! match the normalized policy-grid visualizer contract:
//! - redundant piece rotations are collapsed (`O -> 0`, `I/S/Z -> 0/1`)
//! - permanently inactive cells are removed from the 4 x 20 x 10 grid
//! - hold remains the final action index

use std::sync::OnceLock;

use crate::game::constants::{
    BOARD_HEIGHT, BOARD_WIDTH, I_PIECE, NUM_PIECE_TYPES, O_PIECE, S_PIECE, Z_PIECE,
};
use crate::game::piece::TETROMINO_CELLS;

/// Global cached ActionSpace (initialized once on first use)
static ACTION_SPACE: OnceLock<ActionSpace> = OnceLock::new();

/// Number of canonical placement actions in the action space.
pub const NUM_PLACEMENT_ACTIONS: usize = 671;
/// Dedicated hold action index.
pub const HOLD_ACTION_INDEX: usize = NUM_PLACEMENT_ACTIONS;
/// Total number of actions in the action space.
pub const NUM_ACTIONS: usize = NUM_PLACEMENT_ACTIONS + 1;

/// Legacy `(x, y, rotation)` placement count kept for replay adaptation.
pub const LEGACY_NUM_PLACEMENT_ACTIONS: usize = 734;
/// Legacy hold action index kept for replay adaptation.
pub const LEGACY_HOLD_ACTION_INDEX: usize = LEGACY_NUM_PLACEMENT_ACTIONS;
/// Legacy action count kept for replay adaptation.
pub const LEGACY_NUM_ACTIONS: usize = LEGACY_NUM_PLACEMENT_ACTIONS + 1;

const X_MIN: i32 = -3;
const X_MAX_EXCLUSIVE: i32 = 10;
const Y_MIN: i32 = -3;
const Y_MAX_EXCLUSIVE: i32 = 20;
const X_RANGE: usize = (X_MAX_EXCLUSIVE - X_MIN) as usize;
const Y_RANGE: usize = (Y_MAX_EXCLUSIVE - Y_MIN) as usize;
const LOOKUP_SIZE: usize = X_RANGE * Y_RANGE * 4;
const FULL_GRID_SLOTS: usize = 4 * BOARD_HEIGHT * BOARD_WIDTH;

/// Canonical normalized placement cell `(rotation, grid_x, grid_y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalActionCell {
    pub rotation: usize,
    pub grid_x: i32,
    pub grid_y: i32,
}

/// Global action-space mapping utilities.
#[derive(Clone)]
pub struct ActionSpace {
    pub action_to_cell: Vec<CanonicalActionCell>,
    cell_to_action: Vec<Option<usize>>,
    piece_placement_to_action: Vec<Option<usize>>,
    legacy_action_to_new: Vec<Option<usize>>,
}

impl ActionSpace {
    pub fn new() -> Self {
        let mut action_to_cell = Vec::with_capacity(NUM_PLACEMENT_ACTIONS);
        let mut cell_to_action = vec![None; FULL_GRID_SLOTS];

        for rotation in 0..4 {
            for grid_y in 0..BOARD_HEIGHT as i32 {
                for grid_x in 0..BOARD_WIDTH as i32 {
                    if !Self::is_active_canonical_cell(rotation, grid_x, grid_y) {
                        continue;
                    }
                    let flat_index = Self::grid_flat_index(rotation, grid_x, grid_y)
                        .expect("canonical grid cell must be in bounds");
                    cell_to_action[flat_index] = Some(action_to_cell.len());
                    action_to_cell.push(CanonicalActionCell {
                        rotation,
                        grid_x,
                        grid_y,
                    });
                }
            }
        }

        assert_eq!(
            action_to_cell.len(),
            NUM_PLACEMENT_ACTIONS,
            "canonical action-space build drifted"
        );

        let legacy_positions = Self::build_legacy_action_positions();
        let mut piece_placement_to_action = vec![None; NUM_PIECE_TYPES * LOOKUP_SIZE];
        let mut legacy_action_to_new = vec![None; NUM_PIECE_TYPES * LEGACY_NUM_PLACEMENT_ACTIONS];

        for piece_type in 0..NUM_PIECE_TYPES {
            for (legacy_action_index, &(x, y, rotation)) in legacy_positions.iter().enumerate() {
                let Some(action_index) = Self::map_piece_placement_to_action(
                    piece_type,
                    x,
                    y,
                    rotation,
                    &cell_to_action,
                ) else {
                    continue;
                };

                legacy_action_to_new
                    [piece_type * LEGACY_NUM_PLACEMENT_ACTIONS + legacy_action_index] =
                    Some(action_index);

                let lookup_index = Self::lookup_index(x, y, rotation)
                    .expect("legacy action positions must stay within lookup bounds");
                piece_placement_to_action[piece_type * LOOKUP_SIZE + lookup_index] =
                    Some(action_index);
            }
        }

        Self {
            action_to_cell,
            cell_to_action,
            piece_placement_to_action,
            legacy_action_to_new,
        }
    }

    pub fn placement_to_index(
        &self,
        piece_type: usize,
        x: i32,
        y: i32,
        rotation: usize,
    ) -> Option<usize> {
        if piece_type >= NUM_PIECE_TYPES {
            return None;
        }
        let lookup_index = Self::lookup_index(x, y, rotation)?;
        self.piece_placement_to_action[piece_type * LOOKUP_SIZE + lookup_index]
    }

    pub fn legacy_action_to_index(
        &self,
        piece_type: usize,
        legacy_action_index: usize,
    ) -> Option<usize> {
        if piece_type >= NUM_PIECE_TYPES || legacy_action_index >= LEGACY_NUM_PLACEMENT_ACTIONS {
            return None;
        }
        self.legacy_action_to_new[piece_type * LEGACY_NUM_PLACEMENT_ACTIONS + legacy_action_index]
    }

    pub fn cell_to_index(&self, rotation: usize, grid_x: i32, grid_y: i32) -> Option<usize> {
        let flat_index = Self::grid_flat_index(rotation, grid_x, grid_y)?;
        self.cell_to_action[flat_index]
    }

    fn build_legacy_action_positions() -> Vec<(i32, i32, usize)> {
        let mut valid_positions = Vec::with_capacity(LEGACY_NUM_PLACEMENT_ACTIONS);

        for y in Y_MIN..Y_MAX_EXCLUSIVE {
            for x in X_MIN..X_MAX_EXCLUSIVE {
                for rotation in 0..4 {
                    if (0..NUM_PIECE_TYPES).any(|piece_type| {
                        Self::is_valid_position_empty_board(piece_type, rotation, x, y)
                    }) {
                        valid_positions.push((x, y, rotation));
                    }
                }
            }
        }

        valid_positions.sort_by_key(|&(x, y, rotation)| (rotation, y, x));
        assert_eq!(
            valid_positions.len(),
            LEGACY_NUM_PLACEMENT_ACTIONS,
            "legacy action-space position build drifted"
        );
        valid_positions
    }

    fn map_piece_placement_to_action(
        piece_type: usize,
        x: i32,
        y: i32,
        rotation: usize,
        cell_to_action: &[Option<usize>],
    ) -> Option<usize> {
        if !Self::is_valid_position_empty_board(piece_type, rotation, x, y) {
            return None;
        }

        let (min_dx, min_dy) = Self::piece_min_offsets(piece_type, rotation);
        let grid_x = x + min_dx;
        let grid_y = y + min_dy;
        let canonical_rotation = Self::canonical_rotation(piece_type, rotation);
        let flat_index = Self::grid_flat_index(canonical_rotation, grid_x, grid_y)?;
        cell_to_action[flat_index]
    }

    fn is_active_canonical_cell(rotation: usize, grid_x: i32, grid_y: i32) -> bool {
        (0..NUM_PIECE_TYPES).any(|piece_type| {
            Self::is_valid_canonical_cell_for_piece(piece_type, rotation, grid_x, grid_y)
        })
    }

    fn is_valid_canonical_cell_for_piece(
        piece_type: usize,
        rotation: usize,
        grid_x: i32,
        grid_y: i32,
    ) -> bool {
        if Self::is_redundant_rotation(piece_type, rotation) {
            return false;
        }

        let (min_dx, min_dy) = Self::piece_min_offsets(piece_type, rotation);
        let x = grid_x - min_dx;
        let y = grid_y - min_dy;
        Self::is_valid_position_empty_board(piece_type, rotation, x, y)
    }

    fn is_valid_position_empty_board(piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        let offsets = &TETROMINO_CELLS[piece_type][rotation];
        for &(dx, dy) in offsets {
            let board_x = x + dx as i32;
            let board_y = y + dy as i32;
            if board_x < 0
                || board_x >= BOARD_WIDTH as i32
                || board_y < 0
                || board_y >= BOARD_HEIGHT as i32
            {
                return false;
            }
        }
        true
    }

    fn piece_min_offsets(piece_type: usize, rotation: usize) -> (i32, i32) {
        let offsets = &TETROMINO_CELLS[piece_type][rotation];
        let mut min_dx = i32::MAX;
        let mut min_dy = i32::MAX;
        for &(dx, dy) in offsets {
            min_dx = min_dx.min(dx as i32);
            min_dy = min_dy.min(dy as i32);
        }
        (min_dx, min_dy)
    }

    #[inline]
    fn is_redundant_rotation(piece_type: usize, rotation: usize) -> bool {
        if piece_type == O_PIECE {
            return rotation >= 1;
        }
        if matches!(piece_type, I_PIECE | S_PIECE | Z_PIECE) {
            return rotation >= 2;
        }
        false
    }

    #[inline]
    fn canonical_rotation(piece_type: usize, rotation: usize) -> usize {
        if piece_type == O_PIECE {
            return 0;
        }
        if matches!(piece_type, I_PIECE | S_PIECE | Z_PIECE) {
            return rotation % 2;
        }
        rotation
    }

    #[inline]
    fn grid_flat_index(rotation: usize, grid_x: i32, grid_y: i32) -> Option<usize> {
        if rotation >= 4 {
            return None;
        }
        if !(0..BOARD_WIDTH as i32).contains(&grid_x) {
            return None;
        }
        if !(0..BOARD_HEIGHT as i32).contains(&grid_y) {
            return None;
        }
        Some(
            rotation * BOARD_HEIGHT * BOARD_WIDTH + grid_y as usize * BOARD_WIDTH + grid_x as usize,
        )
    }

    #[inline]
    fn lookup_index(x: i32, y: i32, rotation: usize) -> Option<usize> {
        if !(X_MIN..X_MAX_EXCLUSIVE).contains(&x) {
            return None;
        }
        if !(Y_MIN..Y_MAX_EXCLUSIVE).contains(&y) {
            return None;
        }
        if rotation >= 4 {
            return None;
        }

        let x_offset = (x - X_MIN) as usize;
        let y_offset = (y - Y_MIN) as usize;
        Some((rotation * Y_RANGE + y_offset) * X_RANGE + x_offset)
    }
}

impl Default for ActionSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global cached action space (initialized once on first use).
pub fn get_action_space() -> &'static ActionSpace {
    ACTION_SPACE.get_or_init(ActionSpace::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_space_size_and_rotation_counts() {
        let action_space = ActionSpace::new();
        assert_eq!(action_space.action_to_cell.len(), NUM_PLACEMENT_ACTIONS);

        let mut rotation_counts = [0usize; 4];
        for cell in &action_space.action_to_cell {
            rotation_counts[cell.rotation] += 1;
        }
        assert_eq!(rotation_counts, [178, 179, 152, 162]);
    }

    #[test]
    fn test_piece_valid_action_counts() {
        let action_space = ActionSpace::new();
        let counts = (0..NUM_PIECE_TYPES)
            .map(|piece_type| {
                action_space
                    .piece_placement_to_action
                    .chunks_exact(LOOKUP_SIZE)
                    .nth(piece_type)
                    .expect("piece chunk should exist")
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            })
            .collect::<Vec<_>>();
        assert_eq!(counts, vec![310, 171, 628, 314, 314, 628, 628]);
    }

    #[test]
    fn test_piece_valid_actions_use_only_canonical_rotations() {
        let action_space = ActionSpace::new();
        let per_piece_rotation_counts = (0..NUM_PIECE_TYPES)
            .map(|piece_type| {
                let mut counts = [0usize; 4];
                for action_index in action_space
                    .piece_placement_to_action
                    .chunks_exact(LOOKUP_SIZE)
                    .nth(piece_type)
                    .expect("piece chunk should exist")
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    let rotation = action_space.action_to_cell[action_index].rotation;
                    counts[rotation] += 1;
                }
                counts
            })
            .collect::<Vec<_>>();

        assert_eq!(
            per_piece_rotation_counts,
            vec![
                [140, 170, 0, 0],
                [171, 0, 0, 0],
                [152, 162, 152, 162],
                [152, 162, 0, 0],
                [152, 162, 0, 0],
                [152, 162, 152, 162],
                [152, 162, 152, 162],
            ]
        );
    }

    #[test]
    fn test_redundant_rotations_map_to_same_canonical_index() {
        let action_space = ActionSpace::new();
        let i_target = (0..LEGACY_NUM_PLACEMENT_ACTIONS)
            .filter_map(|legacy_action_index| {
                action_space.legacy_action_to_index(I_PIECE, legacy_action_index)
            })
            .collect::<Vec<_>>();
        let i_duplicate_target = i_target
            .iter()
            .copied()
            .find(|&target_action| {
                i_target
                    .iter()
                    .filter(|&&candidate| candidate == target_action)
                    .count()
                    == 2
            })
            .expect("I piece should have a duplicated legacy mapping");
        let i_duplicate_count = i_target
            .iter()
            .filter(|&&candidate| candidate == i_duplicate_target)
            .count();
        assert_eq!(i_duplicate_count, 2);

        let o_target = (0..LEGACY_NUM_PLACEMENT_ACTIONS)
            .filter_map(|legacy_action_index| {
                action_space.legacy_action_to_index(O_PIECE, legacy_action_index)
            })
            .collect::<Vec<_>>();
        let o_duplicate_target = o_target
            .iter()
            .copied()
            .find(|&target_action| {
                o_target
                    .iter()
                    .filter(|&&candidate| candidate == target_action)
                    .count()
                    == 4
            })
            .expect("O piece should have a quadrupled legacy mapping");
        let o_duplicate_count = o_target
            .iter()
            .filter(|&&candidate| candidate == o_duplicate_target)
            .count();
        assert_eq!(o_duplicate_count, 4);
    }

    #[test]
    fn test_legacy_action_mapping_width() {
        let action_space = ActionSpace::new();
        for piece_type in 0..NUM_PIECE_TYPES {
            let mapped = (0..LEGACY_NUM_PLACEMENT_ACTIONS)
                .filter_map(|legacy_action_index| {
                    action_space.legacy_action_to_index(piece_type, legacy_action_index)
                })
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let expected = match piece_type {
                I_PIECE => 310,
                O_PIECE => 171,
                2 => 628,
                S_PIECE | Z_PIECE => 314,
                _ => 628,
            };
            assert_eq!(mapped, expected);
        }
    }

    #[test]
    fn test_hold_action_index_is_after_placements() {
        assert_eq!(HOLD_ACTION_INDEX, NUM_PLACEMENT_ACTIONS);
        assert_eq!(NUM_ACTIONS, NUM_PLACEMENT_ACTIONS + 1);
    }
}
