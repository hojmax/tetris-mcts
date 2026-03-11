//! Action Space for Tetris MCTS
//!
//! Maps (x, y, rotation) placements to action indices and defines an explicit hold action.

use std::sync::OnceLock;

use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES};

/// Global cached ActionSpace (initialized once on first use)
static ACTION_SPACE: OnceLock<ActionSpace> = OnceLock::new();

/// Number of placement actions in the action space
pub const NUM_PLACEMENT_ACTIONS: usize = 734;
/// Dedicated hold action index.
pub const HOLD_ACTION_INDEX: usize = NUM_PLACEMENT_ACTIONS;
/// Total number of actions in the action space.
pub const NUM_ACTIONS: usize = NUM_PLACEMENT_ACTIONS + 1;

const X_MIN: i32 = -3;
const X_MAX_EXCLUSIVE: i32 = 10;
const Y_MIN: i32 = -3;
const Y_MAX_EXCLUSIVE: i32 = 20;
const X_RANGE: usize = (X_MAX_EXCLUSIVE - X_MIN) as usize;
const Y_RANGE: usize = (Y_MAX_EXCLUSIVE - Y_MIN) as usize;
const LOOKUP_SIZE: usize = X_RANGE * Y_RANGE * 4;

/// Placement action index mapping
/// Maps (x, y, rotation) to placement action indices 0..NUM_PLACEMENT_ACTIONS-1
/// Built at module load time to match Python's action_space.py
#[derive(Clone)]
pub struct ActionSpace {
    pub action_to_placement: Vec<(i32, i32, usize)>, // (x, y, rotation)
    placement_to_action: Vec<Option<usize>>,
}

impl ActionSpace {
    pub fn new() -> Self {
        let mut action_to_placement = Vec::new();
        let mut placement_to_action = vec![None; LOOKUP_SIZE];

        // Same logic as Python's action_space.py
        // Check which positions are valid for at least one piece
        let mut valid_positions: Vec<(i32, i32, usize)> = Vec::new();

        for y in Y_MIN..Y_MAX_EXCLUSIVE {
            for x in X_MIN..X_MAX_EXCLUSIVE {
                for rot in 0..4 {
                    // Check if any piece fits at this position on an empty board
                    for piece_type in 0..NUM_PIECE_TYPES {
                        if Self::is_valid_position_empty_board(piece_type, rot, x, y) {
                            valid_positions.push((x, y, rot));
                            break;
                        }
                    }
                }
            }
        }

        // Sort by rotation, then y, then x (to match Python)
        valid_positions.sort_by_key(|&(x, y, rot)| (rot, y, x));

        for (idx, pos) in valid_positions.iter().enumerate() {
            action_to_placement.push(*pos);
            let lookup_idx = Self::lookup_index(pos.0, pos.1, pos.2)
                .expect("BUG: action-space valid position is outside lookup bounds");
            placement_to_action[lookup_idx] = Some(idx);
        }

        ActionSpace {
            action_to_placement,
            placement_to_action,
        }
    }

    fn is_valid_position_empty_board(piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        let shape = &crate::game::piece::TETROMINOS[piece_type][rotation];
        for dy in 0..4 {
            for dx in 0..4 {
                if shape[dy][dx] == 1 {
                    let cx = x + dx as i32;
                    let cy = y + dy as i32;
                    if cx < 0 || cx >= BOARD_WIDTH as i32 || cy < 0 || cy >= BOARD_HEIGHT as i32 {
                        return false;
                    }
                }
            }
        }
        true
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

    pub fn placement_to_index(&self, x: i32, y: i32, rotation: usize) -> Option<usize> {
        let lookup_idx = Self::lookup_index(x, y, rotation)?;
        self.placement_to_action[lookup_idx]
    }
}

impl Default for ActionSpace {
    fn default() -> Self {
        ActionSpace::new()
    }
}

/// Get the global cached action space (initialized once on first use)
pub fn get_action_space() -> &'static ActionSpace {
    ACTION_SPACE.get_or_init(ActionSpace::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_space() {
        let action_space = ActionSpace::new();
        assert_eq!(
            action_space.action_to_placement.len(),
            NUM_PLACEMENT_ACTIONS
        );

        // Test roundtrip
        for (idx, &(x, y, rot)) in action_space.action_to_placement.iter().enumerate() {
            let idx2 = action_space.placement_to_index(x, y, rot).unwrap();
            assert_eq!(idx, idx2);
        }
    }

    #[test]
    fn test_hold_action_index_is_after_placements() {
        assert_eq!(HOLD_ACTION_INDEX, NUM_PLACEMENT_ACTIONS);
        assert_eq!(NUM_ACTIONS, NUM_PLACEMENT_ACTIONS + 1);
    }
}
