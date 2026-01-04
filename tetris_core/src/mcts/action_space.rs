//! Action Space for Tetris MCTS
//!
//! Maps (x, y, rotation) placements to action indices 0-733.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::piece::NUM_PIECE_TYPES;

/// Global cached ActionSpace (initialized once on first use)
static ACTION_SPACE: OnceLock<ActionSpace> = OnceLock::new();

/// Number of actions in the action space
pub const NUM_ACTIONS: usize = 734;

/// Action index mapping
/// Maps (x, y, rotation) to action index 0-733
/// Built at module load time to match Python's action_space.py
#[derive(Clone)]
pub struct ActionSpace {
    pub action_to_placement: Vec<(i32, i32, usize)>, // (x, y, rotation)
    pub placement_to_action: HashMap<(i32, i32, usize), usize>,
}

impl ActionSpace {
    pub fn new() -> Self {
        let mut action_to_placement = Vec::new();
        let mut placement_to_action = HashMap::new();

        // Same logic as Python's action_space.py
        let x_min = -3i32;
        let x_max = 10i32;
        let y_min = -3i32;
        let y_max = 20i32;

        // Check which positions are valid for at least one piece
        let mut valid_positions: Vec<(i32, i32, usize)> = Vec::new();

        for y in y_min..y_max {
            for x in x_min..x_max {
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
            placement_to_action.insert(*pos, idx);
        }

        ActionSpace {
            action_to_placement,
            placement_to_action,
        }
    }

    fn is_valid_position_empty_board(piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        let shape = &crate::piece::TETROMINOS[piece_type][rotation];
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

    pub fn num_actions(&self) -> usize {
        self.action_to_placement.len()
    }

    pub fn placement_to_index(&self, x: i32, y: i32, rotation: usize) -> Option<usize> {
        self.placement_to_action.get(&(x, y, rotation)).copied()
    }

    pub fn index_to_placement(&self, idx: usize) -> Option<(i32, i32, usize)> {
        self.action_to_placement.get(idx).copied()
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
        assert_eq!(action_space.num_actions(), 734);

        // Test roundtrip
        for (idx, &(x, y, rot)) in action_space.action_to_placement.iter().enumerate() {
            let idx2 = action_space.placement_to_index(x, y, rot).unwrap();
            assert_eq!(idx, idx2);
        }
    }
}
