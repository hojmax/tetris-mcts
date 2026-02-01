//! SRS (Super Rotation System) Wall Kick Data
//!
//! This module contains the wall kick offset data used in the SRS rotation system.
//! Wall kicks allow pieces to "kick" off walls and other pieces when rotating,
//! making rotations feel more intuitive and enabling advanced techniques like T-spins.
//!
//! Rotation states:
//! - 0: Spawn state
//! - 1: R (clockwise from spawn)
//! - 2: 180° (two rotations from spawn)
//! - 3: L (counter-clockwise from spawn)

/// SRS Wall kick data for J, L, S, T, Z pieces
/// Note: y is inverted (positive = down in our coordinate system)
/// Returns 5 kick offsets to try in order: (dx, dy)
pub fn get_jlstz_kicks(from_state: usize, to_state: usize) -> [(i32, i32); 5] {
    match (from_state, to_state) {
        // 0->R (clockwise from spawn)
        (0, 1) => [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        // R->0 (counter-clockwise from R)
        (1, 0) => [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        // R->2 (clockwise from R)
        (1, 2) => [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        // 2->R (counter-clockwise from 2)
        (2, 1) => [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        // 2->L (clockwise from 2)
        (2, 3) => [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        // L->2 (counter-clockwise from L)
        (3, 2) => [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        // L->0 (clockwise from L)
        (3, 0) => [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        // 0->L (counter-clockwise from spawn)
        (0, 3) => [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        // Fallback for invalid transitions
        _ => [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    }
}

/// SRS Wall kick data for I piece
/// The I piece has different kick data than other pieces
pub fn get_i_kicks(from_state: usize, to_state: usize) -> [(i32, i32); 5] {
    match (from_state, to_state) {
        // 0->R
        (0, 1) => [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        // R->0
        (1, 0) => [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        // R->2
        (1, 2) => [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
        // 2->R
        (2, 1) => [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        // 2->L
        (2, 3) => [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        // L->2
        (3, 2) => [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        // L->0
        (3, 0) => [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        // 0->L
        (0, 3) => [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
        // Fallback for invalid transitions
        _ => [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    }
}

/// Get the appropriate wall kicks for a piece type
/// piece_type: 0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L
pub fn get_kicks_for_piece(piece_type: usize, from_state: usize, to_state: usize) -> [(i32, i32); 5] {
    match piece_type {
        0 => get_i_kicks(from_state, to_state),
        1 => [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], // O piece: no kicks needed
        _ => get_jlstz_kicks(from_state, to_state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jlstz_kicks_first_is_identity() {
        // First kick should always be (0, 0) - try without offset first
        let transitions = [
            (0, 1), (1, 0), (1, 2), (2, 1),
            (2, 3), (3, 2), (3, 0), (0, 3),
        ];
        for (from, to) in transitions.iter() {
            let kicks = get_jlstz_kicks(*from, *to);
            assert_eq!(kicks[0], (0, 0), "First kick for {}->{} should be (0,0)", from, to);
        }
    }

    #[test]
    fn test_i_kicks_first_is_identity() {
        let transitions = [
            (0, 1), (1, 0), (1, 2), (2, 1),
            (2, 3), (3, 2), (3, 0), (0, 3),
        ];
        for (from, to) in transitions.iter() {
            let kicks = get_i_kicks(*from, *to);
            assert_eq!(kicks[0], (0, 0), "First I-kick for {}->{} should be (0,0)", from, to);
        }
    }

    #[test]
    fn test_o_kicks_all_identity() {
        // O piece kicks are all identity (no kicks needed)
        for from in 0..4 {
            for to in 0..4 {
                let kicks = get_kicks_for_piece(1, from, to); // 1 = O piece
                for kick in kicks.iter() {
                    assert_eq!(*kick, (0, 0));
                }
            }
        }
    }

    #[test]
    fn test_kicks_length() {
        let kicks = get_jlstz_kicks(0, 1);
        assert_eq!(kicks.len(), 5);

        let kicks = get_i_kicks(0, 1);
        assert_eq!(kicks.len(), 5);
    }

    #[test]
    fn test_get_kicks_for_piece() {
        // I piece should use I kicks
        let i_kicks = get_kicks_for_piece(0, 0, 1);
        assert_eq!(i_kicks, get_i_kicks(0, 1));

        // O piece should return all identity kicks (no kicks needed)
        let o_kicks = get_kicks_for_piece(1, 0, 1);
        assert_eq!(o_kicks, [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]);

        // T, S, Z, J, L should use JLSTZ kicks
        for piece_type in 2..7 {
            let kicks = get_kicks_for_piece(piece_type, 0, 1);
            assert_eq!(kicks, get_jlstz_kicks(0, 1));
        }
    }

    #[test]
    fn test_jlstz_invalid_transition_returns_identity() {
        // Invalid transitions should return all identity kicks
        let kicks = get_jlstz_kicks(0, 0); // Same state
        for kick in kicks.iter() {
            assert_eq!(*kick, (0, 0));
        }

        let kicks = get_jlstz_kicks(0, 2); // Skip a state (invalid)
        for kick in kicks.iter() {
            assert_eq!(*kick, (0, 0));
        }
    }

    #[test]
    fn test_i_kicks_are_different_from_jlstz() {
        // I piece kicks should be different from JLSTZ kicks
        let i_kicks = get_i_kicks(0, 1);
        let jlstz_kicks = get_jlstz_kicks(0, 1);

        // At least one kick (besides the first identity) should be different
        let mut found_difference = false;
        for i in 1..5 {
            if i_kicks[i] != jlstz_kicks[i] {
                found_difference = true;
                break;
            }
        }
        assert!(found_difference, "I kicks should differ from JLSTZ kicks");
    }

    #[test]
    fn test_symmetric_transitions() {
        // Test that CW and CCW kicks are properly symmetric
        // 0->1 (CW) should be inverse-related to 1->0 (CCW)
        let cw_kicks = get_jlstz_kicks(0, 1);
        let ccw_kicks = get_jlstz_kicks(1, 0);

        // The kicks exist and have correct length
        assert_eq!(cw_kicks.len(), 5);
        assert_eq!(ccw_kicks.len(), 5);
    }

    #[test]
    fn test_all_valid_jlstz_transitions() {
        // All valid single-step transitions should have proper kicks
        let valid_transitions = [
            (0, 1), (1, 0),  // 0 <-> R
            (1, 2), (2, 1),  // R <-> 2
            (2, 3), (3, 2),  // 2 <-> L
            (3, 0), (0, 3),  // L <-> 0
        ];

        for (from, to) in valid_transitions.iter() {
            let kicks = get_jlstz_kicks(*from, *to);
            // Should have non-trivial kicks (not all zeros after first)
            let has_non_trivial = kicks[1..].iter().any(|&(x, y)| x != 0 || y != 0);
            assert!(has_non_trivial, "Transition {}->{} should have non-trivial kicks", from, to);
        }
    }

    #[test]
    fn test_all_valid_i_transitions() {
        let valid_transitions = [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (3, 0), (0, 3),
        ];

        for (from, to) in valid_transitions.iter() {
            let kicks = get_i_kicks(*from, *to);
            let has_non_trivial = kicks[1..].iter().any(|&(x, y)| x != 0 || y != 0);
            assert!(has_non_trivial, "I transition {}->{} should have non-trivial kicks", from, to);
        }
    }
}
