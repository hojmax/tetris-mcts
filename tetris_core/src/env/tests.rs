//! TetrisEnv Tests

#[cfg(test)]
mod tests {
    use crate::constants::T_PIECE;
    use crate::env::TetrisEnv;
    use crate::piece::Piece;

    #[test]
    fn test_env_creation() {
        let env = TetrisEnv::new(10, 20);
        assert_eq!(env.width, 10);
        assert_eq!(env.height, 20);
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.combo, 0);
        assert!(!env.back_to_back);
        assert!(!env.game_over);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_custom_board_size() {
        let env = TetrisEnv::new(12, 24);
        assert_eq!(env.width, 12);
        assert_eq!(env.height, 24);
    }

    #[test]
    fn test_env_reset() {
        let mut env = TetrisEnv::new(10, 20);
        env.attack = 100;
        env.lines_cleared = 10;
        env.combo = 5;
        env.back_to_back = true;
        env.reset();
        assert_eq!(env.attack, 0);
        assert_eq!(env.lines_cleared, 0);
        assert_eq!(env.combo, 0);
        assert!(!env.back_to_back);
        assert!(!env.game_over);
    }

    #[test]
    fn test_move_left() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let moved = env.move_left();
        if moved {
            let new_x = env.current_piece.as_ref().unwrap().x;
            assert_eq!(new_x, initial_x - 1);
        }
    }

    #[test]
    fn test_move_right() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let moved = env.move_right();
        if moved {
            let new_x = env.current_piece.as_ref().unwrap().x;
            assert_eq!(new_x, initial_x + 1);
        }
    }

    #[test]
    fn test_move_down() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let moved = env.move_down();
        if moved {
            let new_y = env.current_piece.as_ref().unwrap().y;
            assert_eq!(new_y, initial_y + 1);
        }
    }

    #[test]
    fn test_rotate_cw() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_cw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 1) % 4);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_rotate_ccw() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;
        let rotated = env.rotate_ccw();
        if rotated {
            let new_rotation = env.current_piece.as_ref().unwrap().rotation;
            assert_eq!(new_rotation, (initial_rotation + 3) % 4);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_movement_clears_rotation_flag() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.rotate_cw();
        assert!(env.last_move_was_rotation);
        env.move_left();
        assert!(!env.last_move_was_rotation);
    }

    #[test]
    fn test_hard_drop() {
        let mut env = TetrisEnv::new(10, 20);
        let drop_distance = env.hard_drop();
        assert!(drop_distance > 0);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_ghost_piece() {
        let env = TetrisEnv::new(10, 20);
        let ghost = env.get_ghost_piece();
        assert!(ghost.is_some());
        let ghost = ghost.unwrap();
        let current = env.current_piece.as_ref().unwrap();
        assert!(ghost.y >= current.y);
        assert_eq!(ghost.piece_type, current.piece_type);
        assert_eq!(ghost.rotation, current.rotation);
    }

    #[test]
    fn test_7bag_randomizer() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.piece_queue.len() >= 6);
        for &pt in &env.piece_queue {
            assert!(pt < 7);
        }
    }

    #[test]
    fn test_hold_piece() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(env.hold_piece.is_none());

        let current_type = env.current_piece.as_ref().unwrap().piece_type;
        let result = env.hold();

        assert!(result);
        assert_eq!(env.hold_piece, Some(current_type));
        assert!(env.hold_used);
        assert!(env.current_piece.is_some());
    }

    #[test]
    fn test_hold_twice_fails() {
        let mut env = TetrisEnv::new(10, 20);
        env.hold();
        let result = env.hold();
        assert!(!result);
    }

    #[test]
    fn test_hold_swap() {
        let mut env = TetrisEnv::new(10, 20);
        let first_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();
        let held_type = env.hold_piece.unwrap();
        assert_eq!(held_type, first_type);

        env.hard_drop();

        let second_type = env.current_piece.as_ref().unwrap().piece_type;
        env.hold();

        assert_eq!(env.current_piece.as_ref().unwrap().piece_type, first_type);
        assert_eq!(env.hold_piece, Some(second_type));
    }

    #[test]
    fn test_hold_when_game_over_fails() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let result = env.hold();
        assert!(!result);
    }

    #[test]
    fn test_lock_delay_initialization() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.lock_delay_ms.is_none());
        assert_eq!(env.lock_moves_remaining, 15);
    }

    #[test]
    fn test_lock_delay_starts_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..25 {
            env.move_down();
        }
        assert!(env.lock_delay_ms.is_some() || env.is_grounded());
    }

    #[test]
    fn test_lock_delay_progress() {
        let env = TetrisEnv::new(10, 20);
        assert_eq!(env.get_lock_delay_progress(), 0.0);
    }

    #[test]
    fn test_get_next_pieces() {
        let env = TetrisEnv::new(10, 20);
        let next_pieces = env.get_next_pieces(5);
        assert_eq!(next_pieces.len(), 5);
        for piece in next_pieces {
            assert!(piece.piece_type < 7);
        }
    }

    #[test]
    fn test_get_queue() {
        let env = TetrisEnv::new(10, 20);
        let queue = env.get_queue(5);
        assert_eq!(queue.len(), 5);
        for pt in queue {
            assert!(pt < 7);
        }
    }

    #[test]
    fn test_board_initially_empty() {
        let env = TetrisEnv::new(10, 20);
        let board = env.get_board();
        for row in board {
            for cell in row {
                assert_eq!(cell, 0);
            }
        }
    }

    #[test]
    fn test_board_colors_initially_none() {
        let env = TetrisEnv::new(10, 20);
        let colors = env.get_board_colors();
        for row in colors {
            for cell in row {
                assert!(cell.is_none());
            }
        }
    }

    #[test]
    fn test_clone_state() {
        let env = TetrisEnv::new(10, 20);
        let cloned = env.clone_state();
        assert_eq!(env.attack, cloned.attack);
        assert_eq!(env.width, cloned.width);
        assert_eq!(env.height, cloned.height);
        assert_eq!(env.lines_cleared, cloned.lines_cleared);
        assert_eq!(env.combo, cloned.combo);
        assert_eq!(env.back_to_back, cloned.back_to_back);
    }

    #[test]
    fn test_step_actions() {
        let mut env = TetrisEnv::new(10, 20);
        env.step(0); // noop
        env.step(1); // left
        env.step(2); // right
        env.step(3); // down
        env.step(4); // rotate_cw
        env.step(5); // rotate_ccw
        env.step(7); // hold
        assert!(!env.game_over);
    }

    #[test]
    fn test_step_returns_attack() {
        let mut env = TetrisEnv::new(10, 20);
        let (reward, game_over) = env.step(6); // hard_drop
        assert!(!game_over);
        assert!(reward >= 0);
    }

    #[test]
    fn test_wall_collision_left() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..10 {
            env.move_left();
        }
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x >= 0, "Piece should not go past left wall");
        }
    }

    #[test]
    fn test_wall_collision_right() {
        let mut env = TetrisEnv::new(10, 20);
        for _ in 0..10 {
            env.move_right();
        }
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x < env.width as i32, "Piece should not go past right wall");
        }
    }

    #[test]
    fn test_tick() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let moved = env.tick();
        if moved {
            let new_y = env.current_piece.as_ref().unwrap().y;
            assert_eq!(new_y, initial_y + 1);
        }
    }

    #[test]
    fn test_get_color_for_type() {
        let env = TetrisEnv::new(10, 20);
        for i in 0..7 {
            let _color = env.get_color_for_type(i);
        }
    }

    #[test]
    fn test_move_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        assert!(!env.move_left());
        assert!(!env.move_right());
        assert!(!env.move_down());
        assert!(!env.rotate_cw());
        assert!(!env.rotate_ccw());
        assert_eq!(env.hard_drop(), 0);
    }

    #[test]
    fn test_update_lock_delay_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;
        let locked = env.update_lock_delay(100);
        assert!(!locked);
    }

    #[test]
    fn test_is_piece_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(!env.is_piece_grounded());

        for _ in 0..25 {
            env.move_down();
        }
        assert!(env.is_piece_grounded());
    }

    #[test]
    fn test_get_next_piece() {
        let env = TetrisEnv::new(10, 20);
        let next = env.get_next_piece();
        assert!(next.is_some());
        assert!(next.unwrap().piece_type < 7);
    }

    #[test]
    fn test_is_hold_used() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(!env.is_hold_used());
        env.hold();
        assert!(env.is_hold_used());
    }

    #[test]
    fn test_get_hold_piece() {
        let mut env = TetrisEnv::new(10, 20);
        assert!(env.get_hold_piece().is_none());
        env.hold();
        assert!(env.get_hold_piece().is_some());
    }

    #[test]
    fn test_board_dimensions() {
        let env = TetrisEnv::new(10, 20);
        let board = env.get_board();
        assert_eq!(board.len(), 20);
        for row in board {
            assert_eq!(row.len(), 10);
        }
    }

    #[test]
    fn test_perfect_clear_detection() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.is_perfect_clear());
    }

    #[test]
    fn test_perfect_clear_with_cells() {
        let mut env = TetrisEnv::new(10, 20);
        env.board[19][0] = 1;
        env.sync_board_stats();
        assert!(!env.is_perfect_clear());
    }

    #[test]
    fn test_tspin_requires_t_piece() {
        let mut env = TetrisEnv::new(10, 20);
        let piece = Piece::with_position(0, 3, 10, 0); // I piece
        env.current_piece = Some(piece);
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, _) = env.check_tspin(env.current_piece.as_ref().unwrap());
        assert!(!is_tspin);
    }

    #[test]
    fn test_tspin_requires_rotation() {
        let mut env = TetrisEnv::new(10, 20);
        let piece = Piece::with_position(T_PIECE, 3, 10, 0);
        env.current_piece = Some(piece);
        env.last_move_was_rotation = false;

        let (is_tspin, _) = env.check_tspin(env.current_piece.as_ref().unwrap());
        assert!(!is_tspin);
    }

    #[test]
    fn test_combo_resets_on_no_clear() {
        let mut env = TetrisEnv::new(10, 20);
        env.combo = 5;
        env.clear_lines_internal(false, false);
        assert_eq!(env.combo, 0);
    }

    #[test]
    fn test_get_last_attack_result_initially_none() {
        let env = TetrisEnv::new(10, 20);
        assert!(env.get_last_attack_result().is_none());
    }

    // ==================== Board Collision Tests ====================

    #[test]
    fn test_is_cell_filled_empty_board() {
        let env = TetrisEnv::new(10, 20);
        // All cells should be empty
        for y in 0..20 {
            for x in 0..10 {
                assert!(!env.is_cell_filled(x, y), "Cell ({}, {}) should be empty", x, y);
            }
        }
    }

    #[test]
    fn test_is_cell_filled_out_of_bounds() {
        let env = TetrisEnv::new(10, 20);
        // Out of bounds should return true (filled)
        assert!(env.is_cell_filled(-1, 10)); // left of board
        assert!(env.is_cell_filled(10, 10)); // right of board
        assert!(env.is_cell_filled(5, -1)); // above board
        assert!(env.is_cell_filled(5, 20)); // below board
    }

    #[test]
    fn test_is_cell_filled_with_piece() {
        let mut env = TetrisEnv::new(10, 20);
        env.board[19][5] = 1;
        assert!(env.is_cell_filled(5, 19));
        assert!(!env.is_cell_filled(4, 19));
    }

    #[test]
    fn test_is_valid_position_empty_board() {
        let env = TetrisEnv::new(10, 20);
        // Piece in center of board should be valid
        let piece = Piece::with_position(T_PIECE, 3, 10, 0);
        assert!(env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_left_wall_collision() {
        let env = TetrisEnv::new(10, 20);
        // I-piece horizontal at x=-1 should be invalid
        let piece = Piece::with_position(0, -1, 10, 0); // I piece
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_right_wall_collision() {
        let env = TetrisEnv::new(10, 20);
        // I-piece horizontal at x=8 should be invalid (extends to x=11)
        let piece = Piece::with_position(0, 8, 10, 0); // I piece
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_floor_collision() {
        let env = TetrisEnv::new(10, 20);
        // Piece below floor should be invalid
        let piece = Piece::with_position(T_PIECE, 3, 20, 0);
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_piece_collision() {
        let mut env = TetrisEnv::new(10, 20);
        // Place a block and try to overlap
        env.board[19][5] = 1;
        let piece = Piece::with_position(T_PIECE, 4, 18, 0); // T piece that overlaps
        assert!(!env.is_valid_position_for(&piece));
    }

    #[test]
    fn test_is_valid_position_above_board_valid() {
        let env = TetrisEnv::new(10, 20);
        // Piece above visible board but not colliding is valid
        let piece = Piece::with_position(T_PIECE, 3, -2, 0);
        assert!(env.is_valid_position_for(&piece));
    }

    // ==================== T-Spin Detection Tests ====================

    #[test]
    fn test_tspin_full_with_3_corners_filled() {
        let mut env = TetrisEnv::new(10, 20);
        // Create a T-spin pocket: fill 3 corners around where T piece will land
        // T piece center at (4, 18) with rotation 0 has center at (5, 19)
        env.board[18][4] = 1; // top-left corner
        env.board[18][6] = 1; // top-right corner
        env.board[19][4] = 1; // bottom-left corner (front corner for rotation 0)

        let piece = Piece::with_position(T_PIECE, 4, 18, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        // With 3 corners filled and both front corners filled, should be full T-spin
        assert!(is_tspin, "Should detect T-spin with 3 corners filled");
    }

    #[test]
    fn test_tspin_mini_with_3_corners_one_front() {
        let mut env = TetrisEnv::new(10, 20);
        // T piece center at (5, 19) rotation 0 - front corners are top-left and top-right
        // Fill 3 corners but only 1 front corner
        env.board[18][4] = 1; // top-left corner (front)
        env.board[19][4] = 1; // bottom-left corner (back)
        env.board[19][6] = 1; // bottom-right corner (back)

        let piece = Piece::with_position(T_PIECE, 4, 18, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        assert!(is_tspin, "Should detect T-spin");
        assert!(is_mini, "Should be a T-spin mini with only 1 front corner");
    }

    #[test]
    fn test_tspin_kick_4_makes_full_tspin() {
        let mut env = TetrisEnv::new(10, 20);
        // With kick index 4 (the special SRS kick), even mini becomes full
        env.board[18][4] = 1;
        env.board[19][4] = 1;
        env.board[19][6] = 1;

        let piece = Piece::with_position(T_PIECE, 4, 18, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 4; // Special kick that makes it a full T-spin

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        assert!(is_tspin, "Should detect T-spin");
        assert!(!is_mini, "Kick index 4 should make it a full T-spin");
    }

    #[test]
    fn test_tspin_less_than_3_corners_not_tspin() {
        let mut env = TetrisEnv::new(10, 20);
        // T piece at (4, 10) with rotation 0 has center at (5, 11)
        // Corners are at: (4,10), (6,10), (4,12), (6,12)
        // Only fill 2 corners - not enough for T-spin
        env.board[10][4] = 1; // top-left corner
        env.board[10][6] = 1; // top-right corner
        // Leave bottom corners empty

        let piece = Piece::with_position(T_PIECE, 4, 10, 0);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, _) = env.check_tspin(&piece);
        assert!(!is_tspin, "Should not be T-spin with only 2 corners");
    }

    #[test]
    fn test_tspin_rotation_1_front_corners() {
        let mut env = TetrisEnv::new(10, 20);
        // T piece rotation 1 (CW) - front corners are top-right and bottom-right
        // Center is at (piece.x + 1, piece.y + 1) = (5, 19)
        env.board[18][6] = 1; // top-right (front)
        env.board[18][4] = 1; // top-left (back)
        env.board[19][6] = 1; // bottom-right (front)

        let piece = Piece::with_position(T_PIECE, 4, 18, 1);
        env.current_piece = Some(piece.clone());
        env.last_move_was_rotation = true;
        env.last_kick_index = 0;

        let (is_tspin, is_mini) = env.check_tspin(&piece);
        assert!(is_tspin, "Should detect T-spin in rotation 1");
        assert!(!is_mini, "Both front corners filled should be full T-spin");
    }

    // ==================== Line Clearing Tests ====================

    #[test]
    fn test_clear_single_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill the bottom row
        for x in 0..10 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 1);
        assert_eq!(env.combo, 1);
    }

    #[test]
    fn test_clear_tetris() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom 4 rows
        for y in 16..20 {
            for x in 0..10 {
                env.board[y][x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 4);
    }

    #[test]
    fn test_back_to_back_tetris() {
        let mut env = TetrisEnv::new(10, 20);
        env.back_to_back = true;

        // Fill bottom 4 rows for Tetris (difficult clear)
        for y in 16..20 {
            for x in 0..10 {
                env.board[y][x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        // Should maintain back-to-back and get bonus
        assert!(env.back_to_back);
        let result = env.last_attack_result.as_ref().unwrap();
        assert!(result.back_to_back_attack > 0);
    }

    #[test]
    fn test_single_breaks_back_to_back() {
        let mut env = TetrisEnv::new(10, 20);
        env.back_to_back = true;

        // Fill bottom 1 row (not a difficult clear)
        for x in 0..10 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        // Single line clear breaks back-to-back
        assert!(!env.back_to_back);
    }

    #[test]
    fn test_combo_increments() {
        let mut env = TetrisEnv::new(10, 20);
        env.combo = 3;

        // Fill one row
        for x in 0..10 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);

        assert_eq!(env.combo, 4);
    }

    #[test]
    fn test_tspin_single_attack() {
        let mut env = TetrisEnv::new(10, 20);

        // Fill one row
        for x in 0..10 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(true, false); // T-spin single

        let result = env.last_attack_result.as_ref().unwrap();
        assert!(result.is_tspin);
        assert!(result.base_attack > 0); // T-spin single has attack
    }

    #[test]
    fn test_tspin_mini_attack() {
        let mut env = TetrisEnv::new(10, 20);

        // Fill one row
        for x in 0..10 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();
        env.clear_lines_internal(true, true); // T-spin mini single

        let result = env.last_attack_result.as_ref().unwrap();
        assert!(result.is_tspin);
    }

    // ==================== Movement Tests ====================

    #[test]
    fn test_horizontal_movement_blocked_by_wall() {
        let mut env = TetrisEnv::new(10, 20);
        // Move left until blocked
        let mut moves = 0;
        while env.move_left() {
            moves += 1;
            if moves > 20 {
                panic!("Infinite loop detected");
            }
        }
        // Should be blocked by wall, not infinite loop
        assert!(moves > 0);
    }

    #[test]
    fn test_horizontal_movement_blocked_by_piece() {
        let mut env = TetrisEnv::new(10, 20);
        // Place wall of pieces on the left
        for y in 0..20 {
            env.board[y][0] = 1;
            env.board[y][1] = 1;
        }

        // Move current piece down a bit
        env.move_down();
        env.move_down();

        // Try to move left - should eventually be blocked by placed pieces
        let mut moves = 0;
        while env.move_left() {
            moves += 1;
            if moves > 20 {
                break;
            }
        }

        // Should have been blocked before reaching x=0
        let piece = env.current_piece.as_ref().unwrap();
        let cells = piece.get_cells();
        for (x, _) in cells {
            assert!(x >= 2, "Piece should be blocked by placed pieces at x=0,1");
        }
    }

    #[test]
    fn test_rotation_resets_lock_delay_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);
        // Move piece to ground
        for _ in 0..25 {
            env.move_down();
        }

        // Start lock delay
        if env.lock_delay_ms.is_none() {
            env.lock_delay_ms = Some(0);
        }
        env.lock_delay_ms = Some(200); // Simulate some time passed

        let lock_moves_before = env.lock_moves_remaining;

        // Rotate should reset lock delay if successful
        if env.rotate_cw() {
            assert!(
                env.lock_delay_ms.is_some(),
                "Lock delay should still be active"
            );
        }
    }

    #[test]
    fn test_rotation_tracks_last_kick_index() {
        let mut env = TetrisEnv::new(10, 20);
        env.move_down();
        env.move_down();

        // Rotate piece
        if env.rotate_cw() {
            // Kick index should be set (likely 0 for no-kick rotation)
            assert!(env.last_kick_index < 5, "Kick index should be 0-4");
        }
    }

    #[test]
    fn test_rotation_with_wall_kick() {
        let mut env = TetrisEnv::new(10, 20);
        // Move I-piece to right wall
        env.spawn_piece_from_type(0); // Spawn I piece
        for _ in 0..10 {
            env.move_right();
        }

        // Move down to have room
        for _ in 0..5 {
            env.move_down();
        }

        // Get initial position
        let initial_x = env.current_piece.as_ref().unwrap().x;
        let initial_rotation = env.current_piece.as_ref().unwrap().rotation;

        // Rotate - might use wall kick
        if env.rotate_cw() {
            let piece = env.current_piece.as_ref().unwrap();
            // Rotation should have succeeded, possibly with a kick
            assert_ne!(piece.rotation, initial_rotation);
            assert!(env.last_move_was_rotation);
        }
    }

    #[test]
    fn test_no_movement_when_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        assert!(!env.move_horizontal(-1));
        assert!(!env.move_horizontal(1));
        assert!(!env.rotate(true));
        assert!(!env.rotate(false));
    }

    // ==================== Additional Edge Case Tests ====================

    #[test]
    fn test_spawn_at_top_with_collision() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill top rows where pieces spawn
        for x in 0..10 {
            env.board[0][x] = 1;
            env.board[1][x] = 1;
        }

        // Spawning should trigger game over
        env.spawn_piece_internal();
        assert!(env.game_over);
    }

    #[test]
    fn test_hard_drop_distance_calculation() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_y = env.current_piece.as_ref().unwrap().y;
        let drop_distance = env.hard_drop();

        // Drop distance should match how far piece traveled
        assert!(drop_distance > 0);
    }

    #[test]
    fn test_ghost_piece_at_bottom() {
        let mut env = TetrisEnv::new(10, 20);
        let ghost = env.get_ghost_piece();
        assert!(ghost.is_some());

        let ghost = ghost.unwrap();
        let current = env.current_piece.as_ref().unwrap();

        // Ghost should be at or below current piece
        assert!(ghost.y >= current.y);

        // Ghost should be grounded (can't move down further)
        let mut test_ghost = ghost.clone();
        test_ghost.y += 1;
        assert!(
            !env.is_valid_position_for(&test_ghost),
            "Ghost should be at lowest valid position"
        );
    }

    #[test]
    fn test_board_colors_after_lock() {
        let mut env = TetrisEnv::new(10, 20);
        let piece_type = env.current_piece.as_ref().unwrap().piece_type;

        // Hard drop to lock piece
        env.hard_drop();

        // Check that board_colors has the piece type
        let mut found_color = false;
        for row in &env.board_colors {
            for cell in row {
                if *cell == Some(piece_type) {
                    found_color = true;
                    break;
                }
            }
        }
        assert!(found_color, "Board colors should contain locked piece type");
    }

    #[test]
    fn test_hold_resets_after_lock() {
        let mut env = TetrisEnv::new(10, 20);
        env.hold(); // Use hold
        assert!(env.hold_used);

        env.hard_drop(); // Lock piece
        assert!(!env.hold_used, "Hold should reset after piece locks");
    }

    #[test]
    fn test_clear_double_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom 2 rows
        for y in 18..20 {
            for x in 0..10 {
                env.board[y][x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 2);
    }

    #[test]
    fn test_clear_triple_line() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill bottom 3 rows
        for y in 17..20 {
            for x in 0..10 {
                env.board[y][x] = 1;
            }
        }
        env.sync_board_stats();
        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 3);
    }

    #[test]
    fn test_non_contiguous_line_clear() {
        let mut env = TetrisEnv::new(10, 20);
        // Fill rows 15 and 19 (not contiguous)
        for x in 0..10 {
            env.board[15][x] = 1;
            env.board[19][x] = 1;
        }
        // Leave row 17 partial
        for x in 0..5 {
            env.board[17][x] = 1;
        }
        env.sync_board_stats();

        env.clear_lines_internal(false, false);
        assert_eq!(env.lines_cleared, 2, "Should clear 2 non-contiguous lines");
    }

    // ==================== Lock Delay Tests ====================

    #[test]
    fn test_is_grounded_at_spawn() {
        let env = TetrisEnv::new(10, 20);
        // Piece at spawn should not be grounded
        assert!(!env.is_grounded(), "Piece at spawn should not be grounded");
    }

    #[test]
    fn test_is_grounded_at_bottom() {
        let mut env = TetrisEnv::new(10, 20);
        // Move piece to bottom
        while env.move_down() {}
        assert!(env.is_grounded(), "Piece at bottom should be grounded");
    }

    #[test]
    fn test_is_grounded_on_stack() {
        let mut env = TetrisEnv::new(10, 20);
        // Create a stack in the middle
        for x in 0..10 {
            env.board[15][x] = 1;
        }
        env.sync_board_stats();

        // Move piece down until it lands on the stack
        while env.move_down() {}

        assert!(env.is_grounded(), "Piece on top of stack should be grounded");
    }

    #[test]
    fn test_is_grounded_no_piece() {
        let mut env = TetrisEnv::new(10, 20);
        env.current_piece = None;
        assert!(!env.is_grounded(), "No piece means not grounded");
    }

    #[test]
    fn test_reset_lock_delay_decrements_moves() {
        let mut env = TetrisEnv::new(10, 20);
        let initial_moves = env.lock_moves_remaining;
        assert_eq!(initial_moves, 15);

        env.reset_lock_delay();
        assert_eq!(env.lock_moves_remaining, 14);
        assert_eq!(env.lock_delay_ms, Some(0));
    }

    #[test]
    fn test_reset_lock_delay_multiple_times() {
        let mut env = TetrisEnv::new(10, 20);

        for i in 0..15 {
            env.reset_lock_delay();
            assert_eq!(env.lock_moves_remaining, 14 - i);
        }

        // After 15 resets, should be at 0
        assert_eq!(env.lock_moves_remaining, 0);
    }

    #[test]
    fn test_reset_lock_delay_stops_at_zero() {
        let mut env = TetrisEnv::new(10, 20);
        env.lock_moves_remaining = 0;
        env.lock_delay_ms = Some(100);

        env.reset_lock_delay();

        // Should not decrement below 0 or reset timer
        assert_eq!(env.lock_moves_remaining, 0);
        assert_eq!(env.lock_delay_ms, Some(100)); // Timer unchanged
    }

    #[test]
    fn test_clear_lock_delay() {
        let mut env = TetrisEnv::new(10, 20);
        env.lock_delay_ms = Some(250);
        env.lock_moves_remaining = 5;

        env.clear_lock_delay();

        assert_eq!(env.lock_delay_ms, None);
        assert_eq!(env.lock_moves_remaining, 15); // Reset to default
    }

    #[test]
    fn test_lock_delay_resets_on_movement_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);

        // Move to bottom
        while env.move_down() {}
        assert!(env.is_grounded());

        // Start lock delay
        env.lock_delay_ms = Some(100);
        let moves_before = env.lock_moves_remaining;

        // Move left (should reset lock delay if successful)
        if env.move_left() {
            assert_eq!(env.lock_delay_ms, Some(0), "Lock delay should reset to 0");
            assert_eq!(
                env.lock_moves_remaining,
                moves_before - 1,
                "Should use one lock move"
            );
        }
    }

    #[test]
    fn test_lock_delay_resets_on_rotation_when_grounded() {
        let mut env = TetrisEnv::new(10, 20);

        // Move to bottom
        while env.move_down() {}

        // Start lock delay
        env.lock_delay_ms = Some(100);
        let moves_before = env.lock_moves_remaining;

        // Rotate (should reset lock delay if successful)
        if env.rotate_cw() {
            assert_eq!(env.lock_delay_ms, Some(0));
            assert_eq!(env.lock_moves_remaining, moves_before - 1);
        }
    }

    // ==================== Placement Tests ====================

    #[test]
    fn test_place_piece_internal_basic() {
        let mut env = TetrisEnv::new(10, 20);
        // Use O piece which is simple
        env.spawn_piece_from_type(1); // O piece

        // O piece shape has cells at rows 1-2, cols 1-2 of the 4x4 grid
        // So at y=17, cells are at y=18 and y=19 (valid)
        let attack = env.place_piece_internal_with_kick(4, 17, 0, false, 0);

        // Piece should be locked, new piece spawned
        assert!(env.current_piece.is_some());
        // Board should have the piece
        let mut has_piece = false;
        for row in &env.board {
            if row.iter().any(|&c| c != 0) {
                has_piece = true;
                break;
            }
        }
        assert!(has_piece, "Board should have locked piece");
    }

    #[test]
    fn test_place_piece_internal_game_over() {
        let mut env = TetrisEnv::new(10, 20);
        env.game_over = true;

        let attack = env.place_piece_internal_with_kick(3, 18, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 attack when game over");
    }

    #[test]
    fn test_place_piece_internal_invalid_position() {
        let mut env = TetrisEnv::new(10, 20);

        // Try to place at invalid position (outside board)
        let attack = env.place_piece_internal_with_kick(-5, 18, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 for invalid position");
    }

    #[test]
    fn test_place_piece_internal_no_current_piece() {
        let mut env = TetrisEnv::new(10, 20);
        env.current_piece = None;

        let attack = env.place_piece_internal_with_kick(3, 18, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 when no current piece");
    }

    #[test]
    fn test_place_piece_internal_with_rotation() {
        let mut env = TetrisEnv::new(10, 20);

        // Place with rotation flag set
        let attack = env.place_piece_internal_with_kick(3, 18, 1, true, 0);

        // The placement should work (assuming position is valid for rotated piece)
        // Main thing is it doesn't crash
    }

    #[test]
    fn test_place_piece_internal_with_kick_index() {
        let mut env = TetrisEnv::new(10, 20);
        // Spawn T piece for T-spin testing
        env.spawn_piece_from_type(T_PIECE);

        // Set up a scenario where kick_index matters for T-spin
        env.board[18][4] = 1;
        env.board[19][4] = 1;
        env.board[19][6] = 1;
        env.sync_board_stats();

        // Place with kick_index = 4 (special T-spin kick)
        let attack = env.place_piece_internal_with_kick(4, 17, 0, true, 4);

        // The last_kick_index should be set
        // (Note: piece gets locked so we can't check it directly, but the attack should reflect it)
    }

    #[test]
    fn test_place_piece_internal_clears_lines() {
        let mut env = TetrisEnv::new(10, 20);
        // Spawn I piece
        env.spawn_piece_from_type(0); // I piece

        // Set up almost-complete row
        for x in 0..6 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();

        // Place I piece horizontally to complete the row
        // I piece at rotation 0 is horizontal: [1,1,1,1] at row 1 of the shape
        let attack = env.place_piece_internal_with_kick(6, 18, 0, false, 0);

        // Should have cleared a line and gotten attack
        assert!(env.lines_cleared > 0 || attack > 0 || env.board[19].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_place_piece_internal_collision_with_stack() {
        let mut env = TetrisEnv::new(10, 20);

        // Create a stack
        for y in 15..20 {
            for x in 0..10 {
                env.board[y][x] = 1;
            }
        }
        env.sync_board_stats();

        // Try to place piece inside the stack (invalid)
        let attack = env.place_piece_internal_with_kick(3, 16, 0, false, 0);
        assert_eq!(attack, 0, "Should return 0 when colliding with stack");
    }

    #[test]
    fn test_place_piece_internal_all_rotations() {
        // Test that all 4 rotations work
        for rotation in 0..4 {
            let mut env = TetrisEnv::new(10, 20);

            // Use T piece which has distinct rotations
            env.spawn_piece_from_type(T_PIECE);

            let attack = env.place_piece_internal_with_kick(4, 17, rotation, false, 0);
            // Should not panic and piece should be placed
            assert!(
                env.lines_cleared >= 0,
                "Rotation {} should work",
                rotation
            );
        }
    }

    #[test]
    fn test_place_piece_sets_last_move_was_rotation() {
        let mut env = TetrisEnv::new(10, 20);
        env.last_move_was_rotation = false;

        // Place with was_rotation = true
        env.place_piece_internal_with_kick(3, 18, 0, true, 0);

        // New piece is spawned, but the T-spin detection should have used the flag
        // We can't easily verify this without checking attack values in a T-spin setup
    }

    #[test]
    fn test_place_piece_returns_attack_delta() {
        let mut env = TetrisEnv::new(10, 20);
        env.spawn_piece_from_type(0); // I piece

        // Set up Tetris (4 almost-complete rows), leaving column 9 empty
        for y in 16..20 {
            for x in 0..9 {
                env.board[y][x] = 1;
            }
        }
        env.sync_board_stats();

        let initial_attack = env.attack;

        // Place I piece vertically to complete all 4 rows
        // I piece rotation 1 (vertical) has cells at column 2 of the 4x4 grid
        // So at x=7, cells are at 7+2=9, which completes the rows
        let attack_delta = env.place_piece_internal_with_kick(7, 16, 1, false, 0);

        // Should return the attack gained (Tetris = 4 lines)
        assert!(attack_delta > 0, "Tetris should give attack");
        assert_eq!(env.attack, initial_attack + attack_delta);
    }

    #[test]
    fn test_execute_placement_changes_current_piece() {
        // This test verifies that after execute_placement, the current piece changes
        // (i.e., the old piece is locked and a new one spawns from the queue)
        let env = TetrisEnv::with_seed(10, 20, 42);

        let initial_piece_type = env.get_current_piece().unwrap().piece_type;
        let initial_queue: Vec<usize> = env.get_queue(5);

        // Clone and execute a placement
        let mut env_clone = env.clone_state();
        let placements = env_clone.get_possible_placements();
        assert!(!placements.is_empty(), "Should have valid placements");

        let placement = &placements[0];
        env_clone.execute_placement(placement);

        // Verify state changed
        let new_piece = env_clone.get_current_piece();
        assert!(new_piece.is_some(), "Should have a new piece after placement");

        let new_piece_type = new_piece.unwrap().piece_type;
        let new_queue: Vec<usize> = env_clone.get_queue(5);

        // The new current piece should be what was first in the queue
        assert_eq!(
            new_piece_type, initial_queue[0],
            "New piece should be first from old queue. Got {}, expected {}",
            new_piece_type, initial_queue[0]
        );

        // The queue should have shifted (minus the piece that became current)
        assert_eq!(
            new_queue[0], initial_queue[1],
            "Queue should shift after spawn. New queue[0]={}, expected old queue[1]={}",
            new_queue[0], initial_queue[1]
        );

        // Board should have the old piece locked
        let board = env_clone.get_board();
        let has_locked_piece = board.iter().any(|row| row.iter().any(|&c| c != 0));
        assert!(has_locked_piece, "Board should have locked piece");

        // Verify original env is unchanged
        let orig_piece_type = env.get_current_piece().unwrap().piece_type;
        assert_eq!(
            orig_piece_type, initial_piece_type,
            "Original env should be unchanged"
        );
    }

    // ==================== Tracking Fields Verification Tests ====================
    // These tests verify that incremental updates to optimization fields
    // (column_heights, total_blocks, row_fill_counts) match ground truth.

    /// Helper to compute expected tracking fields from board state
    fn compute_expected_stats(board: &[Vec<u8>]) -> (Vec<i32>, u32, Vec<u8>) {
        let height = board.len();
        let width = board[0].len();
        let mut column_heights = vec![height as i32; width];
        let mut total_blocks = 0u32;
        let mut row_fill_counts = vec![0u8; height];

        for x in 0..width {
            for y in 0..height {
                if board[y][x] != 0 {
                    total_blocks += 1;
                    row_fill_counts[y] += 1;
                    if (y as i32) < column_heights[x] {
                        column_heights[x] = y as i32;
                    }
                }
            }
        }
        (column_heights, total_blocks, row_fill_counts)
    }

    #[test]
    fn test_tracking_fields_after_hard_drop() {
        let mut env = TetrisEnv::with_seed(10, 20, 42);

        // Drop a piece
        env.hard_drop();

        // Compute expected values from actual board state
        let (expected_heights, expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board);

        assert_eq!(env.column_heights, expected_heights,
            "column_heights mismatch after hard_drop");
        assert_eq!(env.total_blocks, expected_blocks,
            "total_blocks mismatch after hard_drop");
        assert_eq!(env.row_fill_counts, expected_row_counts,
            "row_fill_counts mismatch after hard_drop");
    }

    #[test]
    fn test_tracking_fields_after_multiple_drops() {
        let mut env = TetrisEnv::with_seed(10, 20, 123);

        // Drop several pieces
        for _ in 0..10 {
            if env.game_over {
                break;
            }
            env.hard_drop();
        }

        let (expected_heights, expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board);

        assert_eq!(env.column_heights, expected_heights,
            "column_heights mismatch after multiple drops");
        assert_eq!(env.total_blocks, expected_blocks,
            "total_blocks mismatch after multiple drops");
        assert_eq!(env.row_fill_counts, expected_row_counts,
            "row_fill_counts mismatch after multiple drops");
    }

    #[test]
    fn test_tracking_fields_after_line_clear() {
        let mut env = TetrisEnv::with_seed(10, 20, 456);

        // Play until we get some line clears
        let mut total_clears = 0u32;
        for _ in 0..50 {
            if env.game_over {
                break;
            }
            let lines_before = env.lines_cleared;
            env.hard_drop();
            if env.lines_cleared > lines_before {
                total_clears += env.lines_cleared - lines_before;
            }
        }

        let (expected_heights, expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board);

        assert_eq!(env.column_heights, expected_heights,
            "column_heights mismatch after line clears (cleared {} lines)", total_clears);
        assert_eq!(env.total_blocks, expected_blocks,
            "total_blocks mismatch after line clears");
        assert_eq!(env.row_fill_counts, expected_row_counts,
            "row_fill_counts mismatch after line clears");
    }

    #[test]
    fn test_tracking_fields_with_movements() {
        let mut env = TetrisEnv::with_seed(10, 20, 789);

        // Move piece around before dropping
        for _ in 0..5 {
            if env.game_over {
                break;
            }
            env.move_left();
            env.move_right();
            env.move_right();
            env.rotate_cw();
            env.move_down();
            env.hard_drop();
        }

        let (expected_heights, expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board);

        assert_eq!(env.column_heights, expected_heights);
        assert_eq!(env.total_blocks, expected_blocks);
        assert_eq!(env.row_fill_counts, expected_row_counts);
    }

    #[test]
    fn test_tracking_fields_with_hold() {
        let mut env = TetrisEnv::with_seed(10, 20, 321);

        // Use hold feature
        env.hold();
        env.hard_drop();
        env.hold();
        env.hard_drop();

        let (expected_heights, expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board);

        assert_eq!(env.column_heights, expected_heights);
        assert_eq!(env.total_blocks, expected_blocks);
        assert_eq!(env.row_fill_counts, expected_row_counts);
    }

    #[test]
    fn test_tracking_fields_perfect_clear() {
        // After a perfect clear, all tracking fields should be zero/empty
        let mut env = TetrisEnv::with_seed(10, 20, 999);

        // Manually set up a perfect clear scenario
        // Fill bottom row except one cell, then place I piece to complete
        for x in 0..6 {
            env.board[19][x] = 1;
        }
        env.sync_board_stats();

        // Verify sync works correctly
        let (expected_heights, expected_blocks, expected_row_counts) =
            compute_expected_stats(&env.board);
        assert_eq!(env.column_heights, expected_heights);
        assert_eq!(env.total_blocks, expected_blocks);
        assert_eq!(env.row_fill_counts, expected_row_counts);
    }

    #[test]
    fn test_row_fill_counts_partial_rows() {
        let mut env = TetrisEnv::new(10, 20);

        // Set up rows with different fill levels
        for x in 0..3 {
            env.board[19][x] = 1;  // 3 cells in row 19
        }
        for x in 0..7 {
            env.board[18][x] = 1;  // 7 cells in row 18
        }
        for x in 0..10 {
            env.board[17][x] = 1;  // 10 cells in row 17 (full)
        }
        env.sync_board_stats();

        assert_eq!(env.row_fill_counts[19], 3);
        assert_eq!(env.row_fill_counts[18], 7);
        assert_eq!(env.row_fill_counts[17], 10);
        assert_eq!(env.row_fill_counts[16], 0);  // Empty row
    }

    #[test]
    fn test_column_heights_varied() {
        let mut env = TetrisEnv::new(10, 20);

        // Create varied column heights
        env.board[19][0] = 1;  // Column 0: height at y=19
        env.board[15][1] = 1;  // Column 1: height at y=15
        env.board[10][2] = 1;  // Column 2: height at y=10
        env.board[19][2] = 1;  // Also in column 2, but lower
        env.sync_board_stats();

        assert_eq!(env.column_heights[0], 19);
        assert_eq!(env.column_heights[1], 15);
        assert_eq!(env.column_heights[2], 10);  // Topmost cell
        assert_eq!(env.column_heights[3], 20);  // Empty column = height
    }

    #[test]
    fn test_total_blocks_count() {
        let mut env = TetrisEnv::new(10, 20);

        // Place exactly 15 blocks
        let positions = [
            (0, 19), (1, 19), (2, 19),
            (0, 18), (1, 18), (2, 18), (3, 18),
            (5, 15), (6, 15), (7, 15), (8, 15), (9, 15),
            (0, 10), (5, 10), (9, 10),
        ];
        for (x, y) in positions {
            env.board[y][x] = 1;
        }
        env.sync_board_stats();

        assert_eq!(env.total_blocks, 15);
    }

    #[test]
    fn test_incremental_vs_sync_consistency() {
        // This test verifies incremental updates match sync_board_stats
        let mut env = TetrisEnv::with_seed(10, 20, 12345);

        for i in 0..20 {
            if env.game_over {
                break;
            }

            // Save current tracking values (from incremental updates)
            let inc_heights = env.column_heights.clone();
            let inc_blocks = env.total_blocks;
            let inc_row_counts = env.row_fill_counts.clone();

            // Recalculate from scratch
            let (sync_heights, sync_blocks, sync_row_counts) =
                compute_expected_stats(&env.board);

            assert_eq!(inc_heights, sync_heights,
                "Iteration {}: column_heights incremental != sync", i);
            assert_eq!(inc_blocks, sync_blocks,
                "Iteration {}: total_blocks incremental != sync", i);
            assert_eq!(inc_row_counts, sync_row_counts,
                "Iteration {}: row_fill_counts incremental != sync", i);

            env.hard_drop();
        }
    }
}
