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
}
