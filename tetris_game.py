#!/usr/bin/env python3
"""
Tetris game with Rust backend and Pygame visualization.
Implements proper SRS rotation and DAS/ARR for responsive controls.
"""

import pygame
import sys

try:
    from tetris_core import TetrisEnv, Piece
except ImportError:
    print("Error: tetris_core module not found.")
    print("Please build and install it first:")
    print("  cd tetris_core && maturin develop")
    sys.exit(1)

# Constants
CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
LEFT_SIDEBAR_WIDTH = 100  # For HOLD piece
RIGHT_SIDEBAR_WIDTH = 120  # For NEXT pieces and score

WINDOW_WIDTH = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE + RIGHT_SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (40, 40, 40)
GHOST_ALPHA = 80


class TetrisGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris (Rust Backend + SRS)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
        self.fall_time = 0
        self.paused = False

        # DAS (Delayed Auto Shift) and ARR (Auto Repeat Rate) settings
        # DAS: delay before auto-repeat starts (ms)
        # ARR: interval between auto-repeats once DAS triggers (ms)
        # Lower values = faster movement. ARR=0 means instant movement.
        self.das = 110  # TF DAS / NullpoMino style
        self.arr = 0    # TF ARR - instant movement

        # Soft drop settings
        self.soft_drop_das = 0    # No delay for soft drop auto-repeat (instant)
        self.soft_drop_arr = 25   # 25ms between soft drop repeats (~40 cells/sec)

        # Soft drop speed multiplier (how many times faster than normal gravity)
        self.soft_drop_factor = 20

        # Key state tracking for DAS/ARR
        # Structure: {key: {'pressed_time': int, 'das_charged': bool, 'last_move_time': int}}
        self.key_states = {}

        # Track which direction keys are held for prioritization
        self.left_held = False
        self.right_held = False
        self.down_held = False

    def get_fall_speed(self):
        """Get fall speed based on current level (in milliseconds)."""
        level = self.env.level
        # Standard Tetris gravity curve (approximately)
        if level <= 0:
            return 1000
        elif level == 1:
            return 1000
        elif level == 2:
            return 793
        elif level == 3:
            return 618
        elif level == 4:
            return 473
        elif level == 5:
            return 355
        elif level == 6:
            return 262
        elif level == 7:
            return 190
        elif level == 8:
            return 135
        elif level == 9:
            return 94
        elif level == 10:
            return 64
        elif level <= 13:
            return 43
        elif level <= 16:
            return 28
        elif level <= 19:
            return 18
        elif level <= 29:
            return 11
        else:
            return 5

    def handle_events(self):
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    continue

                if event.key == pygame.K_r:
                    self.env.reset()
                    self.fall_time = 0
                    continue

                if self.paused or self.env.game_over:
                    continue

                # Movement keys with DAS/ARR
                if event.key == pygame.K_LEFT:
                    self.left_held = True
                    self.env.move_left()
                    self.key_states[pygame.K_LEFT] = {
                        'pressed_time': current_time,
                        'das_charged': False,
                        'last_move_time': current_time
                    }

                elif event.key == pygame.K_RIGHT:
                    self.right_held = True
                    self.env.move_right()
                    self.key_states[pygame.K_RIGHT] = {
                        'pressed_time': current_time,
                        'das_charged': False,
                        'last_move_time': current_time
                    }

                elif event.key == pygame.K_DOWN:
                    self.down_held = True
                    self.env.move_down()
                    self.key_states[pygame.K_DOWN] = {
                        'pressed_time': current_time,
                        'das_charged': False,
                        'last_move_time': current_time
                    }

                # Rotation keys (no DAS needed)
                elif event.key == pygame.K_UP or event.key == pygame.K_s:
                    self.env.rotate_cw()

                elif event.key == pygame.K_a:
                    self.env.rotate_ccw()

                # Hard drop
                elif event.key == pygame.K_SPACE:
                    self.env.hard_drop()
                    self.fall_time = 0

                # Hold piece
                elif event.key == pygame.K_d:
                    self.env.hold()

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.left_held = False
                    if pygame.K_LEFT in self.key_states:
                        del self.key_states[pygame.K_LEFT]

                elif event.key == pygame.K_RIGHT:
                    self.right_held = False
                    if pygame.K_RIGHT in self.key_states:
                        del self.key_states[pygame.K_RIGHT]

                elif event.key == pygame.K_DOWN:
                    self.down_held = False
                    if pygame.K_DOWN in self.key_states:
                        del self.key_states[pygame.K_DOWN]

        # Handle DAS/ARR for held keys
        if not self.paused and not self.env.game_over:
            self._handle_das_arr(current_time)

        return True

    def _handle_das_arr(self, current_time):
        """Handle Delayed Auto Shift and Auto Repeat Rate for movement keys."""

        # Process horizontal movement (left/right)
        # If both are held, most recent press wins (or we can cancel - here we use most recent)
        for key in [pygame.K_LEFT, pygame.K_RIGHT]:
            if key not in self.key_states:
                continue

            state = self.key_states[key]
            time_held = current_time - state['pressed_time']
            time_since_move = current_time - state['last_move_time']

            # Check if DAS has charged
            if not state['das_charged']:
                if time_held >= self.das:
                    state['das_charged'] = True
                    # Immediate move when DAS charges
                    if key == pygame.K_LEFT:
                        self.env.move_left()
                    else:
                        self.env.move_right()
                    state['last_move_time'] = current_time
            else:
                # DAS is charged, apply ARR
                if self.arr == 0:
                    # ARR=0 means instant: move all the way
                    if key == pygame.K_LEFT:
                        while self.env.move_left():
                            pass
                    else:
                        while self.env.move_right():
                            pass
                    state['last_move_time'] = current_time
                elif time_since_move >= self.arr:
                    if key == pygame.K_LEFT:
                        self.env.move_left()
                    else:
                        self.env.move_right()
                    state['last_move_time'] = current_time

        # Process soft drop (down key) - uses separate, faster settings
        if pygame.K_DOWN in self.key_states:
            state = self.key_states[pygame.K_DOWN]
            time_held = current_time - state['pressed_time']
            time_since_move = current_time - state['last_move_time']

            if not state['das_charged']:
                if time_held >= self.soft_drop_das:
                    state['das_charged'] = True
                    self.env.move_down()
                    state['last_move_time'] = current_time
            else:
                if time_since_move >= self.soft_drop_arr:
                    self.env.move_down()
                    state['last_move_time'] = current_time

    def update(self, dt):
        if self.paused or self.env.game_over:
            return

        # Update lock delay timer
        self.env.update_lock_delay(dt)

        self.fall_time += dt
        fall_speed = self.get_fall_speed()

        # If down is held, use faster gravity (soft drop)
        if self.down_held:
            fall_speed = fall_speed // self.soft_drop_factor

        if self.fall_time >= fall_speed:
            self.env.tick()
            self.fall_time = 0

    def draw_board(self):
        board_x = LEFT_SIDEBAR_WIDTH  # Board starts after left sidebar

        # Draw board background
        board_rect = pygame.Rect(board_x, 0, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
        pygame.draw.rect(self.screen, BLACK, board_rect)

        # Draw grid lines
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                DARK_GRAY,
                (board_x + x * CELL_SIZE, 0),
                (board_x + x * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
            )
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                DARK_GRAY,
                (board_x, y * CELL_SIZE),
                (board_x + BOARD_WIDTH * CELL_SIZE, y * CELL_SIZE)
            )

        # Draw locked pieces
        board = self.env.get_board()
        board_colors = self.env.get_board_colors()

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if board[y][x] != 0:
                    color_idx = board_colors[y][x]
                    if color_idx is not None:
                        color = self.env.get_color_for_type(color_idx)
                    else:
                        color = GRAY

                    self.draw_cell(board_x + x * CELL_SIZE, y * CELL_SIZE, color)

        # Draw ghost piece
        ghost = self.env.get_ghost_piece()
        if ghost:
            color = ghost.get_color()
            ghost_surface = pygame.Surface((CELL_SIZE - 2, CELL_SIZE - 2), pygame.SRCALPHA)
            ghost_surface.fill((*color, GHOST_ALPHA))

            for (x, y) in ghost.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    self.screen.blit(ghost_surface, (board_x + x * CELL_SIZE + 1, y * CELL_SIZE + 1))

        # Draw current piece
        piece = self.env.get_current_piece()
        if piece:
            color = piece.get_color()
            for (x, y) in piece.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    self.draw_cell(board_x + x * CELL_SIZE, y * CELL_SIZE, color)

    def draw_cell(self, px, py, color):
        """Draw a single cell with 3D effect at pixel coordinates."""
        rect = pygame.Rect(
            px + 1,
            py + 1,
            CELL_SIZE - 2,
            CELL_SIZE - 2
        )

        # Main color
        pygame.draw.rect(self.screen, color, rect)

        # Highlight (top-left)
        highlight = tuple(min(255, c + 50) for c in color)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft, 2)

        # Shadow (bottom-right)
        shadow = tuple(max(0, c - 50) for c in color)
        pygame.draw.line(self.screen, shadow, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow, rect.topright, rect.bottomright, 2)

    def draw_sidebar(self):
        # Draw left sidebar (HOLD)
        left_rect = pygame.Rect(0, 0, LEFT_SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, left_rect)

        # HOLD label
        hold_label = self.font.render("HOLD", True, WHITE)
        self.screen.blit(hold_label, (15, 15))

        # HOLD box
        hold_box_rect = pygame.Rect(10, 50, 80, 70)
        pygame.draw.rect(self.screen, BLACK, hold_box_rect)
        pygame.draw.rect(self.screen, GRAY, hold_box_rect, 1)

        hold_piece = self.env.get_hold_piece()
        if hold_piece:
            hold_used = self.env.is_hold_used()
            color = hold_piece.get_color()
            if hold_used:
                color = tuple(c // 2 for c in color)
            shape = hold_piece.get_shape()
            for dy, row in enumerate(shape):
                for dx, cell in enumerate(row):
                    if cell == 1:
                        rect = pygame.Rect(20 + dx * 16, 60 + dy * 16, 14, 14)
                        pygame.draw.rect(self.screen, color, rect)

        # Draw right sidebar (NEXT + stats)
        right_x = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE
        right_rect = pygame.Rect(right_x, 0, RIGHT_SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, right_rect)

        # NEXT label
        next_label = self.font.render("NEXT", True, WHITE)
        self.screen.blit(next_label, (right_x + 15, 15))

        # Draw next 5 pieces
        next_pieces = self.env.get_next_pieces(5)
        for i, piece in enumerate(next_pieces):
            box_y = 50 + i * 55
            box_rect = pygame.Rect(right_x + 10, box_y, 80, 50)
            pygame.draw.rect(self.screen, BLACK, box_rect)
            pygame.draw.rect(self.screen, GRAY, box_rect, 1)

            color = piece.get_color()
            shape = piece.get_shape()
            for dy, row in enumerate(shape):
                for dx, cell in enumerate(row):
                    if cell == 1:
                        rect = pygame.Rect(
                            right_x + 20 + dx * 14,
                            box_y + 8 + dy * 14,
                            12, 12
                        )
                        pygame.draw.rect(self.screen, color, rect)

        # Stats at bottom
        stats_y = 340
        score_label = self.small_font.render("SCORE", True, WHITE)
        self.screen.blit(score_label, (right_x + 15, stats_y))
        score_value = self.font.render(str(self.env.score), True, WHITE)
        self.screen.blit(score_value, (right_x + 15, stats_y + 20))

        lines_label = self.small_font.render(f"LINES: {self.env.lines_cleared}", True, WHITE)
        self.screen.blit(lines_label, (right_x + 15, stats_y + 55))

        level_label = self.small_font.render(f"LEVEL: {self.env.level}", True, WHITE)
        self.screen.blit(level_label, (right_x + 15, stats_y + 77))

    def draw_overlay(self):
        if self.paused:
            self.draw_message("PAUSED", "Press P to continue")
        elif self.env.game_over:
            self.draw_message("GAME OVER", f"Score: {self.env.score} | Press R to restart")

    def draw_message(self, title, subtitle):
        board_x = LEFT_SIDEBAR_WIDTH
        board_center_x = board_x + BOARD_WIDTH * CELL_SIZE // 2
        board_center_y = BOARD_HEIGHT * CELL_SIZE // 2

        # Semi-transparent overlay over board area
        overlay = pygame.Surface((BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (board_x, 0))

        # Title
        title_text = self.font.render(title, True, WHITE)
        title_rect = title_text.get_rect(center=(board_center_x, board_center_y - 20))
        self.screen.blit(title_text, title_rect)

        # Subtitle
        subtitle_text = self.small_font.render(subtitle, True, WHITE)
        subtitle_rect = subtitle_text.get_rect(center=(board_center_x, board_center_y + 20))
        self.screen.blit(subtitle_text, subtitle_rect)

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_board()
        self.draw_sidebar()
        self.draw_overlay()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)

            running = self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()


def main():
    game = TetrisGame()
    game.run()


if __name__ == "__main__":
    main()
