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
LEFT_SIDEBAR_WIDTH = 120
RIGHT_SIDEBAR_WIDTH = 120

WINDOW_WIDTH = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE + RIGHT_SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
GRID_COLOR = (40, 40, 40)
BORDER_COLOR = (80, 80, 80)


class TetrisGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        self.env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
        self.fall_time = 0
        self.paused = False

        # DAS (Delayed Auto Shift) and ARR (Auto Repeat Rate) settings
        self.das = 100
        self.arr = 0

        # Soft drop settings
        self.soft_drop_das = 0
        self.soft_drop_arr = 25
        self.soft_drop_factor = 20

        # Key state tracking for DAS/ARR
        self.key_states = {}
        self.left_held = False
        self.right_held = False
        self.down_held = False

        # Mini block size for sidebar pieces
        self.mini_cell_size = 20

    def get_fall_speed(self):
        """Get fall speed in milliseconds (constant gravity)."""
        return 1000

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

                elif event.key == pygame.K_UP or event.key == pygame.K_s:
                    self.env.rotate_cw()

                elif event.key == pygame.K_a:
                    self.env.rotate_ccw()

                elif event.key == pygame.K_SPACE:
                    self.env.hard_drop()
                    self.fall_time = 0

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

        if not self.paused and not self.env.game_over:
            self._handle_das_arr(current_time)

        return True

    def _handle_das_arr(self, current_time):
        """Handle Delayed Auto Shift and Auto Repeat Rate for movement keys."""
        # Determine which horizontal key to process (most recently pressed wins)
        horizontal_key = None
        if pygame.K_LEFT in self.key_states and pygame.K_RIGHT in self.key_states:
            # Both held - use most recently pressed
            left_time = self.key_states[pygame.K_LEFT]['pressed_time']
            right_time = self.key_states[pygame.K_RIGHT]['pressed_time']
            horizontal_key = pygame.K_LEFT if left_time > right_time else pygame.K_RIGHT
        elif pygame.K_LEFT in self.key_states:
            horizontal_key = pygame.K_LEFT
        elif pygame.K_RIGHT in self.key_states:
            horizontal_key = pygame.K_RIGHT

        if horizontal_key:
            state = self.key_states[horizontal_key]
            time_held = current_time - state['pressed_time']
            time_since_move = current_time - state['last_move_time']

            if not state['das_charged']:
                if time_held >= self.das:
                    state['das_charged'] = True
                    if horizontal_key == pygame.K_LEFT:
                        self.env.move_left()
                    else:
                        self.env.move_right()
                    state['last_move_time'] = current_time
            else:
                if self.arr == 0:
                    if horizontal_key == pygame.K_LEFT:
                        while self.env.move_left():
                            pass
                    else:
                        while self.env.move_right():
                            pass
                    state['last_move_time'] = current_time
                elif time_since_move >= self.arr:
                    if horizontal_key == pygame.K_LEFT:
                        self.env.move_left()
                    else:
                        self.env.move_right()
                    state['last_move_time'] = current_time

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

        self.env.update_lock_delay(dt)
        self.fall_time += dt
        fall_speed = self.get_fall_speed()

        if self.down_held:
            fall_speed = fall_speed // self.soft_drop_factor

        if self.fall_time >= fall_speed:
            self.env.tick()
            self.fall_time = 0

    def draw_board(self):
        board_x = LEFT_SIDEBAR_WIDTH

        # Draw board background
        board_rect = pygame.Rect(board_x, 0, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
        pygame.draw.rect(self.screen, BLACK, board_rect)

        # Draw grid lines
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (board_x + x * CELL_SIZE, 0),
                (board_x + x * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
            )
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (board_x, y * CELL_SIZE),
                (board_x + BOARD_WIDTH * CELL_SIZE, y * CELL_SIZE)
            )

        # Draw border around the board
        pygame.draw.rect(self.screen, BORDER_COLOR, board_rect, 2)

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

        # Draw ghost piece (outline only)
        ghost = self.env.get_ghost_piece()
        if ghost:
            color = ghost.get_color()
            for (x, y) in ghost.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    rect = pygame.Rect(
                        board_x + x * CELL_SIZE + 1,
                        y * CELL_SIZE + 1,
                        CELL_SIZE - 2,
                        CELL_SIZE - 2
                    )
                    pygame.draw.rect(self.screen, color, rect, 2)

        # Draw current piece
        piece = self.env.get_current_piece()
        if piece:
            color = piece.get_color()
            for (x, y) in piece.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    self.draw_cell(board_x + x * CELL_SIZE, y * CELL_SIZE, color)

    def draw_cell(self, px, py, color):
        """Draw a single cell (flat style) at pixel coordinates."""
        rect = pygame.Rect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(self.screen, color, rect)

    def draw_mini_piece(self, piece, center_x, center_y):
        """Draw a mini piece centered at the given position."""
        shape = piece.get_shape()
        color = piece.get_color()

        # Find bounding box of the piece
        min_x, max_x = 4, -1
        min_y, max_y = 4, -1
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if cell == 1:
                    min_x = min(min_x, dx)
                    max_x = max(max_x, dx)
                    min_y = min(min_y, dy)
                    max_y = max(max_y, dy)

        piece_width = (max_x - min_x + 1) * self.mini_cell_size
        piece_height = (max_y - min_y + 1) * self.mini_cell_size

        # Calculate offset to center the piece
        start_x = center_x - piece_width // 2
        start_y = center_y - piece_height // 2

        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if cell == 1:
                    rect = pygame.Rect(
                        start_x + (dx - min_x) * self.mini_cell_size,
                        start_y + (dy - min_y) * self.mini_cell_size,
                        self.mini_cell_size - 1,
                        self.mini_cell_size - 1
                    )
                    pygame.draw.rect(self.screen, color, rect)

    def draw_sidebar(self):
        # Left sidebar - HOLD piece
        hold_piece = self.env.get_hold_piece()
        if hold_piece:
            hold_used = self.env.is_hold_used()
            # Draw hold piece centered in left sidebar
            center_x = LEFT_SIDEBAR_WIDTH // 2
            center_y = 60
            if hold_used:
                # Dim the color if hold was used this turn
                original_color = hold_piece.get_color()
                dimmed = tuple(c // 2 for c in original_color)
                # Temporarily modify for drawing
                shape = hold_piece.get_shape()
                min_x, max_x = 4, -1
                min_y, max_y = 4, -1
                for dy, row in enumerate(shape):
                    for dx, cell in enumerate(row):
                        if cell == 1:
                            min_x = min(min_x, dx)
                            max_x = max(max_x, dx)
                            min_y = min(min_y, dy)
                            max_y = max(max_y, dy)
                piece_width = (max_x - min_x + 1) * self.mini_cell_size
                piece_height = (max_y - min_y + 1) * self.mini_cell_size
                start_x = center_x - piece_width // 2
                start_y = center_y - piece_height // 2
                for dy, row in enumerate(shape):
                    for dx, cell in enumerate(row):
                        if cell == 1:
                            rect = pygame.Rect(
                                start_x + (dx - min_x) * self.mini_cell_size,
                                start_y + (dy - min_y) * self.mini_cell_size,
                                self.mini_cell_size - 1,
                                self.mini_cell_size - 1
                            )
                            pygame.draw.rect(self.screen, dimmed, rect)
            else:
                self.draw_mini_piece(hold_piece, center_x, center_y)

        # Lines display in bottom right corner
        right_x = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE
        lines_text = f"Lines: {self.env.lines_cleared}"
        lines_surface = self.font.render(lines_text, True, WHITE)
        self.screen.blit(lines_surface, (right_x + 15, WINDOW_HEIGHT - 40))

        # Right sidebar - NEXT pieces
        right_x = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE
        next_pieces = self.env.get_next_pieces(5)

        for i, piece in enumerate(next_pieces):
            center_x = right_x + RIGHT_SIDEBAR_WIDTH // 2
            center_y = 40 + i * 55
            self.draw_mini_piece(piece, center_x, center_y)

    def draw_overlay(self):
        if self.paused:
            self.draw_message("PAUSED", "Press P to continue")
        elif self.env.game_over:
            self.draw_message("GAME OVER", f"Attack: {self.env.attack} | Press R to restart")

    def draw_message(self, title, subtitle):
        board_x = LEFT_SIDEBAR_WIDTH
        board_center_x = board_x + BOARD_WIDTH * CELL_SIZE // 2
        board_center_y = BOARD_HEIGHT * CELL_SIZE // 2

        overlay = pygame.Surface((BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (board_x, 0))

        title_text = self.font.render(title, True, WHITE)
        title_rect = title_text.get_rect(center=(board_center_x, board_center_y - 20))
        self.screen.blit(title_text, title_rect)

        small_font = pygame.font.Font(None, 24)
        subtitle_text = small_font.render(subtitle, True, WHITE)
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
