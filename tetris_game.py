#!/usr/bin/env python3
"""
Tetris game with Rust backend and Pygame visualization.
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
SIDEBAR_WIDTH = 200

WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE + SIDEBAR_WIDTH
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
        pygame.display.set_caption("Tetris (Rust Backend)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
        self.fall_time = 0
        self.fall_speed = 500  # milliseconds
        self.paused = False

        # Key repeat settings
        self.das_delay = 170  # Delayed Auto Shift delay (ms)
        self.das_interval = 50  # Auto repeat interval (ms)
        self.key_states = {}

    def get_fall_speed(self):
        """Get fall speed based on current level."""
        level = self.env.level
        # Speed increases with level
        speeds = [500, 450, 400, 350, 300, 250, 200, 150, 100, 80, 60]
        idx = min(level - 1, len(speeds) - 1)
        return speeds[idx]

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
                    continue

                if self.paused or self.env.game_over:
                    continue

                # Handle key presses
                if event.key == pygame.K_LEFT:
                    self.env.move_left()
                    self.key_states[pygame.K_LEFT] = {
                        'pressed': current_time,
                        'last_repeat': current_time
                    }
                elif event.key == pygame.K_RIGHT:
                    self.env.move_right()
                    self.key_states[pygame.K_RIGHT] = {
                        'pressed': current_time,
                        'last_repeat': current_time
                    }
                elif event.key == pygame.K_DOWN:
                    self.env.move_down()
                    self.key_states[pygame.K_DOWN] = {
                        'pressed': current_time,
                        'last_repeat': current_time
                    }
                elif event.key == pygame.K_UP or event.key == pygame.K_x:
                    self.env.rotate_cw()
                elif event.key == pygame.K_z:
                    self.env.rotate_ccw()
                elif event.key == pygame.K_SPACE:
                    self.env.hard_drop()
                    self.fall_time = 0

            if event.type == pygame.KEYUP:
                if event.key in self.key_states:
                    del self.key_states[event.key]

        # Handle key repeats (DAS)
        if not self.paused and not self.env.game_over:
            keys = pygame.key.get_pressed()
            for key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN]:
                if keys[key] and key in self.key_states:
                    state = self.key_states[key]
                    time_held = current_time - state['pressed']
                    time_since_repeat = current_time - state['last_repeat']

                    if time_held > self.das_delay and time_since_repeat > self.das_interval:
                        if key == pygame.K_LEFT:
                            self.env.move_left()
                        elif key == pygame.K_RIGHT:
                            self.env.move_right()
                        elif key == pygame.K_DOWN:
                            self.env.move_down()
                        state['last_repeat'] = current_time

        return True

    def update(self, dt):
        if self.paused or self.env.game_over:
            return

        self.fall_time += dt
        fall_speed = self.get_fall_speed()

        if self.fall_time >= fall_speed:
            self.env.tick()
            self.fall_time = 0

    def draw_board(self):
        # Draw board background
        board_rect = pygame.Rect(0, 0, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
        pygame.draw.rect(self.screen, BLACK, board_rect)

        # Draw grid lines
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                DARK_GRAY,
                (x * CELL_SIZE, 0),
                (x * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
            )
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                DARK_GRAY,
                (0, y * CELL_SIZE),
                (BOARD_WIDTH * CELL_SIZE, y * CELL_SIZE)
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

                    self.draw_cell(x, y, color)

        # Draw ghost piece
        ghost = self.env.get_ghost_piece()
        if ghost:
            color = ghost.get_color()
            ghost_surface = pygame.Surface((CELL_SIZE - 2, CELL_SIZE - 2), pygame.SRCALPHA)
            ghost_surface.fill((*color, GHOST_ALPHA))

            for (x, y) in ghost.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    self.screen.blit(ghost_surface, (x * CELL_SIZE + 1, y * CELL_SIZE + 1))

        # Draw current piece
        piece = self.env.get_current_piece()
        if piece:
            color = piece.get_color()
            for (x, y) in piece.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    self.draw_cell(x, y, color)

    def draw_cell(self, x, y, color):
        """Draw a single cell with 3D effect."""
        rect = pygame.Rect(
            x * CELL_SIZE + 1,
            y * CELL_SIZE + 1,
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
        sidebar_x = BOARD_WIDTH * CELL_SIZE
        sidebar_rect = pygame.Rect(sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, sidebar_rect)

        # Draw next piece preview
        next_label = self.font.render("NEXT", True, WHITE)
        self.screen.blit(next_label, (sidebar_x + 20, 20))

        next_piece = self.env.get_next_piece()
        if next_piece:
            preview_x = sidebar_x + 30
            preview_y = 60
            color = next_piece.get_color()
            shape = next_piece.get_shape()

            for dy, row in enumerate(shape):
                for dx, cell in enumerate(row):
                    if cell == 1:
                        rect = pygame.Rect(
                            preview_x + dx * 20,
                            preview_y + dy * 20,
                            18, 18
                        )
                        pygame.draw.rect(self.screen, color, rect)

        # Draw score
        score_label = self.font.render("SCORE", True, WHITE)
        self.screen.blit(score_label, (sidebar_x + 20, 180))
        score_value = self.font.render(str(self.env.score), True, WHITE)
        self.screen.blit(score_value, (sidebar_x + 20, 220))

        # Draw lines
        lines_label = self.font.render("LINES", True, WHITE)
        self.screen.blit(lines_label, (sidebar_x + 20, 280))
        lines_value = self.font.render(str(self.env.lines_cleared), True, WHITE)
        self.screen.blit(lines_value, (sidebar_x + 20, 320))

        # Draw level
        level_label = self.font.render("LEVEL", True, WHITE)
        self.screen.blit(level_label, (sidebar_x + 20, 380))
        level_value = self.font.render(str(self.env.level), True, WHITE)
        self.screen.blit(level_value, (sidebar_x + 20, 420))

        # Draw controls
        controls_y = 500
        controls = [
            "CONTROLS:",
            "← → : Move",
            "↓ : Soft drop",
            "↑/X : Rotate CW",
            "Z : Rotate CCW",
            "SPACE : Hard drop",
            "P : Pause",
            "R : Restart",
            "ESC : Quit"
        ]

        for i, line in enumerate(controls):
            text = self.small_font.render(line, True, WHITE)
            self.screen.blit(text, (sidebar_x + 20, controls_y + i * 25))

    def draw_overlay(self):
        if self.paused:
            self.draw_message("PAUSED", "Press P to continue")
        elif self.env.game_over:
            self.draw_message("GAME OVER", f"Score: {self.env.score} | Press R to restart")

    def draw_message(self, title, subtitle):
        # Semi-transparent overlay
        overlay = pygame.Surface((BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Title
        title_text = self.font.render(title, True, WHITE)
        title_rect = title_text.get_rect(center=(BOARD_WIDTH * CELL_SIZE // 2, BOARD_HEIGHT * CELL_SIZE // 2 - 20))
        self.screen.blit(title_text, title_rect)

        # Subtitle
        subtitle_text = self.small_font.render(subtitle, True, WHITE)
        subtitle_rect = subtitle_text.get_rect(center=(BOARD_WIDTH * CELL_SIZE // 2, BOARD_HEIGHT * CELL_SIZE // 2 + 20))
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
