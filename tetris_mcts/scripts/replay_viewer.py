"""
Replay viewer for Tetris games saved as JSONL.

Load game replays and step through moves with pygame visualization.

Controls:
    Right Arrow / Space: Next move
    Left Arrow: Previous move
    Up/Down Arrow: Next/Previous game
    R: Restart current game
    Q/Escape: Quit
    +/-: Speed up/slow down auto-play
    P: Toggle auto-play
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pygame

from tetris_core import TetrisEnv

# Constants (same as tetris_game.py)
CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
LEFT_SIDEBAR_WIDTH = 120
RIGHT_SIDEBAR_WIDTH = 120
TOP_PADDING = 60
BOTTOM_PADDING = 20

WINDOW_WIDTH = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE + RIGHT_SIDEBAR_WIDTH
WINDOW_HEIGHT = TOP_PADDING + BOARD_HEIGHT * CELL_SIZE + BOTTOM_PADDING

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
GRID_COLOR = (40, 40, 40)
BORDER_COLOR = (80, 80, 80)
HIGHLIGHT_COLOR = (100, 200, 100)


@dataclass
class ReplayMove:
    x: int
    y: int
    rotation: int
    attack: int


@dataclass
class GameReplay:
    seed: int
    moves: list[ReplayMove]
    total_attack: int
    num_moves: int


def load_replays(path: Path) -> list[GameReplay]:
    """Load game replays from a JSONL file."""
    replays = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            moves = [
                ReplayMove(
                    x=m["x"],
                    y=m["y"],
                    rotation=m["rotation"],
                    attack=m["attack"],
                )
                for m in data["moves"]
            ]
            replays.append(
                GameReplay(
                    seed=data["seed"],
                    moves=moves,
                    total_attack=data["total_attack"],
                    num_moves=data["num_moves"],
                )
            )
    return replays


class ReplayViewer:
    def __init__(self, replays: list[GameReplay]):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris Replay Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        self.replays = replays
        self.game_index = 0
        self.move_index = 0
        self.mini_cell_size = 20

        # Auto-play settings
        self.auto_play = False
        self.auto_play_delay = 500  # ms between moves
        self.last_auto_move = 0

        # State snapshots for rewinding
        self.states: list[TetrisEnv] = []
        self.cumulative_attacks: list[int] = []

        self._load_game()

    def _load_game(self):
        """Load current game and precompute all states."""
        replay = self.replays[self.game_index]
        self.states = []
        self.cumulative_attacks = []

        # Create initial state
        env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, replay.seed)
        self.states.append(env.clone_state())
        self.cumulative_attacks.append(0)

        cumulative = 0
        for move in replay.moves:
            # Find and execute the placement
            placements = env.get_all_placements()
            placement = None
            for p in placements:
                if p.piece.x == move.x and p.piece.y == move.y and p.piece.rotation == move.rotation:
                    placement = p
                    break

            if placement:
                env.execute_placement(placement)
                cumulative += move.attack
                self.states.append(env.clone_state())
                self.cumulative_attacks.append(cumulative)

        self.move_index = 0

    @property
    def current_replay(self) -> GameReplay:
        return self.replays[self.game_index]

    @property
    def current_env(self) -> TetrisEnv:
        return self.states[self.move_index]

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False

                if event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                    self._next_move()
                elif event.key == pygame.K_LEFT:
                    self._prev_move()
                elif event.key == pygame.K_UP:
                    self._prev_game()
                elif event.key == pygame.K_DOWN:
                    self._next_game()
                elif event.key == pygame.K_r:
                    self.move_index = 0
                elif event.key == pygame.K_p:
                    self.auto_play = not self.auto_play
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.auto_play_delay = max(50, self.auto_play_delay - 50)
                elif event.key == pygame.K_MINUS:
                    self.auto_play_delay = min(2000, self.auto_play_delay + 50)

        # Auto-play
        if self.auto_play:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_auto_move >= self.auto_play_delay:
                self._next_move()
                self.last_auto_move = current_time

        return True

    def _next_move(self):
        if self.move_index < len(self.states) - 1:
            self.move_index += 1

    def _prev_move(self):
        if self.move_index > 0:
            self.move_index -= 1

    def _next_game(self):
        if self.game_index < len(self.replays) - 1:
            self.game_index += 1
            self._load_game()

    def _prev_game(self):
        if self.game_index > 0:
            self.game_index -= 1
            self._load_game()

    def draw_cell(self, px: int, py: int, color: tuple):
        """Draw a single cell at pixel coordinates."""
        rect = pygame.Rect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(self.screen, color, rect)

    def draw_mini_piece(self, piece, center_x: int, center_y: int):
        """Draw a mini piece centered at the given position."""
        shape = piece.get_shape()
        color = piece.get_color()

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
                        self.mini_cell_size - 1,
                    )
                    pygame.draw.rect(self.screen, color, rect)

    def draw_board(self):
        env = self.current_env
        board_x = LEFT_SIDEBAR_WIDTH
        board_y = TOP_PADDING

        # Draw board background
        board_rect = pygame.Rect(
            board_x, board_y, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE
        )
        pygame.draw.rect(self.screen, BLACK, board_rect)

        # Draw grid lines
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (board_x + x * CELL_SIZE, board_y),
                (board_x + x * CELL_SIZE, board_y + BOARD_HEIGHT * CELL_SIZE),
            )
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (board_x, board_y + y * CELL_SIZE),
                (board_x + BOARD_WIDTH * CELL_SIZE, board_y + y * CELL_SIZE),
            )

        pygame.draw.rect(self.screen, BORDER_COLOR, board_rect, 2)

        # Draw locked pieces
        board = env.get_board()
        board_colors = env.get_board_colors()

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if board[y][x] != 0:
                    color_idx = board_colors[y][x]
                    if color_idx is not None:
                        color = env.get_color_for_type(color_idx)
                    else:
                        color = GRAY
                    self.draw_cell(
                        board_x + x * CELL_SIZE, board_y + y * CELL_SIZE, color
                    )

        # Draw current piece (if not at end of game)
        piece = env.get_current_piece()
        if piece:
            color = piece.get_color()
            for x, y in piece.get_cells():
                if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                    self.draw_cell(
                        board_x + x * CELL_SIZE, board_y + y * CELL_SIZE, color
                    )

    def draw_sidebar(self):
        env = self.current_env

        # Left sidebar - Hold piece
        hold_piece = env.get_hold_piece()
        if hold_piece:
            center_x = LEFT_SIDEBAR_WIDTH // 2
            center_y = TOP_PADDING + 50
            self.draw_mini_piece(hold_piece, center_x, center_y)

        # Right sidebar - Next pieces
        right_x = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE
        next_pieces = env.get_next_pieces(5)

        for i, piece in enumerate(next_pieces):
            center_x = right_x + RIGHT_SIDEBAR_WIDTH // 2
            center_y = TOP_PADDING + 30 + i * 55
            self.draw_mini_piece(piece, center_x, center_y)

        # Attack display
        attack = self.cumulative_attacks[self.move_index]
        attack_text = f"Attack: {attack}"
        attack_surface = self.font.render(attack_text, True, WHITE)
        self.screen.blit(
            attack_surface, (right_x + 15, WINDOW_HEIGHT - BOTTOM_PADDING - 15)
        )

    def draw_header(self):
        """Draw game/move info at the top."""
        replay = self.current_replay

        # Game info
        game_text = f"Game {self.game_index + 1}/{len(self.replays)} (seed: {replay.seed})"
        game_surface = self.font.render(game_text, True, WHITE)
        self.screen.blit(game_surface, (10, 10))

        # Move info
        move_text = f"Move {self.move_index}/{replay.num_moves}"
        move_surface = self.font.render(move_text, True, WHITE)
        move_rect = move_surface.get_rect(right=WINDOW_WIDTH - 10, top=10)
        self.screen.blit(move_surface, move_rect)

        # Final score
        score_text = f"Final: {replay.total_attack} attack"
        score_surface = self.small_font.render(score_text, True, GRAY)
        score_rect = score_surface.get_rect(centerx=WINDOW_WIDTH // 2, top=12)
        self.screen.blit(score_surface, score_rect)

        # Auto-play indicator
        if self.auto_play:
            auto_text = f"AUTO ({self.auto_play_delay}ms)"
            auto_surface = self.small_font.render(auto_text, True, HIGHLIGHT_COLOR)
            auto_rect = auto_surface.get_rect(centerx=WINDOW_WIDTH // 2, top=32)
            self.screen.blit(auto_surface, auto_rect)

    def draw_controls(self):
        """Draw control hints at bottom."""
        controls = "←/→: Move | ↑/↓: Game | P: Auto | +/-: Speed | R: Reset | Q: Quit"
        surface = self.small_font.render(controls, True, GRAY)
        rect = surface.get_rect(centerx=WINDOW_WIDTH // 2, bottom=WINDOW_HEIGHT - 3)
        self.screen.blit(surface, rect)

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_header()
        self.draw_board()
        self.draw_sidebar()
        self.draw_controls()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            running = self.handle_events()
            self.draw()
        pygame.quit()


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_viewer.py <replays.jsonl>")
        sys.exit(1)

    replay_path = Path(sys.argv[1])
    if not replay_path.exists():
        print(f"File not found: {replay_path}")
        sys.exit(1)

    print(f"Loading replays from {replay_path}...")
    replays = load_replays(replay_path)
    print(f"Loaded {len(replays)} games")

    if not replays:
        print("No replays found in file")
        sys.exit(1)

    viewer = ReplayViewer(replays)
    viewer.run()


if __name__ == "__main__":
    main()
