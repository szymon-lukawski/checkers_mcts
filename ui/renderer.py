"""
Renderer – rysuje planszę, pionki i podpowiedzi ruchów.
"""

import pygame
from models.board_state import BoardState, Move

# Kolory
C_DARK   = (181, 136,  99)
C_LIGHT  = (240, 217, 181)
C_WHITE_PIECE = (230, 230, 210)
C_BLACK_PIECE = ( 40,  30,  20)
C_KING_MARK  = (255, 215,   0)   # złota korona
C_SELECTED   = (100, 200, 100)
C_LEGAL_DOT  = ( 80, 180,  80)
C_LAST_MOVE  = (200, 200,  60)
C_STATUS_BG  = ( 30,  30,  30)
C_STATUS_FG  = (220, 220, 220)


def _sq_to_rc(sq: int) -> tuple[int, int]:
    row = sq // 4
    col = (1 + 2 * (sq % 4)) if row % 2 == 0 else (2 * (sq % 4))
    return row, col


class Renderer:
    def __init__(self, window_size: int = 720) -> None:
        self.window_size = window_size
        self.sq_size = window_size // 8
        self.piece_r = self.sq_size * 38 // 100
        self.dot_r   = self.sq_size * 14 // 100
        self._font: pygame.font.Font | None = None

    # ------------------------------------------------------------------
    # Mapowania współrzędnych
    # ------------------------------------------------------------------

    def sq_center(self, sq: int) -> tuple[int, int]:
        row, col = _sq_to_rc(sq)
        x = col * self.sq_size + self.sq_size // 2
        y = row * self.sq_size + self.sq_size // 2
        return x, y

    def pixel_to_sq(self, px: int, py: int) -> int | None:
        """Zwraca indeks pola (0-31) dla kliknięcia lub None jeśli jasne pole."""
        col = px // self.sq_size
        row = py // self.sq_size
        if row < 0 or row > 7 or col < 0 or col > 7:
            return None
        if row % 2 == 0:
            if col % 2 == 0:
                return None  # jasne pole
            return row * 4 + (col - 1) // 2
        else:
            if col % 2 == 1:
                return None  # jasne pole
            return row * 4 + col // 2

    # ------------------------------------------------------------------
    # Rysowanie
    # ------------------------------------------------------------------

    def draw_board(self, surface: pygame.Surface) -> None:
        s = self.sq_size
        for row in range(8):
            for col in range(8):
                color = C_LIGHT if (row + col) % 2 == 0 else C_DARK
                pygame.draw.rect(surface, color, (col * s, row * s, s, s))

    def draw_highlight(
        self,
        surface: pygame.Surface,
        squares: list[int],
        color: tuple[int, int, int],
        alpha: int = 80,
    ) -> None:
        s = self.sq_size
        overlay = pygame.Surface((s, s), pygame.SRCALPHA)
        overlay.fill((*color, alpha))
        for sq in squares:
            row, col = _sq_to_rc(sq)
            surface.blit(overlay, (col * s, row * s))

    def draw_pieces(
        self,
        surface: pygame.Surface,
        state: BoardState,
        selected_sq: int | None = None,
        legal_targets: list[int] | None = None,
        last_move: Move | None = None,
        skip_sq: int | None = None,
    ) -> None:
        # Podświetl ostatni ruch
        if last_move:
            self.draw_highlight(
                surface,
                [last_move.from_sq, last_move.to_sq],
                C_LAST_MOVE,
                60,
            )

        # Podświetl wybrany pionek
        if selected_sq is not None:
            self.draw_highlight(surface, [selected_sq], C_SELECTED, 120)

        # Rysuj pionki
        for sq in range(32):
            if sq == skip_sq:
                continue
            bit = 1 << sq
            if state.white_pieces & bit:
                self._draw_piece(surface, sq, C_WHITE_PIECE, bool(state.kings & bit))
            elif state.black_pieces & bit:
                self._draw_piece(surface, sq, C_BLACK_PIECE, bool(state.kings & bit))

        # Zielone kropki dla legalnych celów
        if legal_targets:
            for sq in legal_targets:
                cx, cy = self.sq_center(sq)
                pygame.draw.circle(surface, C_LEGAL_DOT, (cx, cy), self.dot_r)

    def draw_animated_piece_for_sq(
        self,
        surface: pygame.Surface,
        sq: int,
        state: BoardState,
        cx: int,
        cy: int,
    ) -> None:
        """Draw the piece that was at `sq` at arbitrary pixel position."""
        bit = 1 << sq
        if state.white_pieces & bit:
            color = C_WHITE_PIECE
        elif state.black_pieces & bit:
            color = C_BLACK_PIECE
        else:
            return
        self._draw_piece(surface, None, color, bool(state.kings & bit), cx=cx, cy=cy)

    def _draw_piece(
        self,
        surface: pygame.Surface,
        sq: int | None,
        color: tuple[int, int, int],
        is_king: bool,
        cx: int | None = None,
        cy: int | None = None,
    ) -> None:
        if cx is None or cy is None:
            cx, cy = self.sq_center(sq)
        # Cień
        pygame.draw.circle(surface, (20, 20, 20), (cx + 2, cy + 3), self.piece_r)
        # Pionek
        pygame.draw.circle(surface, color, (cx, cy), self.piece_r)
        # Obwódka
        border = (180, 180, 160) if color == C_WHITE_PIECE else (80, 60, 40)
        pygame.draw.circle(surface, border, (cx, cy), self.piece_r, 2)
        # Znak damki
        if is_king:
            pygame.draw.circle(surface, C_KING_MARK, (cx, cy), self.piece_r // 3, 2)

    def draw_status(
        self,
        surface: pygame.Surface,
        text: str,
        y_offset: int = 0,
    ) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 18, bold=True)
        w = self.window_size
        bar_h = 32
        y = self.window_size + y_offset
        pygame.draw.rect(surface, C_STATUS_BG, (0, y, w, bar_h))
        img = self._font.render(text, True, C_STATUS_FG)
        surface.blit(img, (10, y + 7))
