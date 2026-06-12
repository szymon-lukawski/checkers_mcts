"""
Testy dla ui/renderer.py
Wymaga SDL_VIDEODRIVER=dummy (ustawione w conftest.py).
"""

import os
import pytest
import pygame

from models.board_state import BoardState, Move
from ui.renderer import (
    _sq_to_rc,
    Renderer,
    C_DARK,
    C_LIGHT,
    C_WHITE_PIECE,
    C_BLACK_PIECE,
    C_KING_MARK,
    C_SELECTED,
    C_LEGAL_DOT,
    C_LAST_MOVE,
    C_STATUS_BG,
    C_STATUS_FG,
)


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def surface():
    return pygame.Surface((800, 800))


@pytest.fixture
def renderer():
    return Renderer(window_size=720)


@pytest.fixture
def initial_state():
    return BoardState.initial()


# ---------------------------------------------------------------------------
# _sq_to_rc
# ---------------------------------------------------------------------------

class TestSqToRc:
    def test_sq0_even_row(self):
        # sq0: row=0 (even), col = 1+2*0 = 1
        r, c = _sq_to_rc(0)
        assert r == 0
        assert c == 1

    def test_sq1_even_row(self):
        # sq1: row=0, col = 1+2*1 = 3
        r, c = _sq_to_rc(1)
        assert r == 0
        assert c == 3

    def test_sq3_even_row(self):
        # sq3: row=0, col = 1+2*3 = 7
        r, c = _sq_to_rc(3)
        assert r == 0
        assert c == 7

    def test_sq4_odd_row(self):
        # sq4: row=1 (odd), col = 2*0 = 0
        r, c = _sq_to_rc(4)
        assert r == 1
        assert c == 0

    def test_sq5_odd_row(self):
        # sq5: row=1, col = 2*1 = 2
        r, c = _sq_to_rc(5)
        assert r == 1
        assert c == 2

    def test_sq7_odd_row(self):
        # sq7: row=1, col = 2*3 = 6
        r, c = _sq_to_rc(7)
        assert r == 1
        assert c == 6

    def test_sq17_even_row(self):
        # sq17: row=4 (even), col = 1+2*1 = 3
        r, c = _sq_to_rc(17)
        assert r == 4
        assert c == 3

    def test_sq28_odd_row(self):
        # sq28: row=7 (odd), col = 2*0 = 0
        r, c = _sq_to_rc(28)
        assert r == 7
        assert c == 0

    def test_sq31_odd_row(self):
        # sq31: row=7, col = 2*3 = 6
        r, c = _sq_to_rc(31)
        assert r == 7
        assert c == 6

    def test_all_squares_in_range(self):
        for sq in range(32):
            r, c = _sq_to_rc(sq)
            assert 0 <= r <= 7
            assert 0 <= c <= 7


# ---------------------------------------------------------------------------
# Renderer.__init__
# ---------------------------------------------------------------------------

class TestRendererInit:
    def test_window_size_stored(self, renderer):
        assert renderer.window_size == 720

    def test_sq_size_calculated(self, renderer):
        assert renderer.sq_size == 720 // 8

    def test_piece_radius(self, renderer):
        assert renderer.piece_r == renderer.sq_size * 38 // 100

    def test_dot_radius(self, renderer):
        assert renderer.dot_r == renderer.sq_size * 14 // 100

    def test_font_initially_none(self, renderer):
        assert renderer._font is None

    def test_custom_window_size(self):
        r = Renderer(window_size=800)
        assert r.window_size == 800
        assert r.sq_size == 100


# ---------------------------------------------------------------------------
# sq_center
# ---------------------------------------------------------------------------

class TestSqCenter:
    def test_sq0_center(self, renderer):
        s = renderer.sq_size
        x, y = renderer.sq_center(0)
        # sq0: row=0, col=1 → x = 1*s + s//2, y = 0*s + s//2
        assert x == 1 * s + s // 2
        assert y == 0 * s + s // 2

    def test_sq5_center(self, renderer):
        s = renderer.sq_size
        x, y = renderer.sq_center(5)
        # sq5: row=1, col=2 → x = 2*s + s//2, y = 1*s + s//2
        assert x == 2 * s + s // 2
        assert y == 1 * s + s // 2

    def test_all_squares_return_valid_coords(self, renderer):
        for sq in range(32):
            x, y = renderer.sq_center(sq)
            assert 0 <= x <= renderer.window_size
            assert 0 <= y <= renderer.window_size


# ---------------------------------------------------------------------------
# pixel_to_sq
# ---------------------------------------------------------------------------

class TestPixelToSq:
    def test_dark_square_sq0(self, renderer):
        s = renderer.sq_size
        # sq0: row=0, col=1 → pixels col=1*s..2*s, row=0*s..1*s
        # Click in center of that cell
        px = 1 * s + s // 2
        py = 0 * s + s // 2
        sq = renderer.pixel_to_sq(px, py)
        assert sq == 0

    def test_dark_square_sq4(self, renderer):
        s = renderer.sq_size
        # sq4: row=1 (odd), col=0 → pixels col=0, row=1*s
        px = 0 * s + s // 2
        py = 1 * s + s // 2
        sq = renderer.pixel_to_sq(px, py)
        assert sq == 4

    def test_light_square_returns_none_even_row(self, renderer):
        s = renderer.sq_size
        # Light square: row=0 (even), col=0 (even col, odd row=light)
        px = 0 * s + s // 2  # col=0 (even), row=0 (even) → light
        py = 0 * s + s // 2
        sq = renderer.pixel_to_sq(px, py)
        assert sq is None

    def test_light_square_returns_none_odd_row(self, renderer):
        s = renderer.sq_size
        # Light square: row=1 (odd), col=1 (odd) → light
        px = 1 * s + s // 2
        py = 1 * s + s // 2
        sq = renderer.pixel_to_sq(px, py)
        assert sq is None

    def test_out_of_bounds_returns_none(self, renderer):
        # Above board
        sq = renderer.pixel_to_sq(100, -1)
        assert sq is None

    def test_out_of_bounds_below(self, renderer):
        sq = renderer.pixel_to_sq(100, renderer.window_size + 10)
        assert sq is None

    def test_returns_valid_sq_range(self, renderer):
        s = renderer.sq_size
        for sq in range(32):
            x, y = renderer.sq_center(sq)
            result = renderer.pixel_to_sq(x, y)
            assert result == sq


# ---------------------------------------------------------------------------
# draw_board
# ---------------------------------------------------------------------------

class TestDrawBoard:
    def test_draw_board_no_exception(self, renderer, surface):
        renderer.draw_board(surface)  # should not raise

    def test_draw_board_fills_surface(self, renderer, surface):
        surface.fill((0, 0, 0))
        renderer.draw_board(surface)
        # Surface should have been drawn on


# ---------------------------------------------------------------------------
# draw_highlight
# ---------------------------------------------------------------------------

class TestDrawHighlight:
    def test_draw_highlight_no_exception(self, renderer, surface):
        renderer.draw_highlight(surface, [0, 5, 10], C_SELECTED, 80)

    def test_draw_highlight_empty_list(self, renderer, surface):
        renderer.draw_highlight(surface, [], C_SELECTED, 80)


# ---------------------------------------------------------------------------
# draw_pieces
# ---------------------------------------------------------------------------

class TestDrawPieces:
    def test_draw_pieces_minimal(self, renderer, surface, initial_state):
        renderer.draw_pieces(surface, initial_state)

    def test_draw_pieces_with_selected(self, renderer, surface, initial_state):
        renderer.draw_pieces(surface, initial_state, selected_sq=20)

    def test_draw_pieces_with_legal_targets(self, renderer, surface, initial_state):
        renderer.draw_pieces(surface, initial_state, legal_targets=[16, 17])

    def test_draw_pieces_with_last_move(self, renderer, surface, initial_state):
        last = Move(from_sq=21, to_sq=17)
        renderer.draw_pieces(surface, initial_state, last_move=last)

    def test_draw_pieces_all_params(self, renderer, surface, initial_state):
        last = Move(from_sq=21, to_sq=17)
        renderer.draw_pieces(
            surface,
            initial_state,
            selected_sq=20,
            legal_targets=[16, 17],
            last_move=last,
        )

    def test_draw_pieces_with_king(self, renderer, surface):
        state = BoardState(
            white_pieces=1 << 17,
            black_pieces=1 << 9,
            kings=1 << 17,
            current_player=1,
        )
        renderer.draw_pieces(surface, state)


# ---------------------------------------------------------------------------
# _draw_piece
# ---------------------------------------------------------------------------

class TestDrawPiece:
    def test_draw_white_pawn(self, renderer, surface):
        renderer._draw_piece(surface, 17, C_WHITE_PIECE, False)

    def test_draw_black_pawn(self, renderer, surface):
        renderer._draw_piece(surface, 17, C_BLACK_PIECE, False)

    def test_draw_white_king(self, renderer, surface):
        renderer._draw_piece(surface, 17, C_WHITE_PIECE, True)

    def test_draw_black_king(self, renderer, surface):
        renderer._draw_piece(surface, 17, C_BLACK_PIECE, True)


# ---------------------------------------------------------------------------
# draw_status
# ---------------------------------------------------------------------------

class TestDrawStatus:
    def test_draw_status_first_call_creates_font(self, renderer, surface):
        assert renderer._font is None
        renderer.draw_status(surface, "Test", 0)
        assert renderer._font is not None

    def test_draw_status_second_call_reuses_font(self, renderer, surface):
        renderer.draw_status(surface, "First", 0)
        font = renderer._font
        renderer.draw_status(surface, "Second", 0)
        assert renderer._font is font  # same object

    def test_draw_status_no_exception(self, renderer, surface):
        renderer.draw_status(surface, "Ruch: Białe  |  Białe: 12  Czarne: 12")

    def test_draw_status_with_offset(self, renderer, surface):
        renderer.draw_status(surface, "Test", y_offset=36)
