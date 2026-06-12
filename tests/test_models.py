"""
Testy dla models/board_state.py
"""

import pytest
from pydantic import ValidationError
from models.board_state import Move, BoardState


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

class TestMove:
    def test_move_basic(self):
        m = Move(from_sq=5, to_sq=10)
        assert m.from_sq == 5
        assert m.to_sq == 10
        assert m.captured == []

    def test_move_with_captured(self):
        m = Move(from_sq=5, to_sq=15, captured=[9])
        assert m.captured == [9]

    def test_move_multiple_captured(self):
        m = Move(from_sq=5, to_sq=20, captured=[9, 14])
        assert len(m.captured) == 2

    def test_move_repr_without_captured(self):
        m = Move(from_sq=3, to_sq=7)
        r = repr(m)
        assert "3" in r
        assert "7" in r
        assert "x" not in r

    def test_move_repr_with_captured(self):
        m = Move(from_sq=3, to_sq=12, captured=[7])
        r = repr(m)
        assert "3" in r
        assert "12" in r
        assert "x" in r
        assert "7" in r

    def test_move_from_sq_out_of_range(self):
        with pytest.raises(ValidationError):
            Move(from_sq=-1, to_sq=5)

    def test_move_to_sq_out_of_range(self):
        with pytest.raises(ValidationError):
            Move(from_sq=0, to_sq=32)

    def test_move_from_sq_boundary(self):
        m = Move(from_sq=0, to_sq=31)
        assert m.from_sq == 0
        assert m.to_sq == 31

    def test_move_path_default_empty(self):
        m = Move(from_sq=5, to_sq=10)
        assert m.path == []

    def test_move_with_path(self):
        m = Move(from_sq=5, to_sq=20, captured=[9, 14], path=[13])
        assert m.path == [13]
        assert len(m.path) == 1


# ---------------------------------------------------------------------------
# BoardState
# ---------------------------------------------------------------------------

class TestBoardState:
    def test_valid_state(self):
        s = BoardState(white_pieces=1, black_pieces=2, kings=0, current_player=1)
        assert s.white_pieces == 1
        assert s.black_pieces == 2

    def test_overlap_raises(self):
        # Nakładające się pola – powinno zgłosić ValidationError
        with pytest.raises((ValidationError, AssertionError)):
            BoardState(white_pieces=0b11, black_pieces=0b01, kings=0, current_player=1)

    def test_no_overlap_valid(self):
        s = BoardState(white_pieces=0b01, black_pieces=0b10, kings=0, current_player=0)
        assert s.white_pieces == 1
        assert s.black_pieces == 2

    def test_to_tuple(self):
        s = BoardState(white_pieces=10, black_pieces=20, kings=5, current_player=0)
        t = s.to_tuple()
        assert t == (10, 20, 5, 0)

    def test_to_tuple_order(self):
        # 0b01 and 0b10 don't overlap
        s = BoardState(white_pieces=0b01, black_pieces=0b10, kings=0, current_player=1)
        wp, bp, kings, player = s.to_tuple()
        assert wp == 1
        assert bp == 2
        assert kings == 0
        assert player == 1

    def test_initial(self):
        s = BoardState.initial()
        # Czarne: bity 0-11
        black = (1 << 12) - 1
        # Białe: bity 20-31
        white = ((1 << 12) - 1) << 20
        assert s.black_pieces == black
        assert s.white_pieces == white
        assert s.kings == 0
        assert s.current_player == 1

    def test_initial_no_overlap(self):
        s = BoardState.initial()
        assert (s.white_pieces & s.black_pieces) == 0

    def test_initial_piece_counts(self):
        s = BoardState.initial()
        assert bin(s.white_pieces).count("1") == 12
        assert bin(s.black_pieces).count("1") == 12

    def test_current_player_out_of_range(self):
        with pytest.raises(ValidationError):
            BoardState(white_pieces=1, black_pieces=2, kings=0, current_player=2)

    def test_white_pieces_out_of_range(self):
        with pytest.raises(ValidationError):
            BoardState(white_pieces=2**32, black_pieces=0, kings=0, current_player=1)

    def test_no_overlap_validator_called(self):
        # Bezpośrednie testowanie modelu z nakładającymi się bitami
        with pytest.raises((ValidationError, AssertionError)):
            BoardState(white_pieces=0xFF, black_pieces=0xF0, kings=0, current_player=0)
