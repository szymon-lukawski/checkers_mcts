"""
Testy dla engine/game_logic.py (klasa Board)
"""

import pytest
from engine.game_logic import Board
from models.board_state import BoardState, Move


class TestBoardInitial:
    def test_initial_creates_board(self):
        board = Board.initial()
        assert isinstance(board, Board)

    def test_initial_player_is_white(self):
        board = Board.initial()
        assert board.current_player == 1

    def test_initial_kings_empty(self):
        board = Board.initial()
        assert board.kings == 0

    def test_initial_piece_counts(self):
        board = Board.initial()
        w, b = board.piece_count()
        assert w == 12
        assert b == 12

    def test_initial_no_overlap(self):
        board = Board.initial()
        assert (board.wp & board.bp) == 0


class TestBoardFromState:
    def test_from_state_preserves_values(self):
        # white=0b01, black=0b10 (no overlap)
        state = BoardState(white_pieces=0b01, black_pieces=0b10, kings=0, current_player=0)
        board = Board.from_state(state)
        assert board.wp == 1
        assert board.bp == 2
        assert board.kings == 0
        assert board.current_player == 0

    def test_from_state_initial(self):
        state = BoardState.initial()
        board = Board.from_state(state)
        board2 = Board.initial()
        assert board.wp == board2.wp
        assert board.bp == board2.bp
        assert board.current_player == board2.current_player


class TestBoardToState:
    def test_to_state_roundtrip(self):
        board = Board.initial()
        state = board.to_state()
        assert isinstance(state, BoardState)
        assert state.white_pieces == board.wp
        assert state.black_pieces == board.bp
        assert state.kings == board.kings
        assert state.current_player == board.current_player

    def test_to_state_after_move(self):
        board = Board.initial()
        legal = board.get_legal_moves()
        new_board = board.apply_move(legal[0])
        state = new_board.to_state()
        assert state.current_player == 0  # after white moves


class TestBoardGetLegalMoves:
    def test_initial_has_legal_moves(self):
        board = Board.initial()
        moves = board.get_legal_moves()
        assert len(moves) > 0

    def test_returns_list_of_moves(self):
        board = Board.initial()
        moves = board.get_legal_moves()
        for m in moves:
            assert isinstance(m, Move)

    def test_terminal_board_no_moves(self):
        # Białe uwięzione w górnym rzędzie, czarne daleko → brak ruchów białych
        wp = 0xF
        bp = 1 << 28
        board = Board(wp, bp, 0, 1)
        moves = board.get_legal_moves()
        assert moves == []


class TestBoardApplyMove:
    def test_apply_move_returns_new_board(self):
        board = Board.initial()
        legal = board.get_legal_moves()
        new_board = board.apply_move(legal[0])
        assert isinstance(new_board, Board)
        assert new_board is not board

    def test_apply_move_alternates_player(self):
        board = Board.initial()
        legal = board.get_legal_moves()
        new_board = board.apply_move(legal[0])
        assert new_board.current_player == 0

    def test_apply_move_does_not_mutate(self):
        board = Board.initial()
        original_wp = board.wp
        legal = board.get_legal_moves()
        board.apply_move(legal[0])
        assert board.wp == original_wp

    def test_apply_capture_removes_piece(self):
        # Białe at sq14 biją sq9, lądują sq5
        board = Board(1 << 14, 1 << 9, 0, 1)
        legal = board.get_legal_moves()
        capture = [m for m in legal if m.captured]
        assert capture
        new_board = board.apply_move(capture[0])
        assert not (new_board.bp & (1 << 9))


class TestBoardIsTerminal:
    def test_initial_not_terminal(self):
        board = Board.initial()
        is_term, result = board.is_terminal()
        assert is_term is False
        assert result == 0

    def test_terminal_when_bp_zero(self):
        board = Board(1, 0, 0, 1)
        is_term, result = board.is_terminal()
        assert is_term is True
        assert result == 1

    def test_terminal_when_wp_zero(self):
        board = Board(0, 1, 0, 0)
        is_term, result = board.is_terminal()
        assert is_term is True
        assert result == -1

    def test_terminal_returns_tuple(self):
        board = Board.initial()
        result = board.is_terminal()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_not_terminal_result_is_zero(self):
        board = Board.initial()
        is_term, result = board.is_terminal()
        assert result == 0


class TestBoardPieceCount:
    def test_initial_piece_count(self):
        board = Board.initial()
        w, b = board.piece_count()
        assert w == 12
        assert b == 12

    def test_piece_count_after_move(self):
        # Bicie zmniejsza liczbę pionków
        board = Board(1 << 14, 1 << 9, 0, 1)
        legal = board.get_legal_moves()
        capture = [m for m in legal if m.captured]
        if capture:
            new_board = board.apply_move(capture[0])
            _, b = new_board.piece_count()
            assert b == 0

    def test_piece_count_returns_tuple(self):
        board = Board.initial()
        result = board.piece_count()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_board(self):
        board = Board(0, 0, 0, 1)
        w, b = board.piece_count()
        assert w == 0
        assert b == 0


class TestBoardRepr:
    def test_repr_contains_piece_counts(self):
        board = Board.initial()
        r = repr(board)
        assert "12" in r
        assert "W=" in r
        assert "B=" in r

    def test_repr_white_player(self):
        board = Board.initial()  # current_player = 1 (white)
        r = repr(board)
        assert "białe" in r.lower() or "biale" in r.lower() or "białe" in r

    def test_repr_black_player(self):
        board = Board(0, 1, 0, 0)  # current_player = 0 (black)
        r = repr(board)
        assert "czarne" in r.lower()

    def test_repr_is_string(self):
        board = Board.initial()
        assert isinstance(repr(board), str)
