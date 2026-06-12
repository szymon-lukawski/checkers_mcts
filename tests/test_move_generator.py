"""
Testy dla engine/move_generator.py

Schemat indeksowania:
   .  0  .  1  .  2  .  3    row 0 (even): cols 1,3,5,7
   4  .  5  .  6  .  7  .    row 1 (odd):  cols 0,2,4,6
   .  8  .  9  . 10  . 11    row 2 (even)
  12  . 13  . 14  . 15  .    row 3 (odd)
   . 16  . 17  . 18  . 19    row 4 (even)
  20  . 21  . 22  . 23  .    row 5 (odd)
   . 24  . 25  . 26  . 27    row 6 (even)
  28  . 29  . 30  . 31  .    row 7 (odd)

Białe idą DO GÓRY (ku sq 0-3). Czarne idą W DÓŁ (ku sq 28-31).
WHITE_PROMO = bity 0-3 (0xF). BLACK_PROMO = bity 28-31 (0xF0000000).
"""

import pytest
from engine.move_generator import (
    NEIGHBORS,
    WHITE_DIRS,
    BLACK_DIRS,
    ALL_DIRS,
    _build_neighbors,
    _find_captures_from,
    get_simple_moves,
    get_captures,
    get_legal_moves,
    apply_move,
    is_game_over,
    BOARD_MASK,
    WHITE_PROMO,
    BLACK_PROMO,
)
from models.board_state import Move


# ---------------------------------------------------------------------------
# NEIGHBORS table
# ---------------------------------------------------------------------------

class TestNeighbors:
    def test_neighbors_length(self):
        assert len(NEIGHBORS) == 32

    def test_sq0_top_left_corner(self):
        # sq0: row0 col1, UL=-1(edge), UR=-1(edge), DL=4, DR=5
        ul, ur, dl, dr = NEIGHBORS[0]
        assert ul == -1
        assert ur == -1
        assert dl == 4
        assert dr == 5

    def test_sq3_top_right_corner(self):
        # sq3: row0 col7, UL=-1, UR=-1, DL=7, DR=-1
        ul, ur, dl, dr = NEIGHBORS[3]
        assert ul == -1
        assert ur == -1
        assert dl == 7
        assert dr == -1

    def test_sq4_left_edge_row1(self):
        # sq4: row1 col0, UL=-1, UR=0, DL=-1, DR=8
        ul, ur, dl, dr = NEIGHBORS[4]
        assert ul == -1
        assert ur == 0
        assert dl == -1
        assert dr == 8

    def test_sq9_middle(self):
        # sq9: row2 col3, UL=5, UR=6, DL=13, DR=14
        ul, ur, dl, dr = NEIGHBORS[9]
        assert ul == 5
        assert ur == 6
        assert dl == 13
        assert dr == 14

    def test_sq11_right_edge_row2(self):
        # sq11: row2 col7, UL=7, UR=-1, DL=15, DR=-1
        ul, ur, dl, dr = NEIGHBORS[11]
        assert ul == 7
        assert ur == -1
        assert dl == 15
        assert dr == -1

    def test_sq17_center(self):
        # sq17: row4 col3, UL=13, UR=14, DL=21, DR=22
        ul, ur, dl, dr = NEIGHBORS[17]
        assert ul == 13
        assert ur == 14
        assert dl == 21
        assert dr == 22

    def test_sq28_bottom_left(self):
        # sq28: row7 col0, UL=-1, UR=24, DL=-1, DR=-1
        ul, ur, dl, dr = NEIGHBORS[28]
        assert ul == -1
        assert ur == 24
        assert dl == -1
        assert dr == -1

    def test_sq31_bottom_right(self):
        # sq31: row7 col6, UL=26, UR=27, DL=-1, DR=-1
        ul, ur, dl, dr = NEIGHBORS[31]
        assert ul == 26
        assert ur == 27
        assert dl == -1
        assert dr == -1

    def test_build_neighbors_returns_same(self):
        fresh = _build_neighbors()
        assert fresh == NEIGHBORS

    def test_directions_constants(self):
        assert WHITE_DIRS == (0, 1)
        assert BLACK_DIRS == (2, 3)
        assert ALL_DIRS == (0, 1, 2, 3)

    def test_constants_module_level(self):
        assert BOARD_MASK == 0xFFFF_FFFF
        assert WHITE_PROMO == 0x0000_000F
        assert BLACK_PROMO == 0xF000_0000


# ---------------------------------------------------------------------------
# get_simple_moves
# ---------------------------------------------------------------------------

class TestGetSimpleMoves:
    def test_white_pawn_moves_up(self):
        # Białe at sq21 (row5): UL=16, UR=17
        wp = 1 << 21
        moves = get_simple_moves(wp, 0, 0, 1)
        to_sqs = {m.to_sq for m in moves}
        assert to_sqs == {16, 17}
        assert all(m.from_sq == 21 for m in moves)
        assert all(m.captured == [] for m in moves)

    def test_white_pawn_blocked_by_own(self):
        # sq8 blocked by own pieces at sq4 (UL) and sq5 (UR)
        wp = (1 << 8) | (1 << 4) | (1 << 5)
        moves = get_simple_moves(wp, 0, 0, 1)
        # sq8 has no moves, sq4 and sq5 can still move
        from_sqs = {m.from_sq for m in moves}
        assert 8 not in from_sqs

    def test_white_pawn_at_top_row_no_moves(self):
        # sq0 at top (row0): UL=-1, UR=-1 → żadnych ruchów
        wp = 1 << 0
        moves = get_simple_moves(wp, 0, 0, 1)
        assert moves == []

    def test_black_pawn_moves_down(self):
        # Czarne at sq9 (row2): DL=13, DR=14
        bp = 1 << 9
        moves = get_simple_moves(0, bp, 0, 0)
        to_sqs = {m.to_sq for m in moves}
        assert to_sqs == {13, 14}
        assert all(m.from_sq == 9 for m in moves)

    def test_black_pawn_at_bottom_row_no_moves(self):
        # sq28 at row7: DL=-1, DR=-1
        bp = 1 << 28
        moves = get_simple_moves(0, bp, 0, 0)
        assert moves == []

    def test_black_pawn_blocked_by_opponent(self):
        # Czarne at sq9, białe at sq13 (DL) and sq14 (DR) → zablokowane
        wp = (1 << 13) | (1 << 14)
        bp = 1 << 9
        moves = get_simple_moves(wp, bp, 0, 0)
        from_sqs = {m.from_sq for m in moves}
        assert 9 not in from_sqs

    def test_king_flies_diagonal(self):
        # Białe damka at sq20 (row5, col0): UR chain → 16,13,9,6,2; DR chain → 24,29
        wp = 1 << 20
        kings = 1 << 20
        moves = get_simple_moves(wp, 0, kings, 1)
        to_sqs = {m.to_sq for m in moves}
        # UR direction from sq20: 16,13,9,6,2
        assert 16 in to_sqs
        assert 13 in to_sqs
        assert 9 in to_sqs
        # DR direction: 24,29
        assert 24 in to_sqs
        assert 29 in to_sqs

    def test_king_blocked_by_own_piece(self):
        # Damka at sq17, własny pionek at sq13 (UL)
        wp = (1 << 17) | (1 << 13)
        kings = 1 << 17
        moves = get_simple_moves(wp, 0, kings, 1)
        to_sqs = {m.to_sq for m in moves if m.from_sq == 17}
        # sq13 is blocked, sq8 and sq4 are behind sq13 → not reachable
        assert 13 not in to_sqs
        assert 8 not in to_sqs

    def test_king_moves_all_directions(self):
        # Damka at sq17: UL=13,UR=14,DL=21,DR=22 (and further)
        wp = 1 << 17
        kings = 1 << 17
        moves = get_simple_moves(wp, 0, kings, 1)
        to_sqs = {m.to_sq for m in moves}
        assert 13 in to_sqs  # UL
        assert 14 in to_sqs  # UR
        assert 21 in to_sqs  # DL
        assert 22 in to_sqs  # DR

    def test_empty_board_no_moves(self):
        moves = get_simple_moves(0, 0, 0, 1)
        assert moves == []

    def test_multiple_white_pawns(self):
        # Dwa białe pionki
        wp = (1 << 21) | (1 << 22)
        moves = get_simple_moves(wp, 0, 0, 1)
        from_sqs = {m.from_sq for m in moves}
        assert 21 in from_sqs
        assert 22 in from_sqs


# ---------------------------------------------------------------------------
# _find_captures_from
# ---------------------------------------------------------------------------

class TestFindCapturesFrom:
    def test_pawn_single_capture(self):
        # Białe at sq14, czarne at sq9, ląduje at sq5
        # sq14 UL=9; sq9 UL=5
        wp = 1 << 14
        bp = 1 << 9
        results = _find_captures_from(14, wp, bp, 0, 1, 0, False)
        assert len(results) >= 1
        # Find capture over sq9 landing at sq5
        lands = {(final_sq, tuple(cap_list)) for (final_sq, cap_list) in results}
        assert (5, (9,)) in lands

    def test_pawn_multi_jump(self):
        # Białe at sq22, czarne at sq17 i sq9
        # sq22 UL=17; sq17 UL=13 -> land at 13; then sq13 UL=8... wait
        # Actually: cap sq17, land at sq13; from sq13 cap sq9, land at sq5 or sq6?
        # sq13 UL=8, UR=9. Cap sq9 (UR dir from sq13), land at NEIGHBORS[9][1]=6
        wp = 1 << 22
        bp = (1 << 17) | (1 << 9)
        results = _find_captures_from(22, wp, bp, 0, 1, 0, False)
        # Should find 2-jump path
        two_jump = [(fs, cl) for (fs, cl) in results if len(cl) == 2]
        assert len(two_jump) >= 1

    def test_king_single_capture(self):
        # Damka at sq17, czarne at sq13 (UL dir)
        # King can land at sq8 or sq4 (further along diagonal)
        wp = 1 << 17
        bp = 1 << 13
        kings = 1 << 17
        results = _find_captures_from(17, wp, bp, kings, 1, 0, True)
        land_sqs = {fs for (fs, _) in results}
        assert 8 in land_sqs  # lands at sq8
        assert 4 in land_sqs  # or sq4 (further along)

    def test_no_captures_available(self):
        # Brak sąsiednich przeciwników
        wp = 1 << 17
        bp = 1 << 0  # far away
        results = _find_captures_from(17, wp, bp, 0, 1, 0, False)
        assert results == []

    def test_cant_capture_own_piece(self):
        # Biały at sq14, biały at sq9 (własny) → brak bicia
        wp = (1 << 14) | (1 << 9)
        bp = 0
        results = _find_captures_from(14, wp, bp, 0, 1, 0, False)
        assert results == []

    def test_capture_blocked_by_piece_behind(self):
        # Biały at sq14, czarny at sq9, biały at sq5 → bicie zablokowane
        wp = (1 << 14) | (1 << 5)
        bp = 1 << 9
        results = _find_captures_from(14, wp, bp, 0, 1, 0, False)
        assert results == []


# ---------------------------------------------------------------------------
# get_captures
# ---------------------------------------------------------------------------

class TestGetCaptures:
    def test_white_single_capture(self):
        # sq14 bije sq9, ląduje sq5
        wp = 1 << 14
        bp = 1 << 9
        captures = get_captures(wp, bp, 0, 1)
        assert len(captures) == 1
        assert captures[0].from_sq == 14
        assert captures[0].to_sq == 5
        assert captures[0].captured == [9]

    def test_black_single_capture(self):
        # sq9 bije sq14, ląduje sq18
        wp = 1 << 14
        bp = 1 << 9
        captures = get_captures(wp, bp, 0, 0)
        assert len(captures) == 1
        assert captures[0].from_sq == 9
        assert captures[0].to_sq == 18
        assert captures[0].captured == [14]

    def test_majority_rule_returns_max_captures(self):
        # sq22 ma 2-skok (przez sq17, sq9), sq25 ma tylko 1-skok (przez sq21)
        # Zasada większości: tylko ścieżka z 2 biciami
        # sq25: UL=21, sq21: UL=16 -> white at sq25 captures sq21 landing at sq16 (1 capture)
        # sq22: UL=17, sq17: UL=13 -> then sq13: UR=9 -> captures 2 (sq17, sq9)
        wp = (1 << 25) | (1 << 22)
        bp = (1 << 17) | (1 << 9) | (1 << 21)
        captures = get_captures(wp, bp, 0, 1)
        # Only 2-jump paths should remain
        assert all(len(c.captured) == 2 for c in captures)

    def test_no_captures_returns_empty(self):
        wp = 1 << 21
        bp = 1 << 5
        captures = get_captures(wp, bp, 0, 1)
        assert captures == []

    def test_king_capture(self):
        # Biała damka at sq17, czarna at sq13 (UL)
        wp = 1 << 17
        bp = 1 << 13
        kings = 1 << 17
        captures = get_captures(wp, bp, kings, 1)
        assert len(captures) >= 1
        assert all(13 in c.captured for c in captures)

    def test_capture_returns_move_objects(self):
        wp = 1 << 14
        bp = 1 << 9
        captures = get_captures(wp, bp, 0, 1)
        for c in captures:
            assert isinstance(c, Move)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------

class TestGetLegalMoves:
    def test_captures_take_priority(self):
        # Gdy bicia są dostępne, zwraca tylko bicia
        wp = 1 << 14
        bp = 1 << 9
        legal = get_legal_moves(wp, bp, 0, 1)
        assert all(len(m.captured) > 0 for m in legal)

    def test_simple_moves_when_no_captures(self):
        # Brak bić → zwykłe ruchy
        wp = 1 << 21
        bp = 1 << 0  # far away
        legal = get_legal_moves(wp, bp, 0, 1)
        assert all(m.captured == [] for m in legal)

    def test_initial_position_no_captures(self):
        # Pozycja startowa: brak bić
        black = (1 << 12) - 1
        white = ((1 << 12) - 1) << 20
        legal = get_legal_moves(white, black, 0, 1)
        assert all(m.captured == [] for m in legal)
        assert len(legal) > 0

    def test_no_legal_moves_when_stuck(self):
        # Białe at top row (sq0-3), czarne at sq28 (daleko)
        wp = 0xF
        bp = 1 << 28
        legal = get_legal_moves(wp, bp, 0, 1)
        assert legal == []


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------

class TestApplyMove:
    def test_white_normal_move(self):
        # Biały at sq21 → sq16
        wp = 1 << 21
        m = Move(from_sq=21, to_sq=16)
        new_wp, new_bp, new_kings, new_player = apply_move(wp, 0, 0, 1, m)
        assert new_wp & (1 << 16)
        assert not (new_wp & (1 << 21))
        assert new_player == 0

    def test_black_normal_move(self):
        # Czarny at sq9 → sq13
        bp = 1 << 9
        m = Move(from_sq=9, to_sq=13)
        new_wp, new_bp, new_kings, new_player = apply_move(0, bp, 0, 0, m)
        assert new_bp & (1 << 13)
        assert not (new_bp & (1 << 9))
        assert new_player == 1

    def test_white_promotion(self):
        # Biały pawn at sq5, moves to sq0 (WHITE_PROMO row)
        wp = 1 << 5
        m = Move(from_sq=5, to_sq=0)
        new_wp, new_bp, new_kings, _ = apply_move(wp, 0, 0, 1, m)
        assert new_kings & (1 << 0)  # sq0 is crowned

    def test_white_promotion_sq1(self):
        # Biały at sq5 → sq1
        wp = 1 << 5
        m = Move(from_sq=5, to_sq=1)
        new_wp, new_bp, new_kings, _ = apply_move(wp, 0, 0, 1, m)
        assert new_kings & (1 << 1)

    def test_black_promotion(self):
        # Czarny pawn at sq25, moves to sq29 (BLACK_PROMO row)
        bp = 1 << 25
        m = Move(from_sq=25, to_sq=29)
        new_wp, new_bp, new_kings, _ = apply_move(0, bp, 0, 0, m)
        assert new_kings & (1 << 29)

    def test_king_move_transfers_flag(self):
        # Biała damka at sq17 → sq13
        wp = 1 << 17
        kings = 1 << 17
        m = Move(from_sq=17, to_sq=13)
        new_wp, _, new_kings, _ = apply_move(wp, 0, kings, 1, m)
        assert new_kings & (1 << 13)
        assert not (new_kings & (1 << 17))

    def test_capture_removes_opponent(self):
        # Biały at sq14 bije sq9, ląduje sq5
        wp = 1 << 14
        bp = 1 << 9
        m = Move(from_sq=14, to_sq=5, captured=[9])
        new_wp, new_bp, new_kings, _ = apply_move(wp, bp, 0, 1, m)
        assert not (new_bp & (1 << 9))  # sq9 usunięty
        assert new_wp & (1 << 5)

    def test_capture_removes_king_flag_from_captured(self):
        # Czarna damka at sq17 jest bita przez białego at sq22
        wp = 1 << 22
        bp = 1 << 17
        kings = 1 << 17  # czarna damka
        m = Move(from_sq=22, to_sq=13, captured=[17])
        _, new_bp, new_kings, _ = apply_move(wp, bp, kings, 1, m)
        assert not (new_kings & (1 << 17))  # king flag usunięty
        assert not (new_bp & (1 << 17))     # piece usunięty

    def test_player_alternates(self):
        wp = 1 << 21
        m = Move(from_sq=21, to_sq=16)
        _, _, _, new_player = apply_move(wp, 0, 0, 1, m)
        assert new_player == 0
        # Czarny ruch
        bp = 1 << 9
        m2 = Move(from_sq=9, to_sq=13)
        _, _, _, new_player2 = apply_move(0, bp, 0, 0, m2)
        assert new_player2 == 1

    def test_result_masked_to_board_mask(self):
        wp = 1 << 21
        m = Move(from_sq=21, to_sq=16)
        new_wp, new_bp, new_kings, _ = apply_move(wp, 0, 0, 1, m)
        assert new_wp & ~BOARD_MASK == 0
        assert new_bp & ~BOARD_MASK == 0
        assert new_kings & ~BOARD_MASK == 0

    def test_black_capture_removes_white(self):
        # Czarny at sq9 bije sq14, ląduje sq18
        wp = 1 << 14
        bp = 1 << 9
        m = Move(from_sq=9, to_sq=18, captured=[14])
        new_wp, new_bp, _, _ = apply_move(wp, bp, 0, 0, m)
        assert not (new_wp & (1 << 14))
        assert new_bp & (1 << 18)


# ---------------------------------------------------------------------------
# is_game_over
# ---------------------------------------------------------------------------

class TestIsGameOver:
    def test_white_wins_when_bp_zero(self):
        assert is_game_over(1, 0, 1, 0) == 1

    def test_black_wins_when_wp_zero(self):
        assert is_game_over(0, 1, 0, 0) == -1

    def test_game_ongoing(self):
        # Pozycja startowa → gra trwa
        black = (1 << 12) - 1
        white = ((1 << 12) - 1) << 20
        assert is_game_over(white, black, 1, 0) == 0

    def test_white_no_legal_moves_returns_minus1(self):
        # Białe na górnym rzędzie (sq0-3), czarne daleko (sq28) → brak ruchów białych
        wp = 0xF
        bp = 1 << 28
        assert is_game_over(wp, bp, 1, 0) == -1

    def test_black_no_legal_moves_returns_1(self):
        # Czarne na dolnym rzędzie (sq28-31), białe daleko (sq0) → brak ruchów czarnych
        wp = 1 << 0
        bp = (1 << 28) | (1 << 29) | (1 << 30) | (1 << 31)
        assert is_game_over(wp, bp, 0, 0) == 1

    def test_both_zeros_bp_zero_takes_priority(self):
        # Gdy wp != 0 i bp == 0 → białe wygrały
        assert is_game_over(1 << 5, 0, 1, 0) == 1

    def test_result_is_int(self):
        result = is_game_over(1, 0, 1, 0)
        assert isinstance(result, int)
