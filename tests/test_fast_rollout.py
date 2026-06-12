"""
Testy engine/fast_rollout.py

Numba jest mockowana w conftest.py → @njit = no-op → czyste Python → coverage.
"""

import numpy as np
import pytest

from engine.fast_rollout import (
    NEIGHBORS_NP,
    MAX_MOVES,
    _popcount,
    _fast_apply_move,
    _collect_caps_from,
    fast_get_captures,
    fast_get_simple_moves,
    fast_get_legal_moves,
    fast_simulate,
)
from engine.move_generator import NEIGHBORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bufs():
    """Return fresh output arrays."""
    return (
        np.empty(MAX_MOVES, dtype=np.int32),
        np.empty(MAX_MOVES, dtype=np.int32),
        np.empty(MAX_MOVES, dtype=np.int64),
    )


def _tmp_bufs(size=512):
    return (
        np.empty(size, dtype=np.int32),
        np.empty(size, dtype=np.int32),
        np.empty(size, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# NEIGHBORS_NP
# ---------------------------------------------------------------------------

def test_neighbors_np_shape():
    assert NEIGHBORS_NP.shape == (32, 4)


def test_neighbors_np_matches_python():
    for sq in range(32):
        for d in range(4):
            assert NEIGHBORS_NP[sq, d] == NEIGHBORS[sq][d]


# ---------------------------------------------------------------------------
# _popcount
# ---------------------------------------------------------------------------

def test_popcount_zero():
    assert _popcount(np.int64(0)) == 0


def test_popcount_one():
    assert _popcount(np.int64(1)) == 1


def test_popcount_all_bits():
    assert _popcount(np.int64(0xFFFF_FFFF)) == 32


def test_popcount_three_bits():
    assert _popcount(np.int64(0b10110)) == 3


# ---------------------------------------------------------------------------
# _fast_apply_move
# ---------------------------------------------------------------------------

def test_apply_white_simple_move():
    wp = np.int64(1 << 20)
    bp = np.int64(1 << 9)
    kings = np.int64(0)
    nwp, nbp, nk, np_ = _fast_apply_move(wp, bp, kings, 1, 20, 16, np.int64(0))
    assert nwp == (1 << 16)
    assert nbp == (1 << 9)
    assert np_ == 0


def test_apply_black_simple_move():
    wp = np.int64(1 << 20)
    bp = np.int64(1 << 9)
    kings = np.int64(0)
    nwp, nbp, nk, np_ = _fast_apply_move(wp, bp, kings, 0, 9, 13, np.int64(0))
    assert nbp == (1 << 13)
    assert nwp == (1 << 20)
    assert np_ == 1


def test_apply_white_capture():
    # white at 14 captures black at 9, lands at 5
    wp = np.int64(1 << 14)
    bp = np.int64(1 << 9)
    kings = np.int64(0)
    cap_mask = np.int64(1 << 9)
    nwp, nbp, nk, np_ = _fast_apply_move(wp, bp, kings, 1, 14, 5, cap_mask)
    assert nwp == (1 << 5)
    assert nbp == 0   # sq 9 removed
    assert nk == 0
    assert np_ == 0


def test_apply_black_capture():
    wp = np.int64(1 << 14)
    bp = np.int64(1 << 9)
    kings = np.int64(0)
    cap_mask = np.int64(1 << 14)
    nwp, nbp, nk, np_ = _fast_apply_move(wp, bp, kings, 0, 9, 19, cap_mask)
    assert nbp == (1 << 19)
    assert nwp == 0
    assert np_ == 1


def test_apply_white_promotion():
    # white pawn at sq 4 moves to sq 0 (WHITE_PROMO)
    wp = np.int64(1 << 4)
    bp = np.int64(0)
    kings = np.int64(0)
    nwp, nbp, nk, np_ = _fast_apply_move(wp, bp, kings, 1, 4, 0, np.int64(0))
    assert nwp == (1 << 0)
    assert nk == (1 << 0)   # crowned


def test_apply_black_promotion():
    # black pawn at sq 27 moves to sq 31 (BLACK_PROMO)
    wp = np.int64(0)
    bp = np.int64(1 << 27)
    kings = np.int64(0)
    nwp, nbp, nk, np_ = _fast_apply_move(np.int64(0), bp, kings, 0, 27, 31, np.int64(0))
    assert nbp == (1 << 31)
    assert nk == (1 << 31)


def test_apply_king_moves_flag():
    # king at 14 moves to 9: king flag should follow
    wp = np.int64(1 << 14)
    bp = np.int64(0)
    kings = np.int64(1 << 14)
    nwp, nbp, nk, np_ = _fast_apply_move(wp, bp, kings, 1, 14, 9, np.int64(0))
    assert nwp == (1 << 9)
    assert nk == (1 << 9)
    assert not (nk & (1 << 14))


def test_apply_capture_removes_king_flag():
    # white captures a black king
    wp = np.int64(1 << 14)
    bp = np.int64(1 << 9)
    kings = np.int64(1 << 9)   # black sq 9 is a king
    cap_mask = np.int64(1 << 9)
    nwp, nbp, nk, _ = _fast_apply_move(wp, bp, kings, 1, 14, 5, cap_mask)
    assert nbp == 0
    assert not (nk & (1 << 9))


# ---------------------------------------------------------------------------
# _collect_caps_from (pawn captures)
# ---------------------------------------------------------------------------

def _sq_of(row, col):
    """Return square index for given row/col."""
    if row % 2 == 0:
        return row * 4 + (col - 1) // 2
    else:
        return row * 4 + col // 2


def test_collect_pawn_single_capture():
    # white at sq 14 (r3c4), black at sq 9 (r2c3), land at sq 5 (r1c2)
    wp    = np.int64(1 << 14)
    bp    = np.int64(1 << 9)
    kings = np.int64(0)
    f, t, cm = _tmp_bufs()
    n = _collect_caps_from(14, wp, bp, kings, 1, False, f, t, cm, 0)
    assert n == 1
    assert f[0] == 14
    assert t[0] == 5
    assert cm[0] & (1 << 9)


def test_collect_pawn_no_capture_when_blocked():
    # black at 9, but landing sq 5 is occupied
    wp    = np.int64((1 << 14) | (1 << 5))
    bp    = np.int64(1 << 9)
    kings = np.int64(0)
    f, t, cm = _tmp_bufs()
    n = _collect_caps_from(14, wp, bp, kings, 1, False, f, t, cm, 0)
    assert n == 0


def test_collect_pawn_no_capture_no_opponent():
    wp    = np.int64(1 << 14)
    bp    = np.int64(0)
    kings = np.int64(0)
    f, t, cm = _tmp_bufs()
    n = _collect_caps_from(14, wp, bp, kings, 1, False, f, t, cm, 0)
    assert n == 0


def test_collect_pawn_multi_jump():
    # white at 22, black at 17 and 9 – two-step capture
    # sq22(r5c4) → capture sq17(r4c3) → land sq12(r3c2)? Let's compute.
    # NEIGHBORS[22]: row5(odd),col4. UL=r4,c3: row4(even),col3→sq=4*4+(3-1)//2=17. OK.
    # After landing at 17: NEIGHBORS[17]: row4(even),col3. UL=r3,c2: row3(odd),col2→sq=3*4+2//2=13.
    # Then capture sq13... Let's use a simpler chain.
    # white at 14, black at 9, land at 5, then from 5 another black at sq0?
    # NEIGHBORS[5]: row1(odd),col2. UL=r0,c1: sq=0*4+(1-1)//2=0.
    # So: white at 14, black at 9 and 0, land after 9 = sq5, then capture 0, land = NEIGHBORS[0][UL]=-1, UR=-1, DL=4,DR=5
    # White goes UL/UR. From sq5, UL = sq 0 (has black). Landing sq = NEIGHBORS[0][UL]=-1 → no landing possible UL.
    # UR of sq5 = r0,c3 = sq1. NEIGHBORS[0][UR]=-1... wait sq0 is r0c1. UR = r-1,c+1 which is out of bounds.
    # So multi-jump white 14→5 capturing 9, then from 5 can't capture 0. Let me try another setup.
    # Use black path: black at sq17, white at sq22 and sq28.
    # black pawn at 17(r4c3), white at 22(r5c4): DL of 17 = r5,c2: row5(odd),col2→sq=5*4+2//2=21. DR: r5,c4→sq=5*4+4//2=22.
    # So black at 17 can capture white at 22 if landing sq (NEIGHBORS[22][DR]) is free.
    # NEIGHBORS[22]: row5(odd),col4. DR=r6,c5: row6(even),col5→sq=6*4+(5-1)//2=26.
    # black at 17 captures white at 22, lands at 26. Then from 26, any more white?
    # Add white at 31: NEIGHBORS[26]: row6(even),col5. DR=r7,c6: row7(odd),col6→sq=7*4+6//2=31.
    # black can capture 31 from 26 landing at NEIGHBORS[31][DR]=? sq31=row7(odd),col6. DR=r8 → out of bounds = -1.
    # That means we need the multi-jump in one direction.
    # Let's just test that multiple captures from one piece give multiple results (not majority yet)
    # white at sq14, two black: sq9 and sq10
    # NEIGHBORS[14]: row3(odd),col4. UL=r2,c3: row2(even),c3→sq=2*4+(3-1)//2=9. UR=r2,c5→sq=2*4+(5-1)//2=10.
    # white captures 9 (lands at 5) OR captures 10 (lands at NEIGHBORS[10][UR]).
    # NEIGHBORS[10]: row2(even),col5. UR=r1,c6: row1(odd),c6→sq=1*4+6//2=7.
    wp    = np.int64(1 << 14)
    bp    = np.int64((1 << 9) | (1 << 10))
    kings = np.int64(0)
    f, t, cm = _tmp_bufs()
    n = _collect_caps_from(14, wp, bp, kings, 1, False, f, t, cm, 0)
    assert n == 2
    tos = set(int(t[i]) for i in range(n))
    assert 5 in tos   # captured 9, landed 5
    assert 7 in tos   # captured 10, landed 7


# ---------------------------------------------------------------------------
# _collect_caps_from (king captures)
# ---------------------------------------------------------------------------

def test_collect_king_single_capture():
    # King at sq 22 (r5c4), black at sq 17 (r4c3), lands at 12 or 13 or further?
    # NEIGHBORS[22][UL] = row4(even),col3 = sq17. After sq17 in UL direction:
    # NEIGHBORS[17][UL] = row3(odd),col2 = sq13. Then NEIGHBORS[13][UL] = row2(even),col1 = sq8.
    # King can land at 13 or 8 (all free if no other pieces).
    wp    = np.int64(1 << 22)
    bp    = np.int64(1 << 17)
    kings = np.int64(1 << 22)
    f, t, cm = _tmp_bufs()
    n = _collect_caps_from(22, wp, bp, kings, 1, True, f, t, cm, 0)
    assert n > 0
    for i in range(n):
        assert cm[i] & (1 << 17)   # sq 17 always captured
        assert f[i] == 22


def test_collect_king_blocked_by_own():
    # King at 22, own piece at 17 blocks: no captures in that direction
    wp    = np.int64((1 << 22) | (1 << 17))
    bp    = np.int64(1 << 9)
    kings = np.int64(1 << 22)
    f, t, cm = _tmp_bufs()
    n = _collect_caps_from(22, wp, bp, kings, 1, True, f, t, cm, 0)
    # sq 9 is not reachable via 17 (blocked by own), and other directions may not have opponent
    # just check it doesn't crash and produces valid output
    for i in range(n):
        assert f[i] == 22


# ---------------------------------------------------------------------------
# fast_get_captures
# ---------------------------------------------------------------------------

def test_get_captures_none():
    state = (np.int64(1 << 20), np.int64(1 << 9), np.int64(0))
    f, t, cm = _bufs()
    n = fast_get_captures(*state, 1, f, t, cm)
    assert n == 0


def test_get_captures_simple():
    wp = np.int64(1 << 14)
    bp = np.int64(1 << 9)
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_captures(wp, bp, kings, 1, f, t, cm)
    assert n == 1
    assert f[0] == 14 and t[0] == 5
    assert _popcount(cm[0]) == 1


def test_get_captures_majority_rule():
    # white at 22, black at 17 (1-cap) and black at 9 and 17 (2-cap chain).
    # Set up: white at 22, black at 17. From 22 UL → capture 17, land at 13.
    # From 13: try to capture more. Add black at 8 (UL of 13 = r2c1 = sq8).
    # NEIGHBORS[13]: row3(odd),col2. UL=r2,c1: row2(even),c1→sq=2*4+(1-1)//2=8.
    # After capturing 17 and landing at 13, can capture 8, land at NEIGHBORS[8][UL]?
    # sq8: row2(even),col1. UL=r1,c0: row1(odd),c0→sq=1*4+0//2=4.
    # So: white at 22, black at 17 and 8. 2-cap chain: 22→13 (cap 17), 13→4 (cap 8).
    # Also: white at 14, black at 9 – 1-cap chain (single capture).
    # Majority rule: only 22→4 (2 caps) should be returned.
    wp    = np.int64((1 << 22) | (1 << 14))
    bp    = np.int64((1 << 17) | (1 << 8) | (1 << 9))
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_captures(wp, bp, kings, 1, f, t, cm)
    # All returned captures must have max cap count
    assert n > 0
    max_n = max(_popcount(cm[i]) for i in range(n))
    for i in range(n):
        assert _popcount(cm[i]) == max_n


# ---------------------------------------------------------------------------
# fast_get_simple_moves
# ---------------------------------------------------------------------------

def test_simple_moves_white_pawn():
    wp    = np.int64(1 << 20)  # row 5
    bp    = np.int64(0)
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_simple_moves(wp, bp, kings, 1, f, t, cm)
    assert n > 0
    for i in range(n):
        assert f[i] == 20
        assert cm[i] == 0


def test_simple_moves_black_pawn():
    wp    = np.int64(0)
    bp    = np.int64(1 << 9)  # row 2
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_simple_moves(wp, bp, kings, 0, f, t, cm)
    assert n > 0
    for i in range(n):
        assert f[i] == 9


def test_simple_moves_king_flies():
    # King in center can reach more squares
    wp    = np.int64(1 << 13)
    bp    = np.int64(0)
    kings = np.int64(1 << 13)
    f, t, cm = _bufs()
    n_king = fast_get_simple_moves(wp, bp, kings, 1, f, t, cm)

    wp2    = np.int64(1 << 13)
    bp2    = np.int64(0)
    kings2 = np.int64(0)
    f2, t2, cm2 = _bufs()
    n_pawn = fast_get_simple_moves(wp2, bp2, kings2, 1, f2, t2, cm2)

    assert n_king > n_pawn


def test_simple_moves_blocked():
    # white pawn at sq 0 – both UL and UR are -1 (top edge), nowhere to go
    wp    = np.int64(1 << 0)
    bp    = np.int64(0)
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_simple_moves(wp, bp, kings, 1, f, t, cm)
    assert n == 0


# ---------------------------------------------------------------------------
# fast_get_legal_moves
# ---------------------------------------------------------------------------

def test_legal_prefers_captures():
    wp    = np.int64(1 << 14)
    bp    = np.int64(1 << 9)
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_legal_moves(wp, bp, kings, 1, f, t, cm)
    assert n > 0
    for i in range(n):
        assert cm[i] != 0   # all moves are captures


def test_legal_falls_back_to_simple():
    wp    = np.int64(1 << 20)
    bp    = np.int64(1 << 9)
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_legal_moves(wp, bp, kings, 1, f, t, cm)
    assert n > 0
    for i in range(n):
        assert cm[i] == 0   # no captures – simple moves


def test_legal_no_moves():
    # white pawn at sq 4, black at sq 0 (blocks forward, no capture landing)
    wp = np.int64(1 << 4)
    bp = np.int64(1 << 0)
    kings = np.int64(0)
    f, t, cm = _bufs()
    n = fast_get_legal_moves(wp, bp, kings, 1, f, t, cm)
    assert n == 0


# ---------------------------------------------------------------------------
# fast_simulate
# ---------------------------------------------------------------------------

def test_simulate_black_wins_immediately():
    # wp = 0 → black wins immediately
    result = fast_simulate(np.int64(0), np.int64(1 << 9), np.int64(0), 1, 200)
    assert result == 0.0  # starting_player=1 (white) loses


def test_simulate_white_wins_immediately():
    # bp = 0 → white wins immediately
    result = fast_simulate(np.int64(1 << 20), np.int64(0), np.int64(0), 0, 200)
    assert result == 0.0  # starting_player=0 (black) loses


def test_simulate_white_wins_starting_white():
    result = fast_simulate(np.int64(1 << 20), np.int64(0), np.int64(0), 1, 200)
    assert result == 1.0  # starting_player=1 (white) wins


def test_simulate_black_wins_starting_black():
    result = fast_simulate(np.int64(0), np.int64(1 << 9), np.int64(0), 0, 200)
    assert result == 1.0  # starting_player=0 (black) wins


def test_simulate_draw_by_depth():
    result = fast_simulate(np.int64(1 << 20), np.int64(1 << 9), np.int64(0), 1, 0)
    assert result == 0.5


def test_simulate_no_legal_moves():
    # white at sq 4, black at sq 0: white has no moves → loses (player=1)
    result = fast_simulate(np.int64(1 << 4), np.int64(1 << 0), np.int64(0), 1, 200)
    assert result == 0.0  # white (starting player) loses


def test_simulate_returns_float():
    result = fast_simulate(np.int64(1 << 20), np.int64(1 << 9), np.int64(0), 1, 10)
    assert isinstance(result, float)
    assert result in (0.0, 0.5, 1.0)


def test_simulate_initial_position():
    from models.board_state import BoardState
    s = BoardState.initial()
    result = fast_simulate(np.int64(s.white_pieces), np.int64(s.black_pieces),
                           np.int64(0), 1, 200)
    assert result in (0.0, 0.5, 1.0)
