"""
engine/fast_rollout.py
======================
Numba-JIT random rollout used by MCTS.

All functions work on plain int64 scalars and pre-allocated numpy arrays –
no Python objects, no Pydantic. When numba is not installed (or mocked in
tests) the @njit decorator is a no-op and the functions run as pure Python.

Move representation inside this module:
    out_from[i]   int32  – from-square
    out_to[i]     int32  – to-square
    out_cmask[i]  int64  – bitmask of ALL captured squares (packed)

MAX_MOVES is a generous upper bound for any legal-move list in Brazilian checkers.
"""

import numpy as np

from engine._numba_compat import njit
from engine.move_generator import NEIGHBORS as _NEIGHBORS_PY

# Pre-built numpy neighbour table used inside JIT functions.
# Shape (32, 4): NEIGHBORS_NP[sq, d] = neighbour in direction d, -1 = edge.
NEIGHBORS_NP: np.ndarray = np.array(_NEIGHBORS_PY, dtype=np.int32)

# Board constants as int64 scalars (accessible inside @njit).
_BOARD_MASK  = np.int64(0xFFFF_FFFF)
_WHITE_PROMO = np.int64(0x0000_000F)   # bits 0-3
_BLACK_PROMO = np.int64(0xF000_0000)   # bits 28-31

MAX_MOVES  = 96    # generous upper bound for legal moves per position
_MAX_DEPTH = 200   # default rollout depth


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _popcount(x: np.int64) -> int:
    n = 0
    xx = np.int64(x)
    while xx:
        n += 1
        xx = xx & (xx - np.int64(1))
    return n


# ---------------------------------------------------------------------------
# Apply move
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fast_apply_move(
    wp: np.int64, bp: np.int64, kings: np.int64, player: int,
    from_sq: int, to_sq: int, cap_mask: np.int64,
):
    """Return (wp, bp, kings, next_player) after applying the move."""
    from_bit = np.int64(1) << np.int64(from_sq)
    to_bit   = np.int64(1) << np.int64(to_sq)
    is_king  = bool(from_bit & kings)

    cm = cap_mask
    if player == 1:
        wp = (wp & ~from_bit) | to_bit
        while cm:
            lsb    = cm & (-cm)
            bp    &= ~lsb
            kings &= ~lsb
            cm    &= cm - np.int64(1)
        if to_bit & _WHITE_PROMO:
            kings |= to_bit
        new_player = 0
    else:
        bp = (bp & ~from_bit) | to_bit
        while cm:
            lsb    = cm & (-cm)
            wp    &= ~lsb
            kings &= ~lsb
            cm    &= cm - np.int64(1)
        if to_bit & _BLACK_PROMO:
            kings |= to_bit
        new_player = 1

    if is_king:
        kings = (kings & ~from_bit) | to_bit

    return (wp & _BOARD_MASK, bp & _BOARD_MASK, kings & _BOARD_MASK, new_player)


# ---------------------------------------------------------------------------
# Capture generator (iterative DFS to stay numba-friendly)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _collect_caps_from(
    from_sq: int,
    wp: np.int64, bp: np.int64, kings: np.int64, player: int,
    is_king: bool,
    tmp_from, tmp_to, tmp_cmask, n_tmp: int,
) -> int:
    """
    Collect ALL capture paths starting from from_sq via iterative DFS.
    Fills tmp_from/tmp_to/tmp_cmask; returns updated n_tmp.
    No majority-rule filtering here – caller handles that.
    """
    SSIZE = 256
    stk_sq   = np.empty(SSIZE, dtype=np.int32)
    stk_mask = np.empty(SSIZE, dtype=np.int64)
    sp = 0

    stk_sq[sp]   = from_sq
    stk_mask[sp] = np.int64(0)
    sp += 1

    while sp > 0:
        sp      -= 1
        sq       = stk_sq[sp]
        cap_mask = stk_mask[sp]

        opponent = bp if player == 1 else wp
        eff_opp  = opponent & ~cap_mask
        eff_occ  = ((wp | bp) & ~cap_mask) | (np.int64(1) << np.int64(sq))

        pushed = False

        for d in range(4):
            if is_king:
                cur = sq
                while True:
                    mid = NEIGHBORS_NP[cur, d]
                    if mid < 0:
                        break
                    if (eff_occ >> np.int64(mid)) & np.int64(1):
                        if not ((eff_opp >> np.int64(mid)) & np.int64(1)):
                            break  # own piece blocks
                        new_cap = cap_mask | (np.int64(1) << np.int64(mid))
                        land = NEIGHBORS_NP[mid, d]
                        while land >= 0 and not ((eff_occ >> np.int64(land)) & np.int64(1)):
                            if sp < SSIZE:
                                stk_sq[sp]   = land
                                stk_mask[sp] = new_cap
                                sp += 1
                                pushed = True
                            land = NEIGHBORS_NP[land, d]
                        break
                    cur = mid
            else:
                mid = NEIGHBORS_NP[sq, d]
                if mid < 0:
                    continue
                if not ((eff_opp >> np.int64(mid)) & np.int64(1)):
                    continue
                land = NEIGHBORS_NP[mid, d]
                if land < 0:
                    continue
                if (eff_occ >> np.int64(land)) & np.int64(1):
                    continue
                new_cap = cap_mask | (np.int64(1) << np.int64(mid))
                if sp < SSIZE:
                    stk_sq[sp]   = land
                    stk_mask[sp] = new_cap
                    sp += 1
                    pushed = True

        if not pushed and cap_mask != np.int64(0):
            if n_tmp < len(tmp_from):
                tmp_from[n_tmp]  = from_sq
                tmp_to[n_tmp]    = sq
                tmp_cmask[n_tmp] = cap_mask
                n_tmp += 1

    return n_tmp


# ---------------------------------------------------------------------------
# Public move generators
# ---------------------------------------------------------------------------

@njit(cache=True)
def fast_get_captures(
    wp: np.int64, bp: np.int64, kings: np.int64, player: int,
    out_from, out_to, out_cmask,
) -> int:
    """
    Fill out arrays with legal captures (majority rule applied).
    Returns number of legal captures.
    """
    TMP = 512
    tmp_from  = np.empty(TMP, dtype=np.int32)
    tmp_to    = np.empty(TMP, dtype=np.int32)
    tmp_cmask = np.empty(TMP, dtype=np.int64)
    n_tmp = 0

    own = wp if player == 1 else bp
    for sq in range(32):
        if (own >> np.int64(sq)) & np.int64(1):
            is_king = bool((kings >> np.int64(sq)) & np.int64(1))
            n_tmp = _collect_caps_from(
                sq, wp, bp, kings, player, is_king,
                tmp_from, tmp_to, tmp_cmask, n_tmp,
            )

    if n_tmp == 0:
        return 0

    # Majority rule: find max capture count
    max_caps = 0
    for i in range(n_tmp):
        nc = _popcount(tmp_cmask[i])
        if nc > max_caps:
            max_caps = nc

    # Copy only max-capture paths to output
    n_out = 0
    for i in range(n_tmp):
        if _popcount(tmp_cmask[i]) == max_caps:
            if n_out < len(out_from):
                out_from[n_out]  = tmp_from[i]
                out_to[n_out]    = tmp_to[i]
                out_cmask[n_out] = tmp_cmask[i]
                n_out += 1

    return n_out


@njit(cache=True)
def fast_get_simple_moves(
    wp: np.int64, bp: np.int64, kings: np.int64, player: int,
    out_from, out_to, out_cmask,
) -> int:
    """Fill out arrays with simple (non-capture) moves. Returns count."""
    occupied = wp | bp
    n = 0
    own = wp if player == 1 else bp

    for from_sq in range(32):
        if not ((own >> np.int64(from_sq)) & np.int64(1)):
            continue
        is_king = bool((kings >> np.int64(from_sq)) & np.int64(1))

        if is_king:
            for d in range(4):
                cur = from_sq
                while True:
                    nxt = NEIGHBORS_NP[cur, d]
                    if nxt < 0:
                        break
                    if (occupied >> np.int64(nxt)) & np.int64(1):
                        break
                    if n < len(out_from):
                        out_from[n]  = from_sq
                        out_to[n]    = nxt
                        out_cmask[n] = np.int64(0)
                        n += 1
                    cur = nxt
        else:
            for d in range(4):
                # Pawns: white goes UL/UR (d=0,1), black goes DL/DR (d=2,3)
                if player == 1 and d >= 2:
                    continue
                if player == 0 and d < 2:
                    continue
                to_sq = NEIGHBORS_NP[from_sq, d]
                if to_sq >= 0 and not ((occupied >> np.int64(to_sq)) & np.int64(1)):
                    if n < len(out_from):
                        out_from[n]  = from_sq
                        out_to[n]    = to_sq
                        out_cmask[n] = np.int64(0)
                        n += 1

    return n


@njit(cache=True)
def fast_get_legal_moves(
    wp: np.int64, bp: np.int64, kings: np.int64, player: int,
    out_from, out_to, out_cmask,
) -> int:
    """Return legal moves (captures if any, else simple). Returns count."""
    n = fast_get_captures(wp, bp, kings, player, out_from, out_to, out_cmask)
    if n > 0:
        return n
    return fast_get_simple_moves(wp, bp, kings, player, out_from, out_to, out_cmask)


# ---------------------------------------------------------------------------
# Random rollout
# ---------------------------------------------------------------------------

@njit(cache=True)
def fast_simulate(
    wp: np.int64, bp: np.int64, kings: np.int64, player: int,
    max_depth: int,
) -> float:
    """
    Random rollout. Returns 1.0 if `player` (the starting player) wins,
    0.0 if they lose, 0.5 for draw (max_depth reached).
    """
    wp    = np.int64(wp)
    bp    = np.int64(bp)
    kings = np.int64(kings)
    starting_player = player

    out_from  = np.empty(MAX_MOVES, dtype=np.int32)
    out_to    = np.empty(MAX_MOVES, dtype=np.int32)
    out_cmask = np.empty(MAX_MOVES, dtype=np.int64)

    for _ in range(max_depth):
        if wp == np.int64(0):
            return 1.0 if starting_player == 0 else 0.0
        if bp == np.int64(0):
            return 1.0 if starting_player == 1 else 0.0

        n = fast_get_legal_moves(wp, bp, kings, player, out_from, out_to, out_cmask)
        if n == 0:
            # Current player has no moves → loses
            winner = 1 if player == 0 else 0
            return 1.0 if starting_player == winner else 0.0

        idx = np.random.randint(0, n)
        wp, bp, kings, player = _fast_apply_move(
            wp, bp, kings, player,
            out_from[idx], out_to[idx], out_cmask[idx],
        )

    return 0.5  # draw by max depth
