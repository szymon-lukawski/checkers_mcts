"""
Minimax z odcinaniem Alpha-Beta.
"""

from models.board_state import BoardState, Move
from engine.game_logic import Board
from engine.move_generator import get_legal_moves, apply_move, is_game_over
from ai.base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Heurystyka oceny planszy
# ---------------------------------------------------------------------------

PAWN_VALUE  = 100
KING_VALUE  = 300

# Bonus pozycyjny dla pionków (indeks = numer pola 0-31)
# Centrum planszy jest preferowane
_CENTER_BONUS = [0] * 32
for _sq in (13, 14, 17, 18):          # środkowe 4 pola
    _CENTER_BONUS[_sq] = 15
for _sq in (9, 10, 21, 22):           # okolice centrum
    _CENTER_BONUS[_sq] = 8
for _sq in (5, 6, 25, 26):            # zewnętrzne środkowe
    _CENTER_BONUS[_sq] = 4

# Bonus za zaawansowanie (białe idą w górę → niższy indeks = dalej)
# Czarne idą w dół → wyższy indeks = dalej
_WHITE_ADV  = [0, 3, 3, 3, 3,  # rząd 0 → maksimum zaawansowania
               2, 2, 2, 2,
               1, 1, 1, 1,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0]

_BLACK_ADV  = [0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               2, 2, 2, 2,
               3, 3, 3, 3,
               3, 3, 3, 3]  # rząd 7 → maksimum zaawansowania


def _popcount(x: int) -> int:
    return bin(x).count("1")


def evaluate(wp: int, bp: int, kings: int) -> float:
    """
    Ocenia planszę z perspektywy białych (wyższy wynik = lepsza pozycja białych).
    """
    if wp == 0:
        return -30_000.0
    if bp == 0:
        return 30_000.0

    score = 0.0

    # Material
    white_kings = wp & kings
    black_kings = bp & kings
    white_pawns = wp & ~kings
    black_pawns = bp & ~kings

    score += _popcount(white_pawns) * PAWN_VALUE
    score += _popcount(white_kings) * KING_VALUE
    score -= _popcount(black_pawns) * PAWN_VALUE
    score -= _popcount(black_kings) * KING_VALUE

    # Pozycja i zaawansowanie
    bb = wp
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        score += _CENTER_BONUS[sq]
        score += _WHITE_ADV[sq]

    bb = bp
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        score -= _CENTER_BONUS[sq]
        score -= _BLACK_ADV[sq]

    return score


# ---------------------------------------------------------------------------
# Sortowanie ruchów (lepsza efektywność Alpha-Beta)
# ---------------------------------------------------------------------------

def _move_order_key(move: Move) -> int:
    # Bicia naprzód, wewnątrz bić: więcej zbitych = lepsza kolejność
    # Ruchy zwykłe: bonus za centrum pola docelowego
    if move.captured:
        return -(len(move.captured) * 1000)
    return -_CENTER_BONUS[move.to_sq]


# ---------------------------------------------------------------------------
# Minimax z Alpha-Beta
# ---------------------------------------------------------------------------

_INF = float("inf")


def _minimax(
    wp: int,
    bp: int,
    kings: int,
    current_player: int,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
) -> float:
    result = is_game_over(wp, bp, current_player, kings)
    if result != 0:
        # Wygrana/przegrana – duże wartości, głębokość premiuje szybsze zwycięstwo
        return (30_000.0 + depth) if result == 1 else -(30_000.0 + depth)

    if depth == 0:
        return evaluate(wp, bp, kings)

    moves = get_legal_moves(wp, bp, kings, current_player)
    if not moves:
        return -(30_000.0 + depth) if maximizing else (30_000.0 + depth)

    moves.sort(key=_move_order_key)

    if maximizing:
        value = -_INF
        for move in moves:
            nwp, nbp, nkings, nplayer = apply_move(wp, bp, kings, current_player, move)
            child = _minimax(nwp, nbp, nkings, nplayer, depth - 1, alpha, beta, False)
            if child > value:
                value = child
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value
    else:
        value = _INF
        for move in moves:
            nwp, nbp, nkings, nplayer = apply_move(wp, bp, kings, current_player, move)
            child = _minimax(nwp, nbp, nkings, nplayer, depth - 1, alpha, beta, True)
            if child < value:
                value = child
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MinimaxAgent(BaseAgent):
    def __init__(self, depth: int = 6) -> None:
        self.depth = depth

    def get_best_move(self, state: BoardState) -> Move | None:
        wp, bp, kings, player = state.to_tuple()
        maximizing = player == 1  # białe maksymalizują

        moves = get_legal_moves(wp, bp, kings, player)
        if not moves:
            return None

        moves.sort(key=_move_order_key)

        best_move = moves[0]
        best_val = -_INF if maximizing else _INF
        alpha, beta = -_INF, _INF

        for move in moves:
            nwp, nbp, nkings, nplayer = apply_move(wp, bp, kings, player, move)
            val = _minimax(
                nwp, nbp, nkings, nplayer,
                self.depth - 1,
                alpha, beta,
                not maximizing,
            )
            if maximizing:
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)

        return best_move
