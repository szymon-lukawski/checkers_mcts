"""
Generator ruchów i bić dla pionków i damek.

Używa precomputed tablicy sąsiadów dla każdego z 32 pól.
Kierunki: 0=UL, 1=UR, 2=DL, 3=DR

Schemat indeksowania:
   .  0  .  1  .  2  .  3    ← rząd 0 (parzysty): cols 1,3,5,7
   4  .  5  .  6  .  7  .    ← rząd 1 (nieparzysty): cols 0,2,4,6
   .  8  .  9  . 10  . 11    ← rząd 2 (parzysty): cols 1,3,5,7
  12  . 13  . 14  . 15  .    ...
"""

from models.board_state import Move

# ---------------------------------------------------------------------------
# Tablica sąsiadów – zbudowana raz przy imporcie
# ---------------------------------------------------------------------------

def _build_neighbors() -> list[tuple[int, int, int, int]]:
    """
    Dla każdego z 32 pól zwraca (UL, UR, DL, DR) – indeksy sąsiednich pól.
    -1 oznacza brak sąsiada (krawędź planszy).
    """
    def sq_to_rc(sq: int) -> tuple[int, int]:
        row = sq // 4
        col = (1 + 2 * (sq % 4)) if row % 2 == 0 else (2 * (sq % 4))
        return row, col

    def rc_to_sq(r: int, c: int) -> int:
        if r < 0 or r > 7 or c < 0 or c > 7:
            return -1
        if r % 2 == 0:
            if c % 2 == 0:
                return -1  # jasne pole  # pragma: no cover
            return r * 4 + (c - 1) // 2
        else:
            if c % 2 == 1:
                return -1  # jasne pole  # pragma: no cover
            return r * 4 + c // 2

    result = []
    for sq in range(32):
        r, c = sq_to_rc(sq)
        ul = rc_to_sq(r - 1, c - 1)
        ur = rc_to_sq(r - 1, c + 1)
        dl = rc_to_sq(r + 1, c - 1)
        dr = rc_to_sq(r + 1, c + 1)
        result.append((ul, ur, dl, dr))
    return result


# NEIGHBORS[sq] = (UL, UR, DL, DR), wartość -1 = krawędź planszy
NEIGHBORS: list[tuple[int, int, int, int]] = _build_neighbors()

# Kierunki dla białych (poruszają się w górę: UL i UR)
WHITE_DIRS = (0, 1)  # indeksy do NEIGHBORS: 0=UL, 1=UR
# Kierunki dla czarnych (poruszają się w dół: DL i DR)
BLACK_DIRS = (2, 3)  # 2=DL, 3=DR
# Wszystkie 4 kierunki (dla damek)
ALL_DIRS = (0, 1, 2, 3)

# ---------------------------------------------------------------------------
# Generator ruchów zwykłych (bez bić)
# ---------------------------------------------------------------------------

def get_simple_moves(
    wp: int, bp: int, kings: int, current_player: int
) -> list[Move]:
    """
    Zwraca listę zwykłych ruchów (bez bić) dla aktywnego gracza.
    Damki (brazylijskie) latają po całej przekątnej.
    """
    occupied = wp | bp
    moves: list[Move] = []
    own = wp if current_player == 1 else bp

    if current_player == 1:
        pawn_dirs = WHITE_DIRS
    else:
        pawn_dirs = BLACK_DIRS

    bb = own
    while bb:
        from_sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        is_king = bool((1 << from_sq) & kings)
        dirs = ALL_DIRS if is_king else pawn_dirs

        if is_king:
            # Damka lata wzdłuż przekątnej – iteruj "promień" w każdym kierunku
            for d in dirs:
                cur = from_sq
                while True:
                    nxt = NEIGHBORS[cur][d]
                    if nxt == -1:
                        break
                    if occupied >> nxt & 1:
                        break  # zablokowane przez pionka
                    moves.append(Move(from_sq=from_sq, to_sq=nxt))
                    cur = nxt
        else:
            for d in dirs:
                to_sq = NEIGHBORS[from_sq][d]
                if to_sq != -1 and not (occupied >> to_sq & 1):
                    moves.append(Move(from_sq=from_sq, to_sq=to_sq))

    return moves


# ---------------------------------------------------------------------------
# Generator bić (z rekurencją dla bić wielokrotnych)
# ---------------------------------------------------------------------------

def _find_captures_from(
    sq: int,
    wp: int,
    bp: int,
    kings: int,
    current_player: int,
    captured_mask: int,
    is_king: bool,
) -> list[tuple[int, list[int], list[int]]]:
    """
    Rekurencyjnie szuka bić z pola sq.
    Zwraca listę (ostateczne_pole_docelowe, lista_zbitych_pól, ścieżka_pośrednia).
    captured_mask: bitmaska już zbitych pionków (wirtualnie usunięte z planszy).
    is_king: czy bijący pionek jest damką.
    """
    opponent = bp if current_player == 1 else wp
    own = wp if current_player == 1 else bp
    effective_opponent = opponent & ~captured_mask
    # Efektywne zajęte – bez zbitych pionków, ale z naszym pionkiem na sq
    effective_occupied = ((wp | bp) & ~captured_mask) | (1 << sq)

    # Warcaby brazylijskie: pionki biją we wszystkich 4 kierunkach (również do tyłu).
    # Tylko zwykłe PRZEMIESZCZANIE jest ograniczone do kierunku do przodu.
    dirs = ALL_DIRS

    results: list[tuple[int, list[int], list[int]]] = []

    for d in dirs:
        if is_king:
            # Damka: szukaj przeciwnika wzdłuż promienia, ląduj za nim
            cur = sq
            while True:
                mid_candidate = NEIGHBORS[cur][d]
                if mid_candidate == -1:
                    break
                if effective_occupied >> mid_candidate & 1:
                    # Trafiliśmy w pionka
                    if not (effective_opponent >> mid_candidate & 1):
                        break  # własny pionek – blokuje
                    # Przeciwnik – sprawdź pola lądowania za nim
                    land = NEIGHBORS[mid_candidate][d]
                    while land != -1 and not (effective_occupied >> land & 1 and land != sq):
                        new_captured = captured_mask | (1 << mid_candidate)
                        continuations = _find_captures_from(
                            land, wp, bp, kings, current_player, new_captured, True
                        )
                        if continuations:
                            for (final_sq, cap_list, ipath) in continuations:
                                results.append((final_sq, [mid_candidate] + cap_list, [land] + ipath))
                        else:
                            results.append((land, [mid_candidate], []))
                        land = NEIGHBORS[land][d]
                    break  # po biciu nie przechodzimy przez kolejne pionki
                cur = mid_candidate
        else:
            # Zwykły pionek: jeden krok do przodu do przeciwnika, jeden za nim
            mid_sq = NEIGHBORS[sq][d]
            if mid_sq == -1:
                continue
            if not (effective_opponent >> mid_sq & 1):
                continue
            to_sq = NEIGHBORS[mid_sq][d]
            if to_sq == -1:
                continue
            if effective_occupied >> to_sq & 1 and to_sq != sq:
                continue
            new_captured = captured_mask | (1 << mid_sq)
            continuations = _find_captures_from(
                to_sq, wp, bp, kings, current_player, new_captured, False
            )
            if continuations:
                for (final_sq, cap_list, ipath) in continuations:
                    results.append((final_sq, [mid_sq] + cap_list, [to_sq] + ipath))
            else:
                results.append((to_sq, [mid_sq], []))

    return results


def get_captures(
    wp: int, bp: int, kings: int, current_player: int
) -> list[Move]:
    """
    Zwraca wszystkie możliwe bicia dla aktywnego gracza.
    Zwraca TYLKO ścieżki z maksymalną liczbą bić (zasada większości – warcaby brazylijskie).
    Obsługuje zwykłe pionki i damki.
    """
    own = wp if current_player == 1 else bp
    all_moves: list[Move] = []
    max_captures = 0

    bb = own
    while bb:
        from_sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        is_king = bool((1 << from_sq) & kings)

        paths = _find_captures_from(
            from_sq, wp, bp, kings, current_player, 0, is_king
        )
        for (to_sq, cap_list, path) in paths:
            n = len(cap_list)
            if n > max_captures:
                max_captures = n
                all_moves = []
            if n == max_captures:
                all_moves.append(Move(from_sq=from_sq, to_sq=to_sq, captured=cap_list, path=path))

    return all_moves


# ---------------------------------------------------------------------------
# Publiczny interfejs
# ---------------------------------------------------------------------------

BOARD_MASK = 0xFFFF_FFFF
WHITE_PROMO = 0x0000_000F  # pola 0-3
BLACK_PROMO = 0xF000_0000  # pola 28-31


def get_legal_moves(
    wp: int, bp: int, kings: int, current_player: int
) -> list[Move]:
    """
    Zwraca listę legalnych ruchów.
    Jeśli jest bicie – tylko bicia (zasada obowiązkowego bicia).
    Spośród bić – tylko z maksymalną liczbą zbitych (zasada większości).
    """
    captures = get_captures(wp, bp, kings, current_player)
    if captures:
        return captures
    return get_simple_moves(wp, bp, kings, current_player)


def apply_move(
    wp: int, bp: int, kings: int, current_player: int, move: Move
) -> tuple[int, int, int, int]:
    """
    Aplikuje ruch, zwraca nowy stan (wp, bp, kings, current_player).
    Obsługuje koronowanie. Nie obsługuje damek w Fazie 1.
    """
    from_bit = 1 << move.from_sq
    to_bit = 1 << move.to_sq
    is_king = bool(from_bit & kings)

    if current_player == 1:
        wp = (wp & ~from_bit) | to_bit
        for cap_sq in move.captured:
            cap_bit = 1 << cap_sq
            bp &= ~cap_bit
            kings &= ~cap_bit
        if to_bit & WHITE_PROMO:
            kings |= to_bit
        new_player = 0
    else:
        bp = (bp & ~from_bit) | to_bit
        for cap_sq in move.captured:
            cap_bit = 1 << cap_sq
            wp &= ~cap_bit
            kings &= ~cap_bit
        if to_bit & BLACK_PROMO:
            kings |= to_bit
        new_player = 1

    # Przenieś flagę damki jeśli poruszał się król
    if is_king:
        kings = (kings & ~from_bit) | to_bit

    return wp & BOARD_MASK, bp & BOARD_MASK, kings & BOARD_MASK, new_player


def is_game_over(wp: int, bp: int, current_player: int, kings: int) -> int:
    """
    Zwraca: 1 (białe wygrały), -1 (czarne wygrały), 0 (gra trwa).
    """
    if wp == 0:
        return -1
    if bp == 0:
        return 1
    if not get_legal_moves(wp, bp, kings, current_player):
        return -1 if current_player == 1 else 1
    return 0
