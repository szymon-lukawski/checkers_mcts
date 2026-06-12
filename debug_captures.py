"""
Skrypt diagnostyczny – uruchom go w trakcie gry żeby zobaczyć jakie bicia
są generowane dla konkretnego stanu planszy.

Użycie: zmodyfikuj wp, bp, kings poniżej i uruchom:
    uv run python3 debug_captures.py
"""

from engine.move_generator import get_captures, NEIGHBORS
from engine.bitboard import popcount

# ---- WKLEJ TUTAJ STAN Z GRY ----
# Możesz wydrukować stan pisząc tymczasowo w main.py:
#   print(f"wp={hex(board.wp)} bp={hex(board.bp)} kings={hex(board.kings)}")

wp     = 0x00000000   # <- hex bitmaska białych
bp     = 0x00000000   # <- hex bitmaska czarnych
kings  = 0x00000000   # <- hex bitmaska damek
player = 0            # 1=białe, 0=czarne

# ---- KONIEC KONFIGURACJI ----

def sq_to_rc(sq):
    row = sq // 4
    col = (1 + 2 * (sq % 4)) if row % 2 == 0 else (2 * (sq % 4))
    return row, col

def describe(sq):
    r, c = sq_to_rc(sq)
    k = " (DAMKA)" if (1 << sq) & kings else ""
    return f"sq{sq}(r{r}c{c}){k}"

print(f"Białe pionki: {[describe(i) for i in range(32) if wp >> i & 1]}")
print(f"Czarne pionki: {[describe(i) for i in range(32) if bp >> i & 1]}")
print()

caps = get_captures(wp, bp, kings, player)
print(f"Bicia dla gracza {'białego' if player==1 else 'czarnego'}: {len(caps)} opcji\n")
for m in caps:
    cap_desc = [describe(c) for c in m.captured]
    print(f"  {describe(m.from_sq)} → {describe(m.to_sq)}  zbija: {cap_desc}  ({len(m.captured)} pionki)")
