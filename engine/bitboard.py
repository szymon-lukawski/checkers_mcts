"""
Stałe i maski dla 32-polowego schematu warcabów.

Schemat indeksowania (tylko ciemne pola):

   Czarny Gracz (Góra planszy)
   .  0  .  1  .  2  .  3
   4  .  5  .  6  .  7  .
   .  8  .  9  . 10  . 11
  12  . 13  . 14  . 15  .
   . 16  . 17  . 18  . 19
  20  . 21  . 22  . 23  .
   . 24  . 25  . 26  . 27
  28  . 29  . 30  . 31  .
   Biały Gracz (Dół planszy)

Rzędy parzyste (0,2,4,6): pola zajmują kolumny 1,3,5,7 (bity xxxxxx00 w ramach rzędu)
Rzędy nieparzyste (1,3,5,7): pola zajmują kolumny 0,2,4,6 (bity xxxxxx10 w ramach rzędu)

Ruchy z rzędów NIEPARZYSTYCH (1,3,5,7 → pola 4-7, 12-15, 20-23, 28-31):
  lewo-górny:  shift >> 4
  prawo-górny: shift >> 3
  lewo-dolny:  shift << 5
  prawo-dolny: shift << 4

Ruchy z rzędów PARZYSTYCH (0,2,4,6 → pola 0-3, 8-11, 16-19, 24-27):
  lewo-górny:  shift >> 5
  prawo-górny: shift >> 4
  lewo-dolny:  shift << 4
  prawo-dolny: shift << 3
"""

BOARD_MASK: int = 0xFFFF_FFFF  # 32 aktywne pola

# Maski rzędów – rzędy parzyste vs nieparzyste
# Rzędy parzyste: 0,2,4,6 → pola 0-3, 8-11, 16-19, 24-27
EVEN_ROWS: int = 0x0F0F_0F0F
# Rzędy nieparzyste: 1,3,5,7 → pola 4-7, 12-15, 20-23, 28-31
ODD_ROWS: int = 0xF0F0_F0F0

# Maski krawędzi lewej/prawej – blokują ruchy które przeskakują krawędź
# Kolumna lewa (najlewa): pola 0,4,8,12,16,20,24,28 w rzędach nieparzystych
# i pola 0,8,16,24 nie mają lewej krawędzi – ale schematy 32-polowe mają:
# W rzędach parzystych (0,2,4,6) lewe krawędzie to pola kol.1 → nie ma
# W rzędach nieparzystych (1,3,5,7) lewe krawędzie to pola kol.0
LEFT_EDGE: int = 0x1111_1111   # pola: 0, 4, 8, 12, 16, 20, 24, 28 (skrajne lewe)
RIGHT_EDGE: int = 0x8888_8888  # pola: 3, 7, 11, 15, 19, 23, 27, 31 (skrajne prawe)

# Rzędy promocji
WHITE_PROMO: int = 0x0000_000F  # pola 0-3 – białe pionki koronowane tutaj
BLACK_PROMO: int = 0xF000_0000  # pola 28-31 – czarne pionki koronowane tutaj


def bit(sq: int) -> int:
    """Maska jednego pola."""
    return 1 << sq


def lsb(bb: int) -> int:
    """Indeks najmniej znaczącego ustawionego bitu (Least Significant Bit)."""
    return (bb & -bb).bit_length() - 1


def pop_lsb(bb: int) -> tuple[int, int]:
    """Zwraca (indeks_lsb, bb_po_usunieciu_lsb)."""
    sq = lsb(bb)
    return sq, bb & (bb - 1)


def popcount(bb: int) -> int:
    """Liczba ustawionych bitów."""
    return bin(bb).count("1")
