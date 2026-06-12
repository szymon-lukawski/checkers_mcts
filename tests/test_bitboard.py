"""
Testy dla engine/bitboard.py
"""

from engine.bitboard import (
    BOARD_MASK,
    EVEN_ROWS,
    ODD_ROWS,
    LEFT_EDGE,
    RIGHT_EDGE,
    WHITE_PROMO,
    BLACK_PROMO,
    bit,
    lsb,
    pop_lsb,
    popcount,
)


# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

class TestConstants:
    def test_board_mask_is_32_bits(self):
        assert BOARD_MASK == 0xFFFF_FFFF
        assert bin(BOARD_MASK).count("1") == 32

    def test_even_rows(self):
        assert EVEN_ROWS == 0x0F0F_0F0F
        # Pola 0-3, 8-11, 16-19, 24-27
        for sq in [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]:
            assert EVEN_ROWS & (1 << sq), f"sq={sq} powinno być w EVEN_ROWS"
        for sq in [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]:
            assert not (EVEN_ROWS & (1 << sq)), f"sq={sq} nie powinno być w EVEN_ROWS"

    def test_odd_rows(self):
        assert ODD_ROWS == 0xF0F0_F0F0
        for sq in [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]:
            assert ODD_ROWS & (1 << sq), f"sq={sq} powinno być w ODD_ROWS"
        for sq in [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]:
            assert not (ODD_ROWS & (1 << sq)), f"sq={sq} nie powinno być w ODD_ROWS"

    def test_even_odd_rows_complementary(self):
        assert (EVEN_ROWS | ODD_ROWS) == BOARD_MASK
        assert (EVEN_ROWS & ODD_ROWS) == 0

    def test_left_edge(self):
        assert LEFT_EDGE == 0x1111_1111
        # pola 0, 4, 8, 12, 16, 20, 24, 28
        for sq in [0, 4, 8, 12, 16, 20, 24, 28]:
            assert LEFT_EDGE & (1 << sq), f"sq={sq} powinno być w LEFT_EDGE"

    def test_right_edge(self):
        assert RIGHT_EDGE == 0x8888_8888
        # pola 3, 7, 11, 15, 19, 23, 27, 31
        for sq in [3, 7, 11, 15, 19, 23, 27, 31]:
            assert RIGHT_EDGE & (1 << sq), f"sq={sq} powinno być w RIGHT_EDGE"

    def test_white_promo(self):
        assert WHITE_PROMO == 0x0000_000F
        # pola 0-3
        for sq in range(4):
            assert WHITE_PROMO & (1 << sq)
        assert not (WHITE_PROMO & (1 << 4))

    def test_black_promo(self):
        assert BLACK_PROMO == 0xF000_0000
        # pola 28-31
        for sq in range(28, 32):
            assert BLACK_PROMO & (1 << sq)
        assert not (BLACK_PROMO & (1 << 27))


# ---------------------------------------------------------------------------
# bit()
# ---------------------------------------------------------------------------

class TestBit:
    def test_bit_sq0(self):
        assert bit(0) == 1

    def test_bit_sq1(self):
        assert bit(1) == 2

    def test_bit_sq31(self):
        assert bit(31) == 1 << 31

    def test_bit_sq15(self):
        assert bit(15) == 1 << 15

    def test_bit_is_power_of_two(self):
        for sq in range(32):
            b = bit(sq)
            assert b > 0
            assert (b & (b - 1)) == 0  # potęga 2


# ---------------------------------------------------------------------------
# lsb()
# ---------------------------------------------------------------------------

class TestLsb:
    def test_lsb_single_bit(self):
        for sq in range(32):
            assert lsb(1 << sq) == sq

    def test_lsb_multiple_bits_returns_lowest(self):
        bb = (1 << 5) | (1 << 10) | (1 << 20)
        assert lsb(bb) == 5

    def test_lsb_bit0_set(self):
        bb = 0b101  # bity 0 i 2
        assert lsb(bb) == 0

    def test_lsb_high_bits(self):
        assert lsb(1 << 31) == 31


# ---------------------------------------------------------------------------
# pop_lsb()
# ---------------------------------------------------------------------------

class TestPopLsb:
    def test_pop_lsb_single_bit(self):
        sq, rest = pop_lsb(1 << 7)
        assert sq == 7
        assert rest == 0

    def test_pop_lsb_multiple_bits(self):
        bb = (1 << 3) | (1 << 7) | (1 << 15)
        sq, rest = pop_lsb(bb)
        assert sq == 3
        assert rest == (1 << 7) | (1 << 15)

    def test_pop_lsb_two_bits(self):
        sq, rest = pop_lsb((1 << 0) | (1 << 31))
        assert sq == 0
        assert rest == (1 << 31)

    def test_pop_lsb_iterates_all_bits(self):
        bb = BOARD_MASK  # 32 bity
        squares = []
        while bb:
            sq, bb = pop_lsb(bb)
            squares.append(sq)
        assert squares == list(range(32))


# ---------------------------------------------------------------------------
# popcount()
# ---------------------------------------------------------------------------

class TestPopcount:
    def test_popcount_zero(self):
        assert popcount(0) == 0

    def test_popcount_one(self):
        assert popcount(1) == 1

    def test_popcount_all_32(self):
        assert popcount(BOARD_MASK) == 32

    def test_popcount_12_initial_white(self):
        # Białe startowe: bity 20-31
        white = ((1 << 12) - 1) << 20
        assert popcount(white) == 12

    def test_popcount_12_initial_black(self):
        # Czarne startowe: bity 0-11
        black = (1 << 12) - 1
        assert popcount(black) == 12

    def test_popcount_various(self):
        assert popcount(0b1010_1010) == 4
        assert popcount(0xFF) == 8
        assert popcount(1 << 15) == 1
