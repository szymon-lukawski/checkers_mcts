"""
Klasa Board – cienka nakładka OOP na silnik bitboardowy.
"""

from models.board_state import BoardState, Move
from engine.move_generator import (
    get_legal_moves,
    apply_move,
    is_game_over,
)


class Board:
    __slots__ = ("wp", "bp", "kings", "current_player")

    def __init__(self, wp: int, bp: int, kings: int, current_player: int) -> None:
        self.wp = wp
        self.bp = bp
        self.kings = kings
        self.current_player = current_player

    @classmethod
    def from_state(cls, state: BoardState) -> "Board":
        return cls(*state.to_tuple())

    @classmethod
    def initial(cls) -> "Board":
        return cls.from_state(BoardState.initial())

    def to_state(self) -> BoardState:
        return BoardState(
            white_pieces=self.wp,
            black_pieces=self.bp,
            kings=self.kings,
            current_player=self.current_player,
        )

    def get_legal_moves(self) -> list[Move]:
        return get_legal_moves(self.wp, self.bp, self.kings, self.current_player)

    def apply_move(self, move: Move) -> "Board":
        wp, bp, kings, player = apply_move(
            self.wp, self.bp, self.kings, self.current_player, move
        )
        return Board(wp, bp, kings, player)

    def is_terminal(self) -> tuple[bool, int]:
        result = is_game_over(self.wp, self.bp, self.current_player, self.kings)
        return result != 0, result

    def piece_count(self) -> tuple[int, int]:
        """Zwraca (liczba_białych, liczba_czarnych)."""
        return bin(self.wp).count("1"), bin(self.bp).count("1")

    def __repr__(self) -> str:
        w, b = self.piece_count()
        player = "białe" if self.current_player == 1 else "czarne"
        return f"Board(W={w}, B={b}, ruch={player})"
