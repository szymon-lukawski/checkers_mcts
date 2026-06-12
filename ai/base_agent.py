from abc import ABC, abstractmethod
from models.board_state import BoardState, Move


class BaseAgent(ABC):
    @abstractmethod
    def get_best_move(self, state: BoardState) -> Move | None:
        """Zwraca najlepszy ruch lub None jeśli brak ruchów (przegrana)."""
        ...
