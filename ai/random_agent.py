import random
from models.board_state import BoardState, Move
from engine.game_logic import Board
from ai.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def get_best_move(self, state: BoardState) -> Move | None:
        board = Board.from_state(state)
        moves = board.get_legal_moves()
        return random.choice(moves) if moves else None
