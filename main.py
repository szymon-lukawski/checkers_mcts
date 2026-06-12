"""
main.py – punkt startowy. Faza 1: tryb headless (konsola).
"""

import multiprocessing as mp
from engine.game_logic import Board
from ai.random_agent import RandomAgent
from models.board_state import BoardState


MAX_MOVES_PER_GAME = 500  # zabezpieczenie przed nieskończoną partią


def play_one_game(white_agent: RandomAgent, black_agent: RandomAgent) -> int:
    """
    Rozgrywa jedną partię. Zwraca wynik: 1 (białe), -1 (czarne), 0 (remis).
    """
    board = Board.initial()

    for _ in range(MAX_MOVES_PER_GAME):
        terminal, result = board.is_terminal()
        if terminal:
            return result

        state = board.to_state()
        agent = white_agent if board.current_player == 1 else black_agent
        move = agent.get_best_move(state)

        if move is None:
            return -1 if board.current_player == 1 else 1

        board = board.apply_move(move)

    return 0  # remis po limicie ruchów


def run_headless(num_games: int = 1000) -> None:
    white = RandomAgent()
    black = RandomAgent()

    wins = {1: 0, -1: 0, 0: 0}

    print(f"Rozgrywam {num_games} partii Random vs Random...")
    for i in range(num_games):
        result = play_one_game(white, black)
        wins[result] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_games}  "
                  f"Białe: {wins[1]}  Czarne: {wins[-1]}  Remisy: {wins[0]}")

    total = num_games
    print(f"\nWyniki końcowe ({total} partii):")
    print(f"  Białe wygrały:  {wins[1]:4d}  ({wins[1]/total*100:.1f}%)")
    print(f"  Czarne wygrały: {wins[-1]:4d}  ({wins[-1]/total*100:.1f}%)")
    print(f"  Remisy:         {wins[0]:4d}  ({wins[0]/total*100:.1f}%)")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    run_headless(1000)
