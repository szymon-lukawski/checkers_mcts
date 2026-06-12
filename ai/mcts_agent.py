"""
Monte Carlo Tree Search (MCTS) z UCT.

Struktura węzła: state jako krotka (wp, bp, kings, player).
Wins liczymy z perspektywy gracza, który wykonał ruch DO tego węzła.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from models.board_state import BoardState, Move
from engine.move_generator import get_legal_moves, apply_move, is_game_over
from ai.base_agent import BaseAgent

State = tuple[int, int, int, int]  # (wp, bp, kings, current_player)

_INF = float("inf")


# ---------------------------------------------------------------------------
# Węzeł drzewa
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    state: State
    parent: MCTSNode | None = None
    move: Move | None = None        # ruch, który doprowadził do tego węzła
    children: list[MCTSNode] = field(default_factory=list)
    wins: float = 0.0               # wygrane dla gracza, który WYKONAŁ ruch do tego węzła
    visits: int = 0
    _untried: list[Move] | None = None  # None = jeszcze nie pobrano

    def untried_moves(self) -> list[Move]:
        if self._untried is None:
            wp, bp, kings, player = self.state
            self._untried = get_legal_moves(wp, bp, kings, player)
        return self._untried

    def is_terminal(self) -> bool:
        wp, bp, kings, player = self.state
        return is_game_over(wp, bp, player, kings) != 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves()) == 0

    def uct_score(self, c: float) -> float:
        if self.visits == 0:
            return _INF
        assert self.parent is not None
        return (self.wins / self.visits) + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c: float) -> MCTSNode:
        return max(self.children, key=lambda n: n.uct_score(c))

    def add_child(self, move: Move, state: State) -> MCTSNode:
        child = MCTSNode(state=state, parent=self, move=move)
        self._untried.remove(move)
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# Cztery fazy MCTS
# ---------------------------------------------------------------------------

def _select(node: MCTSNode, c: float) -> MCTSNode:
    """Schodzi drzewem wybierając najlepsze dzieci (UCT) aż do liścia."""
    while not node.is_terminal():
        if not node.is_fully_expanded():
            return node
        node = node.best_child(c)
    return node


def _expand(node: MCTSNode) -> MCTSNode:
    """Dodaje jeden nieodwiedzony węzeł-dziecko."""
    untried = node.untried_moves()
    if not untried:
        return node
    move = random.choice(untried)
    wp, bp, kings, player = node.state
    new_state = apply_move(wp, bp, kings, player, move)
    return node.add_child(move, new_state)


_MAX_ROLLOUT_DEPTH = 200


def _simulate(state: State) -> float:
    """
    Losowa rozgrywka do końca lub do _MAX_ROLLOUT_DEPTH ruchów.
    Zwraca wynik z perspektywy gracza, który ruszy się JAKO PIERWSZY w tej symulacji
    (czyli gracza-właściciela węzła, z którego startujemy).
    """
    wp, bp, kings, player = state
    starting_player = player

    for _ in range(_MAX_ROLLOUT_DEPTH):
        result = is_game_over(wp, bp, player, kings)
        if result != 0:
            # result: 1=białe wygrały, -1=czarne wygrały
            if result == 1:
                return 1.0 if starting_player == 1 else 0.0
            else:
                return 1.0 if starting_player == 0 else 0.0

        moves = get_legal_moves(wp, bp, kings, player)
        if not moves:
            # Brak ruchów = przegrana aktywnego gracza
            winner = 1 if player == 0 else 0
            return 1.0 if starting_player == winner else 0.0

        move = random.choice(moves)
        wp, bp, kings, player = apply_move(wp, bp, kings, player, move)

    # Remis po wyczerpaniu głębokości
    return 0.5


def _backpropagate(node: MCTSNode, result: float) -> None:
    """
    Propaguje wynik w górę drzewa.

    _simulate zwraca 1.0 gdy wygrywa current_player WĘZŁA startowego.
    node.wins natomiast liczy wygrane gracza, który WYKONAŁ ruch DO tego węzła
    – czyli przeciwnika current_player. Dlatego zaczynamy od flipa (1 - result),
    a następnie naprzemiennie odwracamy perspektywę na każdym poziomie.
    """
    current_result = 1.0 - result   # flip: z perspektywy current_player → creator
    while node is not None:
        node.visits += 1
        node.wins += current_result
        current_result = 1.0 - current_result
        node = node.parent


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MCTSAgent(BaseAgent):
    def __init__(self, time_limit_ms: int = 1000, c: float = math.sqrt(2)) -> None:
        self.time_limit_ms = time_limit_ms
        self.c = c

    def get_best_move(self, state: BoardState) -> Move | None:
        root_state: State = state.to_tuple()
        wp, bp, kings, player = root_state

        moves = get_legal_moves(wp, bp, kings, player)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        root = MCTSNode(state=root_state)
        deadline = time.time() + self.time_limit_ms / 1000.0
        simulations = 0

        while time.time() < deadline:
            # 1. Selekcja
            node = _select(root, self.c)
            # 2. Ekspansja
            if not node.is_terminal():
                node = _expand(node)
            # 3. Symulacja
            result = _simulate(node.state)
            # 4. Propagacja
            _backpropagate(node, result)
            simulations += 1

        if not root.children:
            return moves[0]

        # Wybierz ruch z największą liczbą odwiedzin (najbardziej zbadany)
        best = max(root.children, key=lambda n: n.visits)
        return best.move
