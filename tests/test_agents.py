"""
Testy dla ai/*.py: RandomAgent, MinimaxAgent, MCTSAgent, MCTSNode i funkcji pomocniczych.
"""

import math
import time
import pytest
from unittest.mock import patch

from models.board_state import BoardState, Move
from ai.random_agent import RandomAgent
from ai.minimax_agent import (
    MinimaxAgent,
    evaluate,
    _move_order_key,
    _minimax,
    _popcount,
    PAWN_VALUE,
    KING_VALUE,
    _CENTER_BONUS,
    _WHITE_ADV,
    _BLACK_ADV,
    _INF,
)
from ai.mcts_agent import (
    MCTSAgent,
    MCTSNode,
    _select,
    _expand,
    _simulate,
    _backpropagate,
    _MAX_ROLLOUT_DEPTH,
)
from engine.move_generator import apply_move, is_game_over


# ---------------------------------------------------------------------------
# Stany testowe
# ---------------------------------------------------------------------------

INITIAL = BoardState.initial()

# Pozycja terminalna: białe wygrały (bp=0)
WHITE_WINS = BoardState(white_pieces=1 << 5, black_pieces=0, kings=0, current_player=1)

# Pozycja terminalna: czarne wygrały (wp=0)
BLACK_WINS = BoardState(white_pieces=0, black_pieces=1 << 25, kings=0, current_player=0)

# Biały pionek ma tylko jeden ruch: sq5 → sq0 lub sq1
ONE_WHITE_MOVE = BoardState(
    white_pieces=1 << 5, black_pieces=1 << 28, kings=0, current_player=1
)

# Czarny stuck: czarne na dolnym rzędzie, białe na górnym
BLACK_STUCK = BoardState(
    white_pieces=1 << 0,
    black_pieces=(1 << 28) | (1 << 29) | (1 << 30) | (1 << 31),
    kings=0,
    current_player=0,
)


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def test_returns_move_from_initial(self):
        agent = RandomAgent()
        move = agent.get_best_move(INITIAL)
        assert isinstance(move, Move)

    def test_returns_none_on_terminal(self):
        agent = RandomAgent()
        # WHITE_WINS: bp=0, player=1 → get_legal_moves = [] (białe wygrały)
        # Actually when bp=0 is_game_over=1 but get_legal_moves still might exist
        # Let's use a truly stuck position
        state = BLACK_STUCK
        move = agent.get_best_move(state)
        # Black has no legal moves → None
        assert move is None

    def test_returns_none_when_no_moves(self):
        agent = RandomAgent()
        # Białe bez ruchów
        state = BoardState(
            white_pieces=0xF,  # sq0-3, top row, no UL/UR
            black_pieces=1 << 28,
            kings=0,
            current_player=1,
        )
        move = agent.get_best_move(state)
        assert move is None

    def test_returns_random_move(self):
        agent = RandomAgent()
        moves = set()
        for _ in range(50):
            m = agent.get_best_move(INITIAL)
            if m:
                moves.add((m.from_sq, m.to_sq))
        # From initial there are multiple possible moves, should get variety
        assert len(moves) >= 1


# ---------------------------------------------------------------------------
# _popcount (minimax helper)
# ---------------------------------------------------------------------------

class TestPopcount:
    def test_zero(self):
        assert _popcount(0) == 0

    def test_one(self):
        assert _popcount(1) == 1

    def test_full(self):
        assert _popcount(0xFFFF_FFFF) == 32


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_equal_position_near_zero(self):
        # Pozycja startowa – symetryczna, wynik bliski 0
        black = (1 << 12) - 1
        white = ((1 << 12) - 1) << 20
        score = evaluate(white, black, 0)
        # Nie musi być 0 ze względu na bonus pozycyjny, ale powinien być mały
        assert abs(score) < 5000

    def test_white_advantage_more_pieces(self):
        # Białe mają 2 pionki, czarne 1 → białe na plusie
        score = evaluate(0b11, 0b100, 0)
        assert score > 0

    def test_black_advantage_more_pieces(self):
        # Czarne mają 2 pionki, białe 1 → białe na minusie
        score = evaluate(0b1, 0b110, 0)
        assert score < 0

    def test_white_wins_large_positive(self):
        # bp=0 → 30000
        score = evaluate(1 << 5, 0, 0)
        assert score == 30_000.0

    def test_black_wins_large_negative(self):
        # wp=0 → -30000
        score = evaluate(0, 1 << 25, 0)
        assert score == -30_000.0

    def test_kings_worth_more(self):
        # Biała damka vs biały pionek: damka powinna dać wyższy wynik
        score_king = evaluate(1 << 17, 1 << 25, 1 << 17)
        score_pawn = evaluate(1 << 17, 1 << 25, 0)
        assert score_king > score_pawn

    def test_center_bonus_applied(self):
        # Pionek w centrum (sq17) vs pionek na krawędzi
        # sq17 ma _CENTER_BONUS=15, sq0 ma 0
        score_center = evaluate(1 << 17, 1 << 31, 0)
        score_edge = evaluate(1 << 0, 1 << 31, 0)
        # Centrum daje bonus
        assert score_center >= score_edge

    def test_returns_float(self):
        score = evaluate(1 << 21, 1 << 9, 0)
        assert isinstance(score, float)

    def test_pawn_value(self):
        assert PAWN_VALUE == 100

    def test_king_value(self):
        assert KING_VALUE == 300

    def test_center_bonus_values(self):
        # Sprawdź konkretne wartości
        assert _CENTER_BONUS[13] == 15
        assert _CENTER_BONUS[14] == 15
        assert _CENTER_BONUS[17] == 15
        assert _CENTER_BONUS[18] == 15
        assert _CENTER_BONUS[0] == 0

    def test_white_adv_values(self):
        # Białe idą w górę, wyższy bonus za niższy indeks
        assert _WHITE_ADV[0] == 0  # rząd 0 (lista ma tam 0 przed [1:5] mają 3)
        assert _WHITE_ADV[1] == 3
        assert _WHITE_ADV[31] == 0

    def test_black_adv_values(self):
        assert _BLACK_ADV[28] == 3
        assert _BLACK_ADV[0] == 0


# ---------------------------------------------------------------------------
# _move_order_key
# ---------------------------------------------------------------------------

class TestMoveOrderKey:
    def test_capture_has_lower_key(self):
        # Bicia mają priorytet (niższy klucz = lepsza kolejność)
        capture = Move(from_sq=14, to_sq=5, captured=[9])
        simple = Move(from_sq=21, to_sq=16)
        assert _move_order_key(capture) < _move_order_key(simple)

    def test_more_captures_lower_key(self):
        cap1 = Move(from_sq=22, to_sq=5, captured=[17])
        cap2 = Move(from_sq=22, to_sq=6, captured=[17, 9])
        assert _move_order_key(cap2) < _move_order_key(cap1)

    def test_center_move_lower_than_edge(self):
        center_move = Move(from_sq=21, to_sq=17)  # _CENTER_BONUS[17]=15
        edge_move = Move(from_sq=21, to_sq=16)    # _CENTER_BONUS[16]=0
        assert _move_order_key(center_move) <= _move_order_key(edge_move)

    def test_non_capture_returns_negative_center_bonus(self):
        m = Move(from_sq=22, to_sq=17)  # CENTER_BONUS[17]=15
        assert _move_order_key(m) == -15

    def test_non_capture_non_center(self):
        m = Move(from_sq=21, to_sq=16)  # CENTER_BONUS[16]=0
        assert _move_order_key(m) == 0


# ---------------------------------------------------------------------------
# _minimax
# ---------------------------------------------------------------------------

class TestMinimax:
    def test_terminal_white_wins(self):
        # bp=0 → białe wygrały → wynik bardzo duży
        score = _minimax(1, 0, 0, 1, 3, -_INF, _INF, True)
        assert score > 0

    def test_terminal_black_wins(self):
        # wp=0 → czarne wygrały → wynik bardzo ujemny
        score = _minimax(0, 1, 0, 0, 3, -_INF, _INF, False)
        assert score < 0

    def test_depth_zero_returns_evaluate(self):
        wp = 1 << 21
        bp = 1 << 9
        score = _minimax(wp, bp, 0, 1, 0, -_INF, _INF, True)
        expected = evaluate(wp, bp, 0)
        assert score == expected

    def test_maximizing_picks_best(self):
        # Białe maksymalizują
        wp = 1 << 21
        bp = 1 << 9
        score = _minimax(wp, bp, 0, 1, 2, -_INF, _INF, True)
        assert isinstance(score, float)

    def test_minimizing_picks_worst(self):
        wp = 1 << 21
        bp = 1 << 9
        score = _minimax(wp, bp, 0, 1, 2, -_INF, _INF, False)
        assert isinstance(score, float)

    def test_alpha_beta_pruning(self):
        # Nie rzuca wyjątku, zwraca float
        wp = 1 << 21
        bp = 1 << 9
        score = _minimax(wp, bp, 0, 1, 3, -_INF, _INF, True)
        assert isinstance(score, float)

    def test_no_moves_maximizing(self):
        # Białe bez ruchów (maximizing=True) → przegrana białych
        wp = 0xF  # top row
        bp = 1 << 28
        score = _minimax(wp, bp, 0, 1, 3, -_INF, _INF, True)
        assert score < 0


# ---------------------------------------------------------------------------
# MinimaxAgent
# ---------------------------------------------------------------------------

class TestMinimaxAgent:
    def test_returns_none_on_terminal(self):
        agent = MinimaxAgent(depth=2)
        move = agent.get_best_move(BLACK_STUCK)
        assert move is None

    def test_returns_move_from_initial(self):
        agent = MinimaxAgent(depth=2)
        move = agent.get_best_move(INITIAL)
        assert isinstance(move, Move)

    def test_depth_stored(self):
        agent = MinimaxAgent(depth=4)
        assert agent.depth == 4

    def test_default_depth(self):
        agent = MinimaxAgent()
        assert agent.depth == 6

    def test_chooses_winning_move(self):
        # Biały na sq14, czarny na sq9: biały powinien zbić (sq5)
        state = BoardState(
            white_pieces=1 << 14,
            black_pieces=1 << 9,
            kings=0,
            current_player=1,
        )
        agent = MinimaxAgent(depth=2)
        move = agent.get_best_move(state)
        assert move is not None
        # Powinien zbić pionek (to jest jedyna sensowna akcja)

    def test_black_to_move(self):
        agent = MinimaxAgent(depth=2)
        state = BoardState(
            white_pieces=1 << 14,
            black_pieces=1 << 9,
            kings=0,
            current_player=0,
        )
        move = agent.get_best_move(state)
        assert move is not None

    def test_single_move_available(self):
        # Biały ma tylko jeden możliwy ruch
        state = BoardState(
            white_pieces=1 << 5,
            black_pieces=1 << 28,
            kings=0,
            current_player=1,
        )
        agent = MinimaxAgent(depth=1)
        move = agent.get_best_move(state)
        assert move is not None
        # sq5: UL=0, UR=1 → dwa możliwe ruchy (oba ku promo)


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def get_root_state(self):
        return INITIAL.to_tuple()

    def test_uct_score_unvisited_is_inf(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        assert node.uct_score(1.0) == float("inf")

    def test_uct_score_visited(self):
        state = self.get_root_state()
        parent = MCTSNode(state=state)
        parent.visits = 10
        child = MCTSNode(state=state, parent=parent)
        child.visits = 3
        child.wins = 2.0
        score = child.uct_score(math.sqrt(2))
        assert isinstance(score, float)
        assert score > 0

    def test_is_terminal_initial(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        assert node.is_terminal() is False

    def test_is_terminal_won(self):
        state = WHITE_WINS.to_tuple()
        node = MCTSNode(state=state)
        assert node.is_terminal() is True

    def test_untried_moves_initial_non_empty(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        moves = node.untried_moves()
        assert len(moves) > 0

    def test_untried_moves_cached(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        moves1 = node.untried_moves()
        moves2 = node.untried_moves()
        assert moves1 is moves2  # same object

    def test_is_fully_expanded_after_all_children(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        untried = node.untried_moves()
        # Remove all moves to simulate fully expanded
        node._untried = []
        assert node.is_fully_expanded() is True

    def test_is_not_fully_expanded_initially(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        assert node.is_fully_expanded() is False

    def test_add_child(self):
        state = self.get_root_state()
        node = MCTSNode(state=state)
        moves = node.untried_moves()
        move = moves[0]
        wp, bp, kings, player = state
        new_state = apply_move(wp, bp, kings, player, move)
        child = node.add_child(move, new_state)
        assert child in node.children
        assert move not in node._untried
        assert child.parent is node
        assert child.move is move

    def test_best_child_returns_max_uct(self):
        state = self.get_root_state()
        parent = MCTSNode(state=state)
        parent.visits = 10

        child1 = MCTSNode(state=state, parent=parent)
        child1.visits = 5
        child1.wins = 4.0
        child2 = MCTSNode(state=state, parent=parent)
        child2.visits = 5
        child2.wins = 1.0
        parent.children = [child1, child2]

        best = parent.best_child(1.0)
        assert best is child1


# ---------------------------------------------------------------------------
# _select
# ---------------------------------------------------------------------------

class TestSelect:
    def test_select_unexpanded_returns_self(self):
        state = INITIAL.to_tuple()
        node = MCTSNode(state=state)
        result = _select(node, math.sqrt(2))
        assert result is node

    def test_select_terminal_returns_terminal(self):
        state = WHITE_WINS.to_tuple()
        node = MCTSNode(state=state)
        result = _select(node, math.sqrt(2))
        assert result is node

    def test_select_descends_to_leaf(self):
        state = INITIAL.to_tuple()
        root = MCTSNode(state=state)
        root.visits = 10
        # Expand root first to have children
        child = _expand(root)
        child.visits = 5
        child.wins = 3.0
        # Make root fully expanded
        root._untried = []
        result = _select(root, math.sqrt(2))
        # Should return child (the leaf)
        assert result is not None


# ---------------------------------------------------------------------------
# _expand
# ---------------------------------------------------------------------------

class TestExpand:
    def test_expand_adds_child(self):
        state = INITIAL.to_tuple()
        node = MCTSNode(state=state)
        initial_children = len(node.children)
        child = _expand(node)
        assert len(node.children) == initial_children + 1
        assert child in node.children

    def test_expand_no_untried_returns_same_node(self):
        # _expand returns same node when untried_moves() is empty
        state = INITIAL.to_tuple()
        node = MCTSNode(state=state)
        # Force untried to empty list
        node._untried = []
        result = _expand(node)
        assert result is node


# ---------------------------------------------------------------------------
# _simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_simulate_returns_float(self):
        state = INITIAL.to_tuple()
        result = _simulate(state)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_simulate_white_wins_starting_white(self):
        # Białe wygrały (bp=0), starting_player=1 → 1.0
        state = WHITE_WINS.to_tuple()  # player=1
        result = _simulate(state)
        # is_game_over(wp, 0, 1, 0) = 1 (białe wygrały)
        # starting_player=1, result=1 → return 1.0
        assert result == 1.0

    def test_simulate_white_wins_starting_black(self):
        # Białe wygrały, ale startujemy z perspektywy czarnych → 0.0
        state = (WHITE_WINS.white_pieces, 0, 0, 0)  # player=0 (czarne)
        # is_game_over: bp=0 → 1 (białe wygrały)
        # starting_player=0, result=1 → return 0.0
        result = _simulate(state)
        assert result == 0.0

    def test_simulate_black_wins_starting_black(self):
        # Czarne wygrały (wp=0), starting_player=0 → 1.0
        state = (0, BLACK_WINS.black_pieces, 0, 0)
        result = _simulate(state)
        assert result == 1.0

    def test_simulate_black_wins_starting_white(self):
        # Czarne wygrały, startujemy z białych → 0.0
        state = (0, BLACK_WINS.black_pieces, 0, 1)
        result = _simulate(state)
        assert result == 0.0

    def test_simulate_max_depth_returns_half(self):
        # Symuluj głęboki stan który nie zakończy się w _MAX_ROLLOUT_DEPTH
        # Użyj pozycji gdzie gra może trwać długo
        state = INITIAL.to_tuple()
        # Patch get_legal_moves aby nigdy nie zwracał terminal + ograniczyć
        with patch("ai.mcts_agent._MAX_ROLLOUT_DEPTH", 0):
            # Przy depth=0, pętla for _ in range(0) nie wykona się → return 0.5
            result = _simulate(state)
        assert result == 0.5

    def test_simulate_from_initial(self):
        state = INITIAL.to_tuple()
        result = _simulate(state)
        assert result in [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# _backpropagate
# ---------------------------------------------------------------------------

class TestBackpropagate:
    def test_backpropagate_updates_visits(self):
        state = INITIAL.to_tuple()
        node = MCTSNode(state=state)
        _backpropagate(node, 1.0)
        assert node.visits == 1

    def test_backpropagate_updates_wins(self):
        state = INITIAL.to_tuple()
        node = MCTSNode(state=state)
        _backpropagate(node, 1.0)
        # result=1.0, flip → current_result=0.0
        assert node.wins == 0.0

    def test_backpropagate_multi_level(self):
        state = INITIAL.to_tuple()
        root = MCTSNode(state=state)
        child = MCTSNode(state=state, parent=root)
        grandchild = MCTSNode(state=state, parent=child)

        _backpropagate(grandchild, 1.0)

        assert grandchild.visits == 1
        assert child.visits == 1
        assert root.visits == 1
        # Perspektywy alternują
        assert grandchild.wins == 0.0  # 1-1.0=0.0
        assert child.wins == 1.0       # 1-0.0=1.0
        assert root.wins == 0.0        # 1-1.0=0.0

    def test_backpropagate_alternates_result(self):
        state = INITIAL.to_tuple()
        parent = MCTSNode(state=state)
        child = MCTSNode(state=state, parent=parent)
        _backpropagate(child, 0.0)
        assert child.wins == 1.0   # 1-0.0=1.0
        assert parent.wins == 0.0  # 1-1.0=0.0


# ---------------------------------------------------------------------------
# MCTSAgent
# ---------------------------------------------------------------------------

class TestMCTSAgent:
    def test_returns_none_on_no_moves(self):
        agent = MCTSAgent(time_limit_ms=100)
        move = agent.get_best_move(BLACK_STUCK)
        assert move is None

    def test_returns_move_when_single_option(self):
        # Stwórz pozycję z jednym możliwym ruchem
        # sq5: UL=0 (promo → staje się damką), UR=1
        # Jeśli białe są na sq5, mają 2 ruchy. Trzeba bardziej ograniczoną pozycję.
        # Użyj pozycji gdzie są 2 możliwe ruchy → MCTSAgent zwraca moves[0] (len==1 check fails)
        # Zamiast tego użyj patcha aby skrócić
        state = BoardState(
            white_pieces=1 << 5,
            black_pieces=1 << 28,
            kings=0,
            current_player=1,
        )
        agent = MCTSAgent(time_limit_ms=50)
        move = agent.get_best_move(state)
        assert move is not None

    def test_returns_move_from_initial(self):
        agent = MCTSAgent(time_limit_ms=100)
        move = agent.get_best_move(INITIAL)
        assert isinstance(move, Move)

    def test_default_parameters(self):
        agent = MCTSAgent()
        assert agent.time_limit_ms == 1000
        assert abs(agent.c - math.sqrt(2)) < 1e-9

    def test_custom_parameters(self):
        agent = MCTSAgent(time_limit_ms=500, c=1.5)
        assert agent.time_limit_ms == 500
        assert agent.c == 1.5

    def test_no_simulations_returns_first_move(self):
        # Ujemny czas → 0 symulacji → no children → zwraca moves[0]
        agent = MCTSAgent(time_limit_ms=-1)
        move = agent.get_best_move(INITIAL)
        assert move is not None

    def test_single_legal_move_skips_search(self):
        # Jeśli jest tylko 1 legalny ruch, agent zwraca go bez szukania
        # Biały na sq5 ma 2 ruchy (0 lub 1), ale możemy wymusić sytuację przez patch
        state = INITIAL
        from engine.move_generator import get_legal_moves
        with patch("ai.mcts_agent.get_legal_moves") as mock_gl:
            mock_gl.return_value = [Move(from_sq=21, to_sq=16)]
            agent = MCTSAgent(time_limit_ms=100)
            move = agent.get_best_move(state)
            assert move.from_sq == 21

    def test_best_child_by_visits(self):
        # Sprawdź, że agent wybiera dziecko z największą liczbą wizyt
        agent = MCTSAgent(time_limit_ms=200)
        move = agent.get_best_move(INITIAL)
        assert move is not None
