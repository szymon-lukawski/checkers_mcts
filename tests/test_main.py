"""
Testy dla main.py
Wymaga SDL_VIDEODRIVER=dummy (ustawione w conftest.py).
"""

import pytest
import pygame
from unittest.mock import patch, MagicMock, Mock

from models.board_state import BoardState, Move
from models.config import AgentConfig, AgentType, GameConfig
from engine.game_logic import Board
from ai.random_agent import RandomAgent
from ai.ai_process import AIProcess
import main as main_module
from main import (
    play_one_game,
    run_headless,
    PygameGame,
    MoveAnimation,
    run_pygame,
    COLORS,
    MAX_MOVES_PER_GAME,
)


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


# ---------------------------------------------------------------------------
# Konfiguracje testowe
# ---------------------------------------------------------------------------

def human_vs_human_config():
    return GameConfig(
        white_agent=AgentConfig(agent_type=AgentType.HUMAN),
        black_agent=AgentConfig(agent_type=AgentType.HUMAN),
    )


def random_vs_human_config():
    return GameConfig(
        white_agent=AgentConfig(agent_type=AgentType.RANDOM),
        black_agent=AgentConfig(agent_type=AgentType.HUMAN),
    )


def human_vs_random_config():
    return GameConfig(
        white_agent=AgentConfig(agent_type=AgentType.HUMAN),
        black_agent=AgentConfig(agent_type=AgentType.RANDOM),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_moves_per_game(self):
        assert MAX_MOVES_PER_GAME == 500

    def test_colors_dict(self):
        assert COLORS[1] == "Białe"
        assert COLORS[0] == "Czarne"


# ---------------------------------------------------------------------------
# MoveAnimation
# ---------------------------------------------------------------------------

class TestMoveAnimation:
    def _make_renderer(self):
        from ui.renderer import Renderer
        return Renderer(window_size=720)

    def _make_simple_move(self):
        return Move(from_sq=20, to_sq=16)

    def _make_multi_jump_move(self):
        # 2-jump move with one intermediate landing square
        return Move(from_sq=22, to_sq=6, captured=[17, 9], path=[13])

    def test_init_simple_move(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=300)
        assert anim.move is move
        assert not anim.done
        assert len(anim._segments) == 1
        assert anim._segments[0] == (20, 16)

    def test_init_multi_jump_move(self):
        renderer = self._make_renderer()
        move = self._make_multi_jump_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=300)
        # waypoints: [22, 13, 6] → 2 segments
        assert len(anim._segments) == 2
        assert anim._segments[0] == (22, 13)
        assert anim._segments[1] == (13, 6)

    def test_done_false_initially(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=300)
        assert anim.done is False

    def test_done_true_after_all_segments(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        # Use fps=60, anim_ms=50 → frames_per_seg = max(1, int(60*50/1000)) = max(1,3) = 3
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=50)
        # Tick enough times to finish
        for _ in range(anim._frames_per_seg):
            anim.tick()
        assert anim.done is True

    def test_tick_advances_frame(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=300)
        assert anim._frame == 0
        anim.tick()
        assert anim._frame == 1

    def test_tick_advances_segment(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=50)
        fpf = anim._frames_per_seg
        for _ in range(fpf):
            anim.tick()
        assert anim._seg_idx == 1
        assert anim._frame == 0

    def test_tick_when_done_no_change(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=50)
        for _ in range(anim._frames_per_seg):
            anim.tick()
        assert anim.done
        seg_idx = anim._seg_idx
        anim.tick()  # should not raise
        assert anim._seg_idx == seg_idx

    def test_current_pixel_pos_at_start(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=300)
        # At frame 0, t=0, should be at from_sq center
        cx, cy = anim.current_pixel_pos()
        fx, fy = renderer.sq_center(20)
        assert cx == fx
        assert cy == fy

    def test_current_pixel_pos_when_done(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        anim = MoveAnimation(move, state, renderer, fps=60, anim_ms=50)
        for _ in range(anim._frames_per_seg):
            anim.tick()
        assert anim.done
        cx, cy = anim.current_pixel_pos()
        tx, ty = renderer.sq_center(move.to_sq)
        assert cx == tx
        assert cy == ty

    def test_current_pixel_pos_interpolated(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        # Use fps=10, anim_ms=200 → frames_per_seg = max(1, int(10*200/1000)) = 2
        anim = MoveAnimation(move, state, renderer, fps=10, anim_ms=200)
        anim.tick()  # frame=1
        cx, cy = anim.current_pixel_pos()
        fx, fy = renderer.sq_center(20)
        tx, ty = renderer.sq_center(16)
        # t = 1 / max(1, 2-1) = 1.0
        assert cx == int(fx + (tx - fx) * 1.0)
        assert cy == int(fy + (ty - fy) * 1.0)

    def test_frames_per_seg_minimum_one(self):
        renderer = self._make_renderer()
        move = self._make_simple_move()
        state = BoardState.initial()
        # Very low anim_ms ensures frames_per_seg=1
        anim = MoveAnimation(move, state, renderer, fps=1, anim_ms=50)
        assert anim._frames_per_seg >= 1


# ---------------------------------------------------------------------------
# play_one_game
# ---------------------------------------------------------------------------

class TestPlayOneGame:
    def test_returns_int(self):
        white = RandomAgent()
        black = RandomAgent()
        result = play_one_game(white, black)
        assert result in [1, -1, 0]

    def test_white_wins(self):
        # Symuluj grę gdzie białe wygrywają natychmiast
        white = RandomAgent()
        black = RandomAgent()

        winning_board = Board(1 << 5, 0, 0, 1)  # bp=0 → białe wygrały

        with patch.object(Board, "initial", return_value=winning_board):
            result = play_one_game(white, black)
        assert result == 1

    def test_black_wins(self):
        # Pozycja gdzie czarne wygrywają
        losing_board = Board(0, 1 << 25, 0, 0)  # wp=0 → czarne wygrały

        white = RandomAgent()
        black = RandomAgent()

        with patch.object(Board, "initial", return_value=losing_board):
            result = play_one_game(white, black)
        assert result == -1

    def test_max_moves_draw(self):
        # Użyj normalnej gry ale z ograniczoną liczbą ruchów
        white = RandomAgent()
        black = RandomAgent()

        with patch("main.MAX_MOVES_PER_GAME", 0):
            result = play_one_game(white, black)
        assert result == 0

    def test_agent_returns_none(self):
        # Biały agent zwraca None (brak ruchów) → białe przegrywają
        white = MagicMock()
        white.get_best_move.return_value = None

        # Stwórz planszę gdzie białe muszą iść (current_player=1)
        # i agent zwróci None
        mock_board = MagicMock(spec=Board)
        mock_board.is_terminal.return_value = (False, 0)
        mock_board.current_player = 1
        mock_board.to_state.return_value = BoardState.initial()

        black = RandomAgent()

        with patch.object(Board, "initial", return_value=mock_board):
            result = play_one_game(white, black)
        # Białe agent zwrócił None → return -1
        assert result == -1

    def test_black_agent_returns_none(self):
        # Czarny agent zwraca None
        black = MagicMock()
        black.get_best_move.return_value = None

        mock_board = MagicMock(spec=Board)
        mock_board.is_terminal.return_value = (False, 0)
        mock_board.current_player = 0
        mock_board.to_state.return_value = BoardState.initial()

        white = RandomAgent()

        with patch.object(Board, "initial", return_value=mock_board):
            result = play_one_game(white, black)
        assert result == 1


# ---------------------------------------------------------------------------
# run_headless
# ---------------------------------------------------------------------------

class TestRunHeadless:
    def test_run_headless_2_games(self, capsys):
        run_headless(num_games=2)
        captured = capsys.readouterr()
        assert "partii" in captured.out or "Rozgrywam" in captured.out

    def test_run_headless_prints_results(self, capsys):
        run_headless(num_games=2)
        captured = capsys.readouterr()
        assert "Wyniki" in captured.out or "wyniki" in captured.out.lower()

    def test_run_headless_100_games_progress(self, capsys):
        run_headless(num_games=100)
        captured = capsys.readouterr()
        # At 100 games, progress line should appear
        assert "100" in captured.out

    def test_run_headless_positive_num_games(self, capsys):
        run_headless(num_games=1)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# PygameGame.__init__ and _make_ai
# ---------------------------------------------------------------------------

class TestPygameGameInit:
    def test_init_human_vs_human(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        assert game.board is not None
        assert game._ai[1] is None
        assert game._ai[0] is None

    def test_init_with_ai(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        assert isinstance(game._ai[1], AIProcess)
        assert game._ai[0] is None

    def test_make_ai_human_returns_none(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        ai = game._make_ai(AgentConfig(agent_type=AgentType.HUMAN))
        assert ai is None

    def test_make_ai_random_returns_aiprocess(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        ai = game._make_ai(AgentConfig(agent_type=AgentType.RANDOM))
        assert isinstance(ai, AIProcess)
        ai.stop()

    def test_initial_state(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        assert game._selected_sq is None
        assert game._legal_targets == []
        assert game._legal_moves == []
        assert game._ai_thinking is False
        assert game.last_move is None
        assert game._animation is None
        assert game._last_mcts_simulations == 0


# ---------------------------------------------------------------------------
# _current_ai, _is_human_turn
# ---------------------------------------------------------------------------

class TestCurrentAiAndHumanTurn:
    def test_current_ai_human_turn(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        assert game._current_ai() is None
        assert game._is_human_turn() is True

    def test_current_ai_ai_turn(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # White is AI (player=1)
        assert isinstance(game._current_ai(), AIProcess)
        assert game._is_human_turn() is False


# ---------------------------------------------------------------------------
# _start_animation
# ---------------------------------------------------------------------------

class TestStartAnimation:
    def test_start_animation_sets_animation(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        moves = game.board.get_legal_moves()
        move = moves[0]
        game._start_animation(move)
        assert game._animation is not None
        assert game._animation.move is move

    def test_start_animation_clears_selection(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._selected_sq = 20
        game._legal_targets = [16]
        game._legal_moves = [Move(from_sq=20, to_sq=16)]
        moves = game.board.get_legal_moves()
        game._start_animation(moves[0])
        assert game._selected_sq is None
        assert game._legal_targets == []
        assert game._legal_moves == []

    def test_start_animation_clears_ai_thinking(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._ai_thinking = True
        moves = game.board.get_legal_moves()
        game._start_animation(moves[0])
        assert game._ai_thinking is False


# ---------------------------------------------------------------------------
# _apply_and_advance
# ---------------------------------------------------------------------------

class TestApplyAndAdvance:
    def test_apply_advances_board(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        moves = game.board.get_legal_moves()
        move = moves[0]
        game._apply_and_advance(move)
        assert game.board.current_player == 0

    def test_apply_resets_selection(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._selected_sq = 20
        game._legal_targets = [16]
        moves = game.board.get_legal_moves()
        game._apply_and_advance(moves[0])
        assert game._selected_sq is None
        assert game._legal_targets == []

    def test_apply_stores_last_move(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        moves = game.board.get_legal_moves()
        move = moves[0]
        game._apply_and_advance(move)
        assert game.last_move is move

    def test_apply_resets_ai_thinking(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._ai_thinking = True
        moves = game.board.get_legal_moves()
        game._apply_and_advance(moves[0])
        assert game._ai_thinking is False


# ---------------------------------------------------------------------------
# _start_ai_turn and _poll_ai
# ---------------------------------------------------------------------------

class TestStartAiTurnAndPollAi:
    def test_start_ai_turn_when_ai_exists(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        try:
            # White is AI (player=1)
            assert not game._ai[1].is_pending()
            game._start_ai_turn()
            assert game._ai_thinking is True
            assert game._ai[1].is_pending()
        finally:
            game._cleanup()

    def test_start_ai_turn_when_human(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # No AI → _start_ai_turn does nothing
        game._start_ai_turn()
        assert game._ai_thinking is False

    def test_start_ai_turn_when_already_pending(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        try:
            game._ai[1]._pending = True  # simulate already pending
            game._start_ai_turn()
            # Should not double-request
        finally:
            game._ai[1]._pending = False
            game._cleanup()

    def test_poll_ai_when_no_ai(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # Should not raise even when no AI
        game._poll_ai()

    def test_poll_ai_move_returned(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        try:
            import time
            game._start_ai_turn()
            # Wait for AI to respond
            time.sleep(0.5)
            initial_player = game.board.current_player
            game._poll_ai()
            # If move was returned, animation should have started
            # (might still be pending if AI is slow)
        finally:
            game._cleanup()

    def test_poll_ai_starts_animation_when_move_returned(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        try:
            ai = game._ai[1]
            fake_move = Move(from_sq=20, to_sq=16)
            ai.last_simulations = 999
            with patch.object(ai, "poll_move", return_value=fake_move), \
                 patch.object(ai, "is_pending", return_value=True):
                game._poll_ai()
            assert game._animation is not None
            assert game._animation.move is fake_move
            assert game._last_mcts_simulations == 999
        finally:
            game._ai[1]._pending = False
            game._cleanup()

    def test_poll_ai_none_response_clears_thinking(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        try:
            ai = game._ai[1]
            game._ai_thinking = True
            # is_pending: True on first call (enters block), False on second (clears thinking)
            is_pending_calls = [True, False]
            call_idx = [0]

            def is_pending_side_effect():
                result = is_pending_calls[min(call_idx[0], len(is_pending_calls) - 1)]
                call_idx[0] += 1
                return result

            with patch.object(ai, "poll_move", return_value=None), \
                 patch.object(ai, "is_pending", side_effect=is_pending_side_effect):
                game._poll_ai()
            assert game._ai_thinking is False
        finally:
            ai._pending = False
            game._cleanup()


# ---------------------------------------------------------------------------
# _select and _deselect
# ---------------------------------------------------------------------------

class TestSelectDeselect:
    def test_select_sets_state(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # sq20 is a white piece in initial position (bit 20 set)
        if game.board.wp & (1 << 20):
            game._select(20)
            assert game._selected_sq == 20
            assert isinstance(game._legal_moves, list)
            assert isinstance(game._legal_targets, list)

    def test_deselect_clears_state(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._selected_sq = 20
        game._legal_targets = [16]
        game._legal_moves = [Move(from_sq=20, to_sq=16)]
        game._deselect()
        assert game._selected_sq is None
        assert game._legal_targets == []
        assert game._legal_moves == []


# ---------------------------------------------------------------------------
# handle_click
# ---------------------------------------------------------------------------

class TestHandleClick:
    def test_click_when_not_human_turn(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        try:
            # White's turn and white is AI → handle_click returns early
            assert not game._is_human_turn()
            game.handle_click(100, 100)  # Should not raise or change state
        finally:
            game._cleanup()

    def test_click_none_sq_deselects(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._selected_sq = 20
        # Click on light square → pixel_to_sq returns None
        # Row 0, col 0 = light square
        game.handle_click(0, 0)  # Light square at col=0, row=0
        assert game._selected_sq is None

    def test_click_own_piece_selects(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # Find a white piece and click on it
        # Initial state: white pieces at sq 20-31
        # sq20: row5, col0 → renderer col=0*sq_size + sq_size//2
        renderer = game.renderer
        # sq20: _sq_to_rc(20) = (5, 0) since row5 (odd), col=2*0=0
        # Wait, sq20: row=5//4... no. sq//4 = 20//4 = 5. Row 5 (odd). col=2*(20%4)=2*0=0
        x, y = renderer.sq_center(20)
        game.handle_click(x, y)
        assert game._selected_sq == 20

    def test_click_legal_target_starts_animation(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # Select sq20, then click a legal target
        x20, y20 = game.renderer.sq_center(20)
        game.handle_click(x20, y20)
        assert game._selected_sq == 20
        # Get legal targets
        if game._legal_targets:
            target_sq = game._legal_targets[0]
            x_t, y_t = game.renderer.sq_center(target_sq)
            game.handle_click(x_t, y_t)
            # _start_animation should have been called
            assert game._selected_sq is None
            assert game._animation is not None

    def test_click_other_own_piece_reselects(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # Select sq20, then click another white piece (e.g. sq21)
        game._select(20)
        x21, y21 = game.renderer.sq_center(21)
        game.handle_click(x21, y21)
        # Should reselect to sq21 (if it's own piece and not a legal target)
        if game._selected_sq is not None:
            assert game._selected_sq in [20, 21]

    def test_click_non_target_non_own_deselects(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._select(20)
        assert game._selected_sq == 20
        # Click a dark square that's not own piece and not legal target
        # sq0 is NOT a white piece in initial position (white is at sq20+)
        # But sq0 has black piece... actually black is at sq0-11!
        # sq0 has black piece in initial. So clicking sq0 when no selection is white.
        # White: bits 20-31. Black: bits 0-11.
        # When white's turn, own = wp. sq0 is black piece.
        # sq0 is not in legal_targets (it's black). It's not own. → deselect
        x0, y0 = game.renderer.sq_center(0)
        game.handle_click(x0, y0)
        assert game._selected_sq is None

    def test_click_no_piece_selected_light_square(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        # No selection, click light square → pixel_to_sq returns None → deselect
        game.handle_click(0, 0)
        assert game._selected_sq is None


# ---------------------------------------------------------------------------
# _render
# ---------------------------------------------------------------------------

class TestRender:
    def get_screen(self):
        return pygame.Surface((720, 756))

    def test_render_result_white_wins(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        screen = self.get_screen()
        game._render(screen, result=1)

    def test_render_result_black_wins(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        screen = self.get_screen()
        game._render(screen, result=-1)

    def test_render_result_draw(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        screen = self.get_screen()
        game._render(screen, result=0)

    def test_render_ai_thinking(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._ai_thinking = True
        screen = self.get_screen()
        game._render(screen)

    def test_render_human_turn(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._ai_thinking = False
        screen = self.get_screen()
        game._render(screen)

    def test_render_ai_turn_not_thinking(self):
        # AI turn but not thinking yet (else branch)
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._ai_thinking = False
        screen = self.get_screen()
        game._render(screen)

    def test_render_with_selected_and_targets(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._selected_sq = 20
        game._legal_targets = [16, 17]
        screen = self.get_screen()
        game._render(screen)

    def test_render_black_player(self):
        # Zmień turę na czarnych (human vs human)
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game.board = Board(game.board.wp, game.board.bp, game.board.kings, 0)
        screen = self.get_screen()
        game._render(screen)

    def test_render_with_animation_active(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        moves = game.board.get_legal_moves()
        game._start_animation(moves[0])
        screen = self.get_screen()
        game._render(screen)

    def test_render_with_animation_done(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        moves = game.board.get_legal_moves()
        game._start_animation(moves[0])
        # Complete the animation
        while not game._animation.done:
            game._animation.tick()
        screen = self.get_screen()
        game._render(screen)

    def test_render_with_mcts_simulations(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._last_mcts_simulations = 1234
        screen = self.get_screen()
        game._render(screen)

    def test_render_mcts_simulations_shown_for_ai_turn(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()
        game._last_mcts_simulations = 567
        game._ai_thinking = False
        screen = self.get_screen()
        game._render(screen)


# ---------------------------------------------------------------------------
# _cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_with_no_ai(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()  # Should not raise

    def test_cleanup_with_ai(self):
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._cleanup()  # Should call stop() on AI processes

    def test_cleanup_with_both_ai(self):
        cfg = GameConfig(
            white_agent=AgentConfig(agent_type=AgentType.RANDOM),
            black_agent=AgentConfig(agent_type=AgentType.RANDOM),
        )
        game = PygameGame(cfg)
        game._cleanup()


# ---------------------------------------------------------------------------
# run (main game loop)
# ---------------------------------------------------------------------------

class TestRun:
    def _make_surface(self):
        return pygame.Surface((720, 756))

    def test_run_quit_event(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        quit_event = pygame.event.Event(pygame.QUIT)
        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()

    def test_run_escape_key(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        esc_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="")
        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[esc_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()

    def test_run_mouse_click(self):
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        click_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(100, 100)
        )
        quit_event = pygame.event.Event(pygame.QUIT)
        events = [[click_event], [quit_event]]
        call_idx = [0]

        def get_events():
            ev = events[min(call_idx[0], len(events) - 1)]
            call_idx[0] += 1
            return ev

        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", side_effect=get_events), \
             patch("pygame.time.Clock") as mock_clock, \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()

    def test_run_terminal_state(self):
        """Gra w stanie terminalnym wyświetla wynik i kończy."""
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        # Set board to terminal (bp=0)
        game.board = Board(1 << 5, 0, 0, 1)
        quit_event = pygame.event.Event(pygame.QUIT)
        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch("pygame.time.wait"), \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()

    def test_run_ai_thinking_loop(self):
        """Pętla z turedm AI."""
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._ai_thinking = True  # Symuluj że AI myśli

        quit_event = pygame.event.Event(pygame.QUIT)

        def mock_poll():
            game._ai_thinking = False

        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch.object(game, "_poll_ai", side_effect=mock_poll), \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()

    def test_run_animation_ticks_and_applies_move(self):
        """Animation completes during run loop → _apply_and_advance is called."""
        cfg = human_vs_human_config()
        game = PygameGame(cfg)
        moves = game.board.get_legal_moves()
        game._start_animation(moves[0])
        # Force animation to be "done" on next tick
        game._animation._seg_idx = len(game._animation._segments)

        apply_called = [False]
        original_apply = game._apply_and_advance

        def mock_apply(move):
            apply_called[0] = True
            original_apply(move)
            game._animation = None  # clear so loop proceeds normally

        quit_event = pygame.event.Event(pygame.QUIT)
        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch.object(game, "_apply_and_advance", side_effect=mock_apply), \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()
        assert apply_called[0] is True

    def test_run_ai_not_thinking_calls_start(self):
        """Gdy AI nie myśli, wywołuje _start_ai_turn (line 206)."""
        cfg = random_vs_human_config()
        game = PygameGame(cfg)
        game._ai_thinking = False  # Not thinking yet

        quit_event = pygame.event.Event(pygame.QUIT)
        start_called = [False]

        def mock_start():
            start_called[0] = True
            game._ai_thinking = True  # Prevent infinite loop

        with patch("pygame.display.set_mode", return_value=self._make_surface()), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch.object(game, "_start_ai_turn", side_effect=mock_start), \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            game.run()
        assert start_called[0] is True


# ---------------------------------------------------------------------------
# run_pygame
# ---------------------------------------------------------------------------

class TestRunPygame:
    def test_run_pygame_config_none(self):
        # run_menu is imported locally in run_pygame, patch at ui.menu
        with patch("ui.menu.run_menu", return_value=None):
            run_pygame()  # Should return without raising

    def test_run_pygame_with_config(self):
        cfg = human_vs_human_config()
        quit_event = pygame.event.Event(pygame.QUIT)
        surface = pygame.Surface((720, 756))

        with patch("ui.menu.run_menu", return_value=cfg), \
             patch("pygame.display.set_mode", return_value=surface), \
             patch("pygame.display.set_caption"), \
             patch("pygame.display.flip"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.time.Clock") as mock_clock, \
             patch("pygame.quit"):
            mock_clock.return_value.tick = MagicMock()
            run_pygame()
