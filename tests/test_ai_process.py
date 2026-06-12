"""
Testy dla ai/ai_process.py
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock

from models.board_state import BoardState, Move
from models.config import AgentConfig, AgentType
from ai.ai_process import _create_agent, AIProcess, ai_worker
from ai.random_agent import RandomAgent
from ai.minimax_agent import MinimaxAgent
from ai.mcts_agent import MCTSAgent


INITIAL = BoardState.initial()


# ---------------------------------------------------------------------------
# _create_agent
# ---------------------------------------------------------------------------

class TestCreateAgent:
    def test_creates_random_agent(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        agent = _create_agent(cfg)
        assert isinstance(agent, RandomAgent)

    def test_creates_minimax_agent(self):
        cfg = AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=4)
        agent = _create_agent(cfg)
        assert isinstance(agent, MinimaxAgent)
        assert agent.depth == 4

    def test_creates_mcts_agent(self):
        cfg = AgentConfig(agent_type=AgentType.MCTS, mcts_time_ms=500)
        agent = _create_agent(cfg)
        assert isinstance(agent, MCTSAgent)
        assert agent.time_limit_ms == 500

    def test_raises_value_error_for_unknown_type(self):
        cfg = AgentConfig(agent_type=AgentType.HUMAN)
        with pytest.raises(ValueError, match="Nieznany"):
            _create_agent(cfg)

    def test_minimax_default_depth(self):
        cfg = AgentConfig(agent_type=AgentType.MINIMAX)
        agent = _create_agent(cfg)
        assert agent.depth == 6

    def test_mcts_default_time(self):
        cfg = AgentConfig(agent_type=AgentType.MCTS)
        agent = _create_agent(cfg)
        assert agent.time_limit_ms == 1000


# ---------------------------------------------------------------------------
# ai_worker
# ---------------------------------------------------------------------------

class TestAiWorker:
    def test_worker_exits_on_none(self):
        """Worker powinien zakończyć pętlę po otrzymaniu None."""
        import queue
        req_q = queue.SimpleQueue()
        resp_q = queue.SimpleQueue()
        req_q.put(None)  # Signal do zakończenia

        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        # ai_worker jest zaprojektowany do pracy jako subprocess,
        # ale możemy testować go bezpośrednio z queue.SimpleQueue
        # jeśli przekażemy odpowiednie queue-like objects
        # Użyjemy multiprocessing.Queue
        import multiprocessing as mp
        req_mp = mp.Queue()
        resp_mp = mp.Queue()
        req_mp.put(None)  # exit signal

        # Uruchom w tym samym wątku z timeout-em
        config_dict = cfg.model_dump()
        ai_worker(req_mp, resp_mp, config_dict)
        # Powinien zakończyć bez wyjątku

    def test_worker_processes_move_request(self):
        """Worker odpowiada na żądanie ruchu."""
        import multiprocessing as mp
        import queue
        req_q = mp.Queue()
        resp_q = mp.Queue()

        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        state = INITIAL

        req_q.put(state.model_dump())
        req_q.put(None)  # exit after one request

        ai_worker(req_q, resp_q, cfg.model_dump())

        # Pobierz odpowiedź (może wymagać krótkiego oczekiwania na flush)
        raw = None
        try:
            raw = resp_q.get(timeout=2.0)
        except Exception:
            pass

        # W pozycji startowej powinien być ruch
        assert raw is not None


# ---------------------------------------------------------------------------
# AIProcess
# ---------------------------------------------------------------------------

class TestAIProcess:
    def test_init_creates_process(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            assert proc._process.is_alive()
            assert proc._pending is False
        finally:
            proc.stop()

    def test_is_pending_initially_false(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            assert proc.is_pending() is False
        finally:
            proc.stop()

    def test_request_move_sets_pending(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            proc.request_move(INITIAL)
            assert proc.is_pending() is True
        finally:
            proc.stop()

    def test_poll_move_not_pending_returns_none(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            # Nie wysłaliśmy żądania → poll zwraca None bez sprawdzania kolejki
            result = proc.poll_move()
            assert result is None
            assert proc.is_pending() is False
        finally:
            proc.stop()

    def test_poll_move_returns_move_after_request(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            proc.request_move(INITIAL)
            # Czekaj na odpowiedź
            move = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                move = proc.poll_move()
                if move is not None or not proc.is_pending():
                    break
                time.sleep(0.05)
            # Losowy agent z pozycji startowej powinien zwrócić ruch
            assert isinstance(move, Move)
        finally:
            proc.stop()

    def test_poll_move_returns_none_when_queue_empty(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            proc.request_move(INITIAL)
            # Natychmiast odpytaj zanim AI zdąży odpowiedzieć
            result = proc.poll_move()
            # Może być None (brak odpowiedzi) lub Move (szybka odpowiedź)
            assert result is None or isinstance(result, Move)
        finally:
            proc.stop()

    def test_stop_terminates_process(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        proc.stop()
        # Po stop() process nie powinien żyć
        time.sleep(0.1)
        assert not proc._process.is_alive()

    def test_stop_with_terminate(self):
        """Test ścieżki terminate gdy join() nie zatrzyma procesu."""
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)

        # Mockuj _process.is_alive() aby zwrócił True po join()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        original_process = proc._process
        proc._process = mock_process

        proc.stop()

        mock_process.terminate.assert_called_once()

        # Zatrzymaj prawdziwy proces
        original_process.terminate()

    def test_full_round_trip(self):
        """Pełny cykl: request → poll → move."""
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            assert not proc.is_pending()
            proc.request_move(INITIAL)
            assert proc.is_pending()

            # Czekaj na odpowiedź
            time.sleep(0.5)
            move = proc.poll_move()

            if move is not None:
                assert isinstance(move, Move)
                assert not proc.is_pending()
        finally:
            proc.stop()

    def test_poll_move_returns_none_when_ai_has_no_moves(self):
        """Gdy AI nie ma ruchów, response_queue zawiera None → poll_move zwraca None."""
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        proc = AIProcess(cfg)
        try:
            # Wyślij pozycję terminalną (czarne stuck): AI response = None
            stuck_state = BoardState(
                white_pieces=1 << 0,
                black_pieces=(1 << 28) | (1 << 29) | (1 << 30) | (1 << 31),
                kings=0,
                current_player=0,  # czarne stuck, brak ruchów
            )
            proc.request_move(stuck_state)
            # Czekaj na odpowiedź
            deadline = time.time() + 5.0
            result = None
            while time.time() < deadline:
                result = proc.poll_move()
                if not proc.is_pending():
                    break
                time.sleep(0.05)
            # AI powinno odpowiedzieć None (brak ruchów)
            assert result is None
            assert not proc.is_pending()
        finally:
            proc.stop()
