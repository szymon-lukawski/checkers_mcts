"""
Worker multiprocessing dla agentów AI.

Komunikacja przez Queue przy użyciu słowników (model_dump/model_validate)
– bezpieczne picklowanie na macOS (spawn).
"""

import multiprocessing as mp
from models.board_state import BoardState, Move
from models.config import AgentConfig, AgentType
from ai.base_agent import BaseAgent


def _create_agent(config: AgentConfig) -> BaseAgent:
    if config.agent_type == AgentType.RANDOM:
        from ai.random_agent import RandomAgent
        return RandomAgent()
    if config.agent_type == AgentType.MINIMAX:
        from ai.minimax_agent import MinimaxAgent
        return MinimaxAgent(depth=config.minimax_depth)
    if config.agent_type == AgentType.MCTS:
        from ai.mcts_agent import MCTSAgent
        return MCTSAgent(time_limit_ms=config.mcts_time_ms)
    raise ValueError(f"Nieznany typ agenta: {config.agent_type}")


def ai_worker(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    config_dict: dict,
) -> None:
    """
    Uruchamiane w osobnym procesie. Pętla: czekaj → licz → odpowiedz.
    Kończy działanie po otrzymaniu None.
    """
    config = AgentConfig.model_validate(config_dict)
    agent = _create_agent(config)

    while True:
        raw = request_queue.get()   # blokujące – OK w oddzielnym procesie
        if raw is None:
            break
        state = BoardState.model_validate(raw)
        move = agent.get_best_move(state)
        response_queue.put(move.model_dump() if move else None)


class AIProcess:
    """
    Opakowanie procesu AI z nieblokującym API dla pętli Pygame.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._req_q: mp.Queue = mp.Queue()
        self._resp_q: mp.Queue = mp.Queue()
        self._process = mp.Process(
            target=ai_worker,
            args=(self._req_q, self._resp_q, config.model_dump()),
            daemon=True,
        )
        self._process.start()
        self._pending = False

    def request_move(self, state: BoardState) -> None:
        """Wyślij żądanie ruchu (nieblokujące)."""
        self._req_q.put(state.model_dump())
        self._pending = True

    def poll_move(self) -> Move | None:
        """
        Sprawdź czy AI odpowiedziało. Zwraca Move lub None.
        None oznacza: albo jeszcze nie odpowiedziało, albo brak ruchów.
        Użyj is_pending() żeby odróżnić te przypadki.
        """
        if not self._pending:
            return None
        try:
            raw = self._resp_q.get_nowait()
            self._pending = False
            if raw is None:
                return None
            return Move.model_validate(raw)
        except Exception:
            return None

    def is_pending(self) -> bool:
        return self._pending

    def stop(self) -> None:
        self._req_q.put(None)
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
