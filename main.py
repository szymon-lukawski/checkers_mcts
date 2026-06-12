"""
main.py – punkt startowy.
Faza 1: run_headless()  – testy konsolowe
Faza 2: run_pygame()    – UI z obsługą człowieka i AI
"""

import multiprocessing as mp
import queue as queue_module

import pygame

from engine.game_logic import Board
from models.board_state import BoardState, Move
from models.config import AgentConfig, GameConfig
from ai.random_agent import RandomAgent
from ai.ai_process import AIProcess
from ui.renderer import Renderer

# ---------------------------------------------------------------------------
# Faza 1 – tryb headless
# ---------------------------------------------------------------------------

MAX_MOVES_PER_GAME = 500


def play_one_game(white_agent: RandomAgent, black_agent: RandomAgent) -> int:
    board = Board.initial()
    for _ in range(MAX_MOVES_PER_GAME):
        terminal, result = board.is_terminal()
        if terminal:
            return result
        agent = white_agent if board.current_player == 1 else black_agent
        move = agent.get_best_move(board.to_state())
        if move is None:
            return -1 if board.current_player == 1 else 1
        board = board.apply_move(move)
    return 0


def run_headless(num_games: int = 1000) -> None:
    white = RandomAgent()
    black = RandomAgent()
    wins: dict[int, int] = {1: 0, -1: 0, 0: 0}
    print(f"Rozgrywam {num_games} partii Random vs Random...")
    for i in range(num_games):
        wins[play_one_game(white, black)] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_games}  "
                  f"Białe: {wins[1]}  Czarne: {wins[-1]}  Remisy: {wins[0]}")
    total = num_games
    print(f"\nWyniki końcowe ({total} partii):")
    print(f"  Białe wygrały:  {wins[1]:4d}  ({wins[1]/total*100:.1f}%)")
    print(f"  Czarne wygrały: {wins[-1]:4d}  ({wins[-1]/total*100:.1f}%)")
    print(f"  Remisy:         {wins[0]:4d}  ({wins[0]/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# Faza 2 – pętla Pygame
# ---------------------------------------------------------------------------

COLORS = {1: "Białe", 0: "Czarne"}


class PygameGame:
    """
    Zarządza stanem gry w UI. Obsługuje zarówno gracza-człowieka jak i AI.
    """

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.renderer = Renderer(config.window_size)
        self.board = Board.initial()
        self.last_move: Move | None = None

        # Procesy AI (None = człowiek)
        self._ai: dict[int, AIProcess | None] = {
            1: self._make_ai(config.white_agent),
            0: self._make_ai(config.black_agent),
        }

        # Stan interakcji człowieka
        self._selected_sq: int | None = None
        self._legal_targets: list[int] = []
        self._legal_moves: list[Move] = []

        # Czy AI właśnie "myśli"
        self._ai_thinking = False

    def _make_ai(self, cfg: AgentConfig) -> AIProcess | None:
        if cfg.agent_type == AgentType.HUMAN:
            return None
        return AIProcess(cfg)

    def _current_ai(self) -> AIProcess | None:
        return self._ai[self.board.current_player]

    def _is_human_turn(self) -> bool:
        return self._current_ai() is None

    # ------------------------------------------------------------------
    # Logika ruchu
    # ------------------------------------------------------------------

    def _apply_and_advance(self, move: Move) -> None:
        self.board = self.board.apply_move(move)
        self.last_move = move
        self._selected_sq = None
        self._legal_targets = []
        self._legal_moves = []
        self._ai_thinking = False

    def _start_ai_turn(self) -> None:
        ai = self._current_ai()
        if ai and not ai.is_pending():
            ai.request_move(self.board.to_state())
            self._ai_thinking = True

    def _poll_ai(self) -> None:
        ai = self._current_ai()
        if ai and ai.is_pending():
            move = ai.poll_move()
            if move is not None:
                self._apply_and_advance(move)
            elif not ai.is_pending():
                # AI odpowiedziało None → brak ruchów → koniec gry
                self._ai_thinking = False

    # ------------------------------------------------------------------
    # Obsługa kliknięć człowieka
    # ------------------------------------------------------------------

    def handle_click(self, px: int, py: int) -> None:
        if not self._is_human_turn():
            return
        sq = self.renderer.pixel_to_sq(px, py)
        if sq is None:
            self._deselect()
            return

        own = self.board.wp if self.board.current_player == 1 else self.board.bp

        if self._selected_sq is None:
            # Wybierz pionek
            if own >> sq & 1:
                self._select(sq)
        else:
            if sq in self._legal_targets:
                # Znajdź odpowiedni ruch (może być kilka ścieżek bić do tego samego pola)
                candidates = [m for m in self._legal_moves if m.to_sq == sq]
                # Wybierz tę z największą liczbą bić (zasada większości już jest w get_legal_moves)
                move = max(candidates, key=lambda m: len(m.captured))
                self._apply_and_advance(move)
            elif own >> sq & 1:
                # Kliknięto inny własny pionek
                self._select(sq)
            else:
                self._deselect()

    def _select(self, sq: int) -> None:
        self._selected_sq = sq
        all_moves = self.board.get_legal_moves()
        self._legal_moves = [m for m in all_moves if m.from_sq == sq]
        self._legal_targets = [m.to_sq for m in self._legal_moves]

    def _deselect(self) -> None:
        self._selected_sq = None
        self._legal_targets = []
        self._legal_moves = []

    # ------------------------------------------------------------------
    # Pętla główna
    # ------------------------------------------------------------------

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Warcaby MCTS")
        status_h = 36
        screen = pygame.display.set_mode(
            (self.config.window_size, self.config.window_size + status_h)
        )
        clock = pygame.time.Clock()

        running = True
        while running:
            # Zdarzenia
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)

            # Sprawdź koniec gry
            terminal, result = self.board.is_terminal()
            if terminal:
                self._render(screen, result)
                pygame.display.flip()
                pygame.time.wait(3000)
                running = False
                continue

            # Tura AI – nieblokująco
            if not self._is_human_turn():
                if not self._ai_thinking:
                    self._start_ai_turn()
                else:
                    self._poll_ai()

            # Rysuj
            self._render(screen)
            pygame.display.flip()
            clock.tick(self.config.fps)

        self._cleanup()
        pygame.quit()

    def _render(
        self,
        screen: pygame.Surface,
        result: int | None = None,
    ) -> None:
        screen.fill((20, 20, 20))
        self.renderer.draw_board(screen)
        self.renderer.draw_pieces(
            screen,
            self.board.to_state(),
            selected_sq=self._selected_sq,
            legal_targets=self._legal_targets,
            last_move=self.last_move,
        )

        if result is not None:
            winner = {1: "Białe wygrały!", -1: "Czarne wygrały!", 0: "Remis!"}
            status = winner.get(result, "Koniec gry")
        elif self._ai_thinking:
            status = f"{COLORS[self.board.current_player]} myślą..."
        elif self._is_human_turn():
            w, b = self.board.piece_count()
            status = (
                f"Ruch: {COLORS[self.board.current_player]}  |  "
                f"Białe: {w}  Czarne: {b}"
            )
        else:
            status = f"AI ({COLORS[self.board.current_player]}) myśli..."

        self.renderer.draw_status(screen, status)

    def _cleanup(self) -> None:
        for ai in self._ai.values():
            if ai:
                ai.stop()


# ---------------------------------------------------------------------------
# Punkt startowy
# ---------------------------------------------------------------------------

def run_pygame() -> None:
    from ui.menu import run_menu
    config = run_menu()
    if config is None:
        return
    PygameGame(config).run()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    run_pygame()
