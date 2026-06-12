"""
Ekran menu – wybór graczy i parametrów przed rozpoczęciem gry.
"""

import pygame

from ui.renderer import C_DARK, C_LIGHT, C_STATUS_BG, C_STATUS_FG
from models.config import AgentConfig, AgentType, GameConfig

# ---------------------------------------------------------------------------
# Kolory
# ---------------------------------------------------------------------------
C_BG           = (20,  20,  20)
C_PANEL_BG     = (35,  28,  20)
C_BTN_ACTIVE   = C_DARK
C_BTN_INACTIVE = (60,  50,  40)
C_BTN_HOVER    = (210, 185, 140)
C_BTN_TEXT_ON  = C_LIGHT
C_BTN_TEXT_OFF = (160, 140, 110)
C_STEP_BTN     = (80,  65,  45)
C_TRACK_FILL   = C_DARK
C_TRACK_EMPTY  = (55,  45,  35)
C_START        = (55, 120,  55)
C_START_HOVER  = (75, 155,  75)
C_PARAM_LABEL  = (170, 148, 105)
C_SEPARATOR    = (80,  65,  45)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
WIN_W        = 720
WIN_H        = 756   # 720 + 36 (pasek statusu gry)

TITLE_TOP    = 15
PANEL_Y      = 75
PANEL_H      = 570
PANEL_W      = 320
PANEL_L_X    = 20
PANEL_R_X    = 380

BTN_H        = 44
BTN_GAP      = 8
FIRST_BTN_DY = 38    # offset od górnej krawędzi panelu

STEP_BTN_W   = 32
STEP_BTN_H   = 32

START_W      = 220
START_H      = 52
START_Y      = WIN_H - 76

_AGENT_ORDER = [AgentType.HUMAN, AgentType.RANDOM, AgentType.MINIMAX, AgentType.MCTS]
_AGENT_LABELS = {
    AgentType.HUMAN:   "Człowiek",
    AgentType.RANDOM:  "Losowy",
    AgentType.MINIMAX: "Minimax",
    AgentType.MCTS:    "MCTS",
}


# ---------------------------------------------------------------------------
# Widgety pomocnicze
# ---------------------------------------------------------------------------

class _Button:
    def __init__(self, rect: pygame.Rect, label: str) -> None:
        self.rect  = rect
        self.label = label

    def contains(self, mx: int, my: int) -> bool:
        return self.rect.collidepoint(mx, my)

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        active: bool,
        hover: bool,
    ) -> None:
        if active:
            bg, fg = C_BTN_ACTIVE, C_BTN_TEXT_ON
        elif hover:
            bg, fg = C_BTN_HOVER, C_STATUS_BG
        else:
            bg, fg = C_BTN_INACTIVE, C_BTN_TEXT_OFF

        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, C_DARK, self.rect, 2, border_radius=6)
        img = font.render(self.label, True, fg)
        surface.blit(img, img.get_rect(center=self.rect.center))


class _Stepper:
    def __init__(
        self,
        x: int, y: int,
        value: int, min_val: int, max_val: int, step: int,
        total_w: int,
    ) -> None:
        self.value   = value
        self.min_val = min_val
        self.max_val = max_val
        self.step    = step

        bw = STEP_BTN_W
        gap = 6
        track_w = total_w - 2 * bw - 2 * gap
        self._r_minus = pygame.Rect(x, y, bw, STEP_BTN_H)
        self._r_track = pygame.Rect(x + bw + gap, y + 6, track_w, 20)
        self._r_plus  = pygame.Rect(x + bw + gap + track_w + gap, y, bw, STEP_BTN_H)

    def handle_click(self, mx: int, my: int) -> bool:
        if self._r_minus.collidepoint(mx, my):
            self.value = max(self.min_val, self.value - self.step)
            return True
        if self._r_plus.collidepoint(mx, my):
            self.value = min(self.max_val, self.value + self.step)
            return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        for rect, lbl in ((self._r_minus, "−"), (self._r_plus, "+")):
            pygame.draw.rect(surface, C_STEP_BTN, rect, border_radius=4)
            pygame.draw.rect(surface, C_DARK, rect, 1, border_radius=4)
            img = font.render(lbl, True, C_LIGHT)
            surface.blit(img, img.get_rect(center=rect.center))

        ratio = (self.value - self.min_val) / max(1, self.max_val - self.min_val)
        pygame.draw.rect(surface, C_TRACK_EMPTY, self._r_track, border_radius=4)
        filled = self._r_track.copy()
        filled.w = max(4, int(self._r_track.w * ratio))
        pygame.draw.rect(surface, C_TRACK_FILL, filled, border_radius=4)


# ---------------------------------------------------------------------------
# Panel gracza
# ---------------------------------------------------------------------------

class _AgentPanel:
    def __init__(self, title: str, x: int, y: int) -> None:
        self.title    = title
        self.rect     = pygame.Rect(x, y, PANEL_W, PANEL_H)
        self.selected = AgentType.HUMAN

        btn_x = x + 10
        btn_w = PANEL_W - 20
        self._agent_btns: list[tuple[AgentType, _Button]] = []
        for i, atype in enumerate(_AGENT_ORDER):
            r = pygame.Rect(
                btn_x,
                y + FIRST_BTN_DY + i * (BTN_H + BTN_GAP),
                btn_w, BTN_H,
            )
            self._agent_btns.append((atype, _Button(r, _AGENT_LABELS[atype])))

        # Parametry – pozycja pod przyciskami agentów
        param_top = y + FIRST_BTN_DY + len(_AGENT_ORDER) * (BTN_H + BTN_GAP) + 18
        self._sep_y        = param_top - 8
        self._param_lbl_y  = param_top
        self._depth_lbl_y  = param_top + 28
        self._depth_step   = _Stepper(btn_x, param_top + 50, 6, 1, 12, 1, btn_w)
        self._time_lbl_y   = param_top + 28
        self._time_step    = _Stepper(btn_x, param_top + 50, 1500, 500, 5000, 500, btn_w)

    def handle_click(self, mx: int, my: int) -> bool:
        for atype, btn in self._agent_btns:
            if btn.contains(mx, my):
                self.selected = atype
                return True
        if self.selected == AgentType.MINIMAX:
            return self._depth_step.handle_click(mx, my)
        if self.selected == AgentType.MCTS:
            return self._time_step.handle_click(mx, my)
        return False

    def draw(
        self,
        surface: pygame.Surface,
        hover_pos: tuple[int, int],
        fonts: dict,
    ) -> None:
        pygame.draw.rect(surface, C_PANEL_BG,  self.rect, border_radius=8)
        pygame.draw.rect(surface, C_DARK, self.rect, 2,  border_radius=8)

        hdr = fonts["bold"].render(self.title, True, C_LIGHT)
        surface.blit(hdr, hdr.get_rect(centerx=self.rect.centerx, top=self.rect.top + 8))

        mx, my = hover_pos
        for atype, btn in self._agent_btns:
            btn.draw(surface, fonts["normal"],
                     active=(atype == self.selected),
                     hover=btn.contains(mx, my))

        pygame.draw.line(surface, C_SEPARATOR,
                         (self.rect.left + 10, self._sep_y),
                         (self.rect.right - 10, self._sep_y), 1)

        lbl = fonts["small"].render("Parametry:", True, C_PARAM_LABEL)
        surface.blit(lbl, (self.rect.left + 10, self._param_lbl_y))

        if self.selected == AgentType.MINIMAX:
            txt = fonts["small"].render(f"Głębokość: {self._depth_step.value}",
                                        True, C_STATUS_FG)
            surface.blit(txt, (self.rect.left + 10, self._depth_lbl_y))
            self._depth_step.draw(surface, fonts["normal"])
        elif self.selected == AgentType.MCTS:
            txt = fonts["small"].render(f"Czas: {self._time_step.value} ms",
                                        True, C_STATUS_FG)
            surface.blit(txt, (self.rect.left + 10, self._time_lbl_y))
            self._time_step.draw(surface, fonts["normal"])
        else:
            msg = fonts["small"].render("Brak parametrów", True, C_PARAM_LABEL)
            surface.blit(msg, (self.rect.left + 10, self._depth_lbl_y))

    def to_agent_config(self) -> AgentConfig:
        return AgentConfig(
            agent_type=self.selected,
            minimax_depth=self._depth_step.value,
            mcts_time_ms=self._time_step.value,
        )


# ---------------------------------------------------------------------------
# Ekran menu
# ---------------------------------------------------------------------------

class MenuScreen:
    def __init__(self, window_size: int = 720) -> None:
        self.window_size  = window_size
        self._white_panel = _AgentPanel("BIAŁE",  PANEL_L_X, PANEL_Y)
        self._black_panel = _AgentPanel("CZARNE", PANEL_R_X, PANEL_Y)
        start_x = (window_size - START_W) // 2
        self._start_btn   = _Button(
            pygame.Rect(start_x, START_Y, START_W, START_H),
            "ROZPOCZNIJ GRĘ",
        )
        self._fonts: dict | None = None

    def _get_fonts(self) -> dict:
        if self._fonts is None:
            self._fonts = {
                "title":  pygame.font.SysFont("monospace", 32, bold=True),
                "bold":   pygame.font.SysFont("monospace", 18, bold=True),
                "normal": pygame.font.SysFont("monospace", 15),
                "small":  pygame.font.SysFont("monospace", 13),
            }
        return self._fonts

    def run(self, screen: pygame.Surface) -> GameConfig | None:
        clock = pygame.time.Clock()
        fonts = self._get_fonts()

        while True:
            mx, my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_RETURN:
                        return self._make_config()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._start_btn.contains(mx, my):
                        return self._make_config()
                    self._white_panel.handle_click(mx, my)
                    self._black_panel.handle_click(mx, my)

            self._draw(screen, fonts, (mx, my))
            pygame.display.flip()
            clock.tick(60)

    def _make_config(self) -> GameConfig:
        return GameConfig(
            white_agent=self._white_panel.to_agent_config(),
            black_agent=self._black_panel.to_agent_config(),
        )

    def _draw(
        self,
        screen: pygame.Surface,
        fonts: dict,
        hover_pos: tuple[int, int],
    ) -> None:
        screen.fill(C_BG)

        title = fonts["title"].render("WARCABY  MCTS", True, C_LIGHT)
        screen.blit(title, title.get_rect(
            centerx=self.window_size // 2, top=TITLE_TOP
        ))
        sub = fonts["small"].render(
            "Wybierz graczy i naciśnij START  (lub Enter)", True, C_PARAM_LABEL
        )
        screen.blit(sub, sub.get_rect(
            centerx=self.window_size // 2, top=TITLE_TOP + 42
        ))

        self._white_panel.draw(screen, hover_pos, fonts)
        self._black_panel.draw(screen, hover_pos, fonts)

        mx, my = hover_pos
        hover = self._start_btn.contains(mx, my)
        bg = C_START_HOVER if hover else C_START
        pygame.draw.rect(screen, bg, self._start_btn.rect, border_radius=10)
        pygame.draw.rect(screen, C_LIGHT, self._start_btn.rect, 2, border_radius=10)
        lbl = fonts["bold"].render(self._start_btn.label, True, C_LIGHT)
        screen.blit(lbl, lbl.get_rect(center=self._start_btn.rect.center))


# ---------------------------------------------------------------------------
# Publiczne API
# ---------------------------------------------------------------------------

def run_menu(window_size: int = 720) -> GameConfig | None:
    """
    Inicjalizuje Pygame, pokazuje menu, zwraca GameConfig lub None (zamknięcie).
    Wywołaj przed PygameGame.run() – Pygame pozostaje zainicjalizowane.
    """
    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size + 36))
    pygame.display.set_caption("Warcaby MCTS")
    return MenuScreen(window_size).run(screen)
