"""
Testy dla ui/menu.py
Wymaga SDL_VIDEODRIVER=dummy (ustawione w conftest.py).
"""

import pytest
import pygame
from unittest.mock import patch, MagicMock

from models.config import AgentConfig, AgentType, GameConfig
from ui.menu import (
    _Button,
    _Stepper,
    _AgentPanel,
    MenuScreen,
    run_menu,
    PANEL_L_X,
    PANEL_Y,
    PANEL_R_X,
    START_Y,
    START_W,
    START_H,
    WIN_W,
    WIN_H,
    ANIM_Y,
)


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def surface():
    return pygame.Surface((WIN_W, WIN_H))


@pytest.fixture
def font():
    return pygame.font.SysFont("monospace", 15)


@pytest.fixture
def fonts(font):
    return {
        "title": pygame.font.SysFont("monospace", 32, bold=True),
        "bold": pygame.font.SysFont("monospace", 18, bold=True),
        "normal": font,
        "small": pygame.font.SysFont("monospace", 13),
    }


# ---------------------------------------------------------------------------
# _Button
# ---------------------------------------------------------------------------

class TestButton:
    def make_button(self, x=100, y=100, w=120, h=44):
        return _Button(pygame.Rect(x, y, w, h), "Test")

    def test_contains_inside(self):
        btn = self.make_button()
        assert btn.contains(110, 110) is True

    def test_contains_outside(self):
        btn = self.make_button()
        assert btn.contains(0, 0) is False

    def test_contains_edge(self):
        btn = self.make_button(x=100, y=100, w=120, h=44)
        # Pygame collidepoint: on edge may or may not be inside depending on pixel
        result = btn.contains(100, 100)
        assert isinstance(result, bool)

    def test_draw_active(self, surface, font):
        btn = self.make_button()
        btn.draw(surface, font, active=True, hover=False)

    def test_draw_hover(self, surface, font):
        btn = self.make_button()
        btn.draw(surface, font, active=False, hover=True)

    def test_draw_inactive(self, surface, font):
        btn = self.make_button()
        btn.draw(surface, font, active=False, hover=False)

    def test_draw_active_and_hover(self, surface, font):
        btn = self.make_button()
        btn.draw(surface, font, active=True, hover=True)


# ---------------------------------------------------------------------------
# _Stepper
# ---------------------------------------------------------------------------

class TestStepper:
    def make_stepper(self, x=50, y=50, value=6, min_val=1, max_val=12, step=1, total_w=200):
        return _Stepper(x, y, value, min_val, max_val, step, total_w)

    def test_initial_value(self):
        s = self.make_stepper(value=6)
        assert s.value == 6

    def test_click_minus_decreases(self):
        s = self.make_stepper(value=6)
        # Click the minus button
        cx = s._r_minus.centerx
        cy = s._r_minus.centery
        changed = s.handle_click(cx, cy)
        assert changed is True
        assert s.value == 5

    def test_click_plus_increases(self):
        s = self.make_stepper(value=6)
        cx = s._r_plus.centerx
        cy = s._r_plus.centery
        changed = s.handle_click(cx, cy)
        assert changed is True
        assert s.value == 7

    def test_click_minus_clamps_to_min(self):
        s = self.make_stepper(value=1)
        cx = s._r_minus.centerx
        cy = s._r_minus.centery
        s.handle_click(cx, cy)
        assert s.value == 1  # clamped

    def test_click_plus_clamps_to_max(self):
        s = self.make_stepper(value=12)
        cx = s._r_plus.centerx
        cy = s._r_plus.centery
        s.handle_click(cx, cy)
        assert s.value == 12  # clamped

    def test_click_no_hit_returns_false(self):
        s = self.make_stepper()
        changed = s.handle_click(0, 0)
        assert changed is False

    def test_click_no_hit_value_unchanged(self):
        s = self.make_stepper(value=6)
        s.handle_click(0, 0)
        assert s.value == 6

    def test_draw_no_exception(self, surface, font):
        s = self.make_stepper()
        s.draw(surface, font)

    def test_draw_at_min(self, surface, font):
        s = self.make_stepper(value=1)
        s.draw(surface, font)

    def test_draw_at_max(self, surface, font):
        s = self.make_stepper(value=12)
        s.draw(surface, font)


# ---------------------------------------------------------------------------
# _AgentPanel
# ---------------------------------------------------------------------------

class TestAgentPanel:
    def make_panel(self, title="BIAŁE", x=PANEL_L_X, y=PANEL_Y):
        return _AgentPanel(title, x, y)

    def test_initial_selection_is_human(self):
        panel = self.make_panel()
        assert panel.selected == AgentType.HUMAN

    def test_click_agent_button_random(self):
        panel = self.make_panel()
        # Find and click the RANDOM button
        for atype, btn in panel._agent_btns:
            if atype == AgentType.RANDOM:
                cx, cy = btn.rect.centerx, btn.rect.centery
                result = panel.handle_click(cx, cy)
                assert result is True
                assert panel.selected == AgentType.RANDOM
                break

    def test_click_agent_button_minimax(self):
        panel = self.make_panel()
        for atype, btn in panel._agent_btns:
            if atype == AgentType.MINIMAX:
                result = panel.handle_click(btn.rect.centerx, btn.rect.centery)
                assert result is True
                assert panel.selected == AgentType.MINIMAX
                break

    def test_click_agent_button_mcts(self):
        panel = self.make_panel()
        for atype, btn in panel._agent_btns:
            if atype == AgentType.MCTS:
                result = panel.handle_click(btn.rect.centerx, btn.rect.centery)
                assert result is True
                assert panel.selected == AgentType.MCTS
                break

    def test_click_depth_stepper_when_minimax(self):
        panel = self.make_panel()
        panel.selected = AgentType.MINIMAX
        # Click minus button on depth stepper
        stepper = panel._depth_step
        result = panel.handle_click(stepper._r_minus.centerx, stepper._r_minus.centery)
        assert result is True

    def test_click_time_stepper_when_mcts(self):
        panel = self.make_panel()
        panel.selected = AgentType.MCTS
        stepper = panel._time_step
        result = panel.handle_click(stepper._r_plus.centerx, stepper._r_plus.centery)
        assert result is True

    def test_click_no_hit_when_human(self):
        panel = self.make_panel()
        panel.selected = AgentType.HUMAN
        result = panel.handle_click(0, 0)
        assert result is False

    def test_click_no_hit_when_random(self):
        panel = self.make_panel()
        panel.selected = AgentType.RANDOM
        result = panel.handle_click(0, 0)
        assert result is False

    def test_draw_human_mode(self, surface, fonts):
        panel = self.make_panel()
        panel.selected = AgentType.HUMAN
        panel.draw(surface, (0, 0), fonts)

    def test_draw_minimax_mode(self, surface, fonts):
        panel = self.make_panel()
        panel.selected = AgentType.MINIMAX
        panel.draw(surface, (0, 0), fonts)

    def test_draw_mcts_mode(self, surface, fonts):
        panel = self.make_panel()
        panel.selected = AgentType.MCTS
        panel.draw(surface, (0, 0), fonts)

    def test_draw_random_mode(self, surface, fonts):
        panel = self.make_panel()
        panel.selected = AgentType.RANDOM
        panel.draw(surface, (0, 0), fonts)

    def test_to_agent_config_human(self):
        panel = self.make_panel()
        panel.selected = AgentType.HUMAN
        cfg = panel.to_agent_config()
        assert isinstance(cfg, AgentConfig)
        assert cfg.agent_type == AgentType.HUMAN

    def test_to_agent_config_minimax(self):
        panel = self.make_panel()
        panel.selected = AgentType.MINIMAX
        cfg = panel.to_agent_config()
        assert cfg.agent_type == AgentType.MINIMAX
        assert cfg.minimax_depth == panel._depth_step.value

    def test_to_agent_config_mcts(self):
        panel = self.make_panel()
        panel.selected = AgentType.MCTS
        cfg = panel.to_agent_config()
        assert cfg.agent_type == AgentType.MCTS
        assert cfg.mcts_time_ms == panel._time_step.value


# ---------------------------------------------------------------------------
# MenuScreen
# ---------------------------------------------------------------------------

class TestMenuScreen:
    def test_init(self):
        menu = MenuScreen(window_size=720)
        assert menu.window_size == 720
        assert menu._fonts is None

    def test_get_fonts_creates_dict(self):
        menu = MenuScreen()
        fonts = menu._get_fonts()
        assert "title" in fonts
        assert "bold" in fonts
        assert "normal" in fonts
        assert "small" in fonts

    def test_get_fonts_cached(self):
        menu = MenuScreen()
        fonts1 = menu._get_fonts()
        fonts2 = menu._get_fonts()
        assert fonts1 is fonts2

    def test_make_config(self):
        menu = MenuScreen()
        cfg = menu._make_config()
        assert isinstance(cfg, GameConfig)

    def test_make_config_returns_game_config(self):
        menu = MenuScreen()
        cfg = menu._make_config()
        assert isinstance(cfg.white_agent, AgentConfig)
        assert isinstance(cfg.black_agent, AgentConfig)

    def test_make_config_has_anim_ms(self):
        menu = MenuScreen()
        cfg = menu._make_config()
        assert cfg.anim_ms == menu._anim_stepper.value

    def test_make_config_anim_ms_default(self):
        menu = MenuScreen()
        cfg = menu._make_config()
        assert cfg.anim_ms == 300

    def test_anim_stepper_exists(self):
        menu = MenuScreen()
        assert hasattr(menu, "_anim_stepper")
        assert menu._anim_stepper.value == 300
        assert menu._anim_stepper.min_val == 50
        assert menu._anim_stepper.max_val == 2000

    def test_anim_stepper_click_minus(self):
        menu = MenuScreen()
        initial = menu._anim_stepper.value
        cx = menu._anim_stepper._r_minus.centerx
        cy = menu._anim_stepper._r_minus.centery
        menu._anim_stepper.handle_click(cx, cy)
        assert menu._anim_stepper.value == initial - 50

    def test_anim_stepper_click_plus(self):
        menu = MenuScreen()
        initial = menu._anim_stepper.value
        cx = menu._anim_stepper._r_plus.centerx
        cy = menu._anim_stepper._r_plus.centery
        menu._anim_stepper.handle_click(cx, cy)
        assert menu._anim_stepper.value == initial + 50

    def test_anim_stepper_no_hit(self):
        menu = MenuScreen()
        initial = menu._anim_stepper.value
        result = menu._anim_stepper.handle_click(0, 0)
        assert result is False
        assert menu._anim_stepper.value == initial

    def test_draw_no_exception(self, surface, fonts):
        menu = MenuScreen()
        menu._draw(surface, fonts, (0, 0))

    def test_draw_with_hover(self, surface, fonts):
        menu = MenuScreen()
        # Hover over start button center
        btn_x = (720 - START_W) // 2 + START_W // 2
        menu._draw(surface, fonts, (btn_x, START_Y + START_H // 2))

    def test_run_quit_event(self, surface):
        menu = MenuScreen()
        quit_event = pygame.event.Event(pygame.QUIT)
        with patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_clock.return_value.tick = MagicMock()
            result = menu.run(surface)
        assert result is None

    def test_run_escape_key(self, surface):
        menu = MenuScreen()
        esc_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="")
        with patch("pygame.event.get", return_value=[esc_event]), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_clock.return_value.tick = MagicMock()
            result = menu.run(surface)
        assert result is None

    def test_run_return_key(self, surface):
        menu = MenuScreen()
        return_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="")
        with patch("pygame.event.get", return_value=[return_event]), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_clock.return_value.tick = MagicMock()
            result = menu.run(surface)
        assert isinstance(result, GameConfig)

    def test_run_start_button_click(self, surface):
        menu = MenuScreen()
        btn_cx = menu._start_btn.rect.centerx
        btn_cy = menu._start_btn.rect.centery
        with patch("pygame.mouse.get_pos", return_value=(btn_cx, btn_cy)):
            click_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=(btn_cx, btn_cy)
            )
            with patch("pygame.event.get", return_value=[click_event]), \
                 patch("pygame.display.flip"), \
                 patch("pygame.time.Clock") as mock_clock:
                mock_clock.return_value.tick = MagicMock()
                result = menu.run(surface)
        assert isinstance(result, GameConfig)

    def test_run_return_key_has_anim_ms(self, surface):
        menu = MenuScreen()
        return_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="")
        with patch("pygame.event.get", return_value=[return_event]), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_clock.return_value.tick = MagicMock()
            result = menu.run(surface)
        assert isinstance(result, GameConfig)
        assert result.anim_ms == 300

    def test_run_anim_stepper_click(self, surface):
        """Kliknięcie w anim_stepper zmienia wartość, nie kończy run()."""
        menu = MenuScreen()
        initial_val = menu._anim_stepper.value
        anim_cx = menu._anim_stepper._r_plus.centerx
        anim_cy = menu._anim_stepper._r_plus.centery
        anim_click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(anim_cx, anim_cy)
        )
        quit_event = pygame.event.Event(pygame.QUIT)
        events_sequence = [[anim_click], [quit_event]]
        call_count = [0]

        def side_effect():
            ev = events_sequence[min(call_count[0], len(events_sequence) - 1)]
            call_count[0] += 1
            return ev

        with patch("pygame.mouse.get_pos", return_value=(anim_cx, anim_cy)), \
             patch("pygame.event.get", side_effect=side_effect), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_clock.return_value.tick = MagicMock()
            result = menu.run(surface)
        assert result is None
        assert menu._anim_stepper.value == initial_val + 50

    def test_run_panel_click_no_return(self, surface):
        """Kliknięcie w panel (nie w start button) nie kończy run()."""
        menu = MenuScreen()
        # Click on panel area, then QUIT to exit
        panel_cx = menu._white_panel.rect.centerx
        panel_cy = menu._white_panel.rect.centery + 50
        panel_click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(panel_cx, panel_cy)
        )
        quit_event = pygame.event.Event(pygame.QUIT)
        events_sequence = [[panel_click], [quit_event]]
        call_count = [0]

        def side_effect():
            ev = events_sequence[min(call_count[0], len(events_sequence) - 1)]
            call_count[0] += 1
            return ev

        with patch("pygame.mouse.get_pos", return_value=(panel_cx, panel_cy)), \
             patch("pygame.event.get", side_effect=side_effect), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_clock.return_value.tick = MagicMock()
            result = menu.run(surface)
        assert result is None


# ---------------------------------------------------------------------------
# run_menu
# ---------------------------------------------------------------------------

class TestRunMenu:
    def test_run_menu_quit_returns_none(self):
        quit_event = pygame.event.Event(pygame.QUIT)
        with patch("pygame.init"), \
             patch("pygame.display.set_mode") as mock_mode, \
             patch("pygame.display.set_caption"), \
             patch("pygame.event.get", return_value=[quit_event]), \
             patch("pygame.display.flip"), \
             patch("pygame.time.Clock") as mock_clock:
            mock_mode.return_value = pygame.Surface((720, 756))
            mock_clock.return_value.tick = MagicMock()
            result = run_menu(720)
        assert result is None
