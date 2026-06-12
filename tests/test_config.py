"""
Testy dla models/config.py
"""

import pytest
from pydantic import ValidationError
from models.config import AgentType, AgentConfig, GameConfig


class TestAgentType:
    def test_human_value(self):
        assert AgentType.HUMAN == "human"

    def test_random_value(self):
        assert AgentType.RANDOM == "random"

    def test_minimax_value(self):
        assert AgentType.MINIMAX == "minimax"

    def test_mcts_value(self):
        assert AgentType.MCTS == "mcts"

    def test_is_str_enum(self):
        assert isinstance(AgentType.HUMAN, str)
        assert AgentType.HUMAN == "human"

    def test_all_values(self):
        values = {a.value for a in AgentType}
        assert values == {"human", "random", "minimax", "mcts"}


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(agent_type=AgentType.RANDOM)
        assert cfg.agent_type == AgentType.RANDOM
        assert cfg.minimax_depth == 6
        assert cfg.mcts_time_ms == 1000

    def test_default_agent_type(self):
        cfg = AgentConfig()
        assert cfg.agent_type == AgentType.RANDOM

    def test_custom_minimax_depth(self):
        cfg = AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=4)
        assert cfg.minimax_depth == 4

    def test_custom_mcts_time(self):
        cfg = AgentConfig(agent_type=AgentType.MCTS, mcts_time_ms=2000)
        assert cfg.mcts_time_ms == 2000

    def test_minimax_depth_min(self):
        cfg = AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=1)
        assert cfg.minimax_depth == 1

    def test_minimax_depth_max(self):
        cfg = AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=20)
        assert cfg.minimax_depth == 20

    def test_minimax_depth_out_of_range_low(self):
        with pytest.raises(ValidationError):
            AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=0)

    def test_minimax_depth_out_of_range_high(self):
        with pytest.raises(ValidationError):
            AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=21)

    def test_mcts_time_min(self):
        cfg = AgentConfig(agent_type=AgentType.MCTS, mcts_time_ms=100)
        assert cfg.mcts_time_ms == 100

    def test_mcts_time_max(self):
        cfg = AgentConfig(agent_type=AgentType.MCTS, mcts_time_ms=30000)
        assert cfg.mcts_time_ms == 30000

    def test_mcts_time_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentConfig(agent_type=AgentType.MCTS, mcts_time_ms=50)

    def test_human_config(self):
        cfg = AgentConfig(agent_type=AgentType.HUMAN)
        assert cfg.agent_type == AgentType.HUMAN


class TestGameConfig:
    def test_defaults(self):
        cfg = GameConfig(
            white_agent=AgentConfig(agent_type=AgentType.HUMAN),
            black_agent=AgentConfig(agent_type=AgentType.RANDOM),
        )
        assert cfg.fps == 60
        assert cfg.window_size == 720

    def test_default_agents(self):
        cfg = GameConfig()
        assert cfg.white_agent.agent_type == AgentType.HUMAN
        assert cfg.black_agent.agent_type == AgentType.RANDOM

    def test_custom_fps(self):
        cfg = GameConfig(fps=30)
        assert cfg.fps == 30

    def test_custom_window_size(self):
        cfg = GameConfig(window_size=800)
        assert cfg.window_size == 800

    def test_fps_min(self):
        cfg = GameConfig(fps=1)
        assert cfg.fps == 1

    def test_fps_max(self):
        cfg = GameConfig(fps=120)
        assert cfg.fps == 120

    def test_fps_out_of_range(self):
        with pytest.raises(ValidationError):
            GameConfig(fps=0)

    def test_fps_too_high(self):
        with pytest.raises(ValidationError):
            GameConfig(fps=121)

    def test_window_size_min(self):
        cfg = GameConfig(window_size=400)
        assert cfg.window_size == 400

    def test_window_size_max(self):
        cfg = GameConfig(window_size=1200)
        assert cfg.window_size == 1200

    def test_window_size_too_small(self):
        with pytest.raises(ValidationError):
            GameConfig(window_size=399)

    def test_window_size_too_large(self):
        with pytest.raises(ValidationError):
            GameConfig(window_size=1201)

    def test_custom_agents(self):
        white = AgentConfig(agent_type=AgentType.MINIMAX, minimax_depth=4)
        black = AgentConfig(agent_type=AgentType.MCTS, mcts_time_ms=500)
        cfg = GameConfig(white_agent=white, black_agent=black)
        assert cfg.white_agent.agent_type == AgentType.MINIMAX
        assert cfg.black_agent.agent_type == AgentType.MCTS

    def test_anim_ms_default(self):
        cfg = GameConfig()
        assert cfg.anim_ms == 300

    def test_anim_ms_min(self):
        cfg = GameConfig(anim_ms=50)
        assert cfg.anim_ms == 50

    def test_anim_ms_max(self):
        cfg = GameConfig(anim_ms=2000)
        assert cfg.anim_ms == 2000

    def test_anim_ms_too_low(self):
        with pytest.raises(ValidationError):
            GameConfig(anim_ms=49)

    def test_anim_ms_too_high(self):
        with pytest.raises(ValidationError):
            GameConfig(anim_ms=2001)

    def test_anim_ms_custom(self):
        cfg = GameConfig(anim_ms=500)
        assert cfg.anim_ms == 500
