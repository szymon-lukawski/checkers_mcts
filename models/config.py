from pydantic import BaseModel, Field
from enum import Enum


class AgentType(str, Enum):
    HUMAN = "human"
    RANDOM = "random"
    MINIMAX = "minimax"
    MCTS = "mcts"


class AgentConfig(BaseModel):
    agent_type: AgentType = AgentType.RANDOM
    minimax_depth: int = Field(default=6, ge=1, le=20)
    mcts_time_ms: int = Field(default=1000, ge=100, le=30000)


class GameConfig(BaseModel):
    white_agent: AgentConfig = AgentConfig(agent_type=AgentType.HUMAN)
    black_agent: AgentConfig = AgentConfig(agent_type=AgentType.RANDOM)
    fps: int = Field(default=60, ge=1, le=120)
    window_size: int = Field(default=720, ge=400, le=1200)
