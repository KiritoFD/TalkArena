from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import uuid


class AggressionLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AgentProfile(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str
    personality_traits: List[str] = Field(default_factory=list)
    initial_dominance: float = Field(default=50.0, ge=0, le=100)
    system_prompt: str
    tts_speaker_id: str = "default"
    style_config: Dict[str, Any] = Field(default_factory=dict)


class EmotionMetadata(BaseModel):
    aggression_level: AggressionLevel = AggressionLevel.MEDIUM
    confidence_level: int = Field(default=50, ge=0, le=100)
    stress_level: int = Field(default=50, ge=0, le=100)

    def get_aggression_score(self) -> int:
        mapping = {
            AggressionLevel.LOW: 20,
            AggressionLevel.MEDIUM: 50,
            AggressionLevel.HIGH: 80,
        }
        return mapping[self.aggression_level]


class DialogueTurn(BaseModel):
    speaker: str
    speaker_name: str
    text: str
    emotion: Optional[EmotionMetadata] = None
    dominance_delta: float = 0.0
    audio_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str
    agents: List[AgentProfile] = Field(default_factory=list)
    history: List[DialogueTurn] = Field(default_factory=list)
    current_dominance: Dict[str, float] = Field(default_factory=dict)
    turn_count: int = 0

    def initialize_dominance(self):
        for agent in self.agents:
            self.current_dominance[agent.agent_id] = agent.initial_dominance


def compute_dominance_delta(emotion: EmotionMetadata) -> float:
    """Calculate dominance change based on emotion metadata."""
    aggression_score = emotion.get_aggression_score()
    delta = (aggression_score * 0.4 + emotion.confidence_level * 0.6 - emotion.stress_level * 0.3) / 10
    return round(delta, 2)


def clamp_dominance(value: float) -> float:
    """Clamp dominance value to [0, 100] range."""
    return max(0.0, min(100.0, value))
