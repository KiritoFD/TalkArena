"""Contracts for multimodal + multi-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

IntentType = Literal["probe", "challenge", "agree", "summary", "shift_topic", "rescue"]
SceneType = Literal["interview", "dinner"]


@dataclass
class UserState:
    wants_to_speak: float
    confusion: float
    stress_arousal: float
    valence: float
    addressee_distribution: Dict[str, float]
    pace_preference: Literal["slow", "normal", "fast"]


@dataclass
class NPCProposal:
    npc_id: str
    wants_to_speak: bool
    urgency: float
    intent: IntentType
    interrupt_ok: bool
    max_duration_ms: int = 5000
    nonverbal: Dict[str, str] = field(default_factory=dict)


@dataclass
class DirectorIntent:
    scene: SceneType
    beat: str
    pressure_bias: float
    rescue_bias: float
    preferred_roles: List[str] = field(default_factory=list)


@dataclass
class FloorDecision:
    speaker_id: Optional[str]
    reason: str
    allow_interrupt: bool
    interrupt_window_ms: int
    runner_up: Optional[str] = None


@dataclass
class SpeechInstruction:
    npc_id: str
    text_prompt: str
    max_tokens: int = 160


@dataclass
class NonverbalInstruction:
    npc_id: str
    state: Literal["idle", "listening", "reacting", "speaking"]
    expression: str
    intensity: float
    gaze_target: Literal["user", "speaker", "table"]
    backchannel: Optional[str] = None


@dataclass
class RenderFrame:
    timestamp_ms: int
    nonverbals: List[NonverbalInstruction]
