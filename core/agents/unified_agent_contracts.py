from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal


@dataclass
class NPCCharacter:
    npc_id: str
    name: str
    personality: str
    speaking_style: str
    avatar: str = "👤"


@dataclass
class NPCUtterance:
    npc_id: str
    text: str
    delay_ms: int = 1500
    emotion: Optional[str] = None


@dataclass
class ConversationTurn:
    speaker: str
    text: str
    timestamp_ms: int
    is_user: bool = False


@dataclass
class UnifiedAgentResponse:
    utterances: List[NPCUtterance]
    should_await_user: bool
    reason: str
    continue_conversation: bool = True


SceneType = Literal["shandong_dinner", "business_dinner", "interview", "debate"]
