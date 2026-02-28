"""Speaker selection using bid scoring, cooldown and interrupt window."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .contracts import DirectorIntent, FloorDecision, NPCProposal, UserState


@dataclass
class NPCProfile:
    npc_id: str
    role: str
    base_priority: float


class SpeakerSelector:
    def __init__(self, interrupt_window_ms: int = 700, cooldown_s: float = 4.0):
        self.interrupt_window_ms = interrupt_window_ms
        self.cooldown_s = cooldown_s
        self.last_spoken_at: Dict[str, float] = {}

    def decide(
        self,
        proposals: List[NPCProposal],
        profiles: Dict[str, NPCProfile],
        user_state: UserState,
        director_intent: DirectorIntent,
    ) -> FloorDecision:
        ranked = sorted(
            (
                (proposal, self._score(proposal, profiles.get(proposal.npc_id), user_state, director_intent))
                for proposal in proposals
                if proposal.wants_to_speak
            ),
            key=lambda row: row[1],
            reverse=True,
        )

        if not ranked:
            return FloorDecision(
                speaker_id=None,
                reason="no_valid_proposal",
                allow_interrupt=False,
                interrupt_window_ms=self.interrupt_window_ms,
            )

        winner, win_score = ranked[0]
        runner_up = ranked[1][0].npc_id if len(ranked) > 1 else None
        self.last_spoken_at[winner.npc_id] = time.time()

        allow_interrupt = bool(runner_up and ranked[1][0].interrupt_ok and ranked[1][1] > win_score * 0.78)
        return FloorDecision(
            speaker_id=winner.npc_id,
            runner_up=runner_up,
            reason=f"score={win_score:.3f}, intent={winner.intent}, beat={director_intent.beat}",
            allow_interrupt=allow_interrupt,
            interrupt_window_ms=self.interrupt_window_ms,
        )

    def _score(
        self,
        proposal: NPCProposal,
        profile: Optional[NPCProfile],
        user_state: UserState,
        director_intent: DirectorIntent,
    ) -> float:
        profile = profile or NPCProfile(npc_id=proposal.npc_id, role="guest", base_priority=0.5)

        priority = profile.base_priority + (0.25 if profile.role in director_intent.preferred_roles else 0.0)
        addressee_bonus = user_state.addressee_distribution.get(proposal.npc_id, 0.0) * 0.45
        urgency_bonus = proposal.urgency * 0.35
        rescue_bonus = director_intent.rescue_bias * 0.25 if proposal.intent == "rescue" else 0.0
        challenge_bonus = director_intent.pressure_bias * 0.2 if proposal.intent in ("probe", "challenge") else 0.0

        elapsed = time.time() - self.last_spoken_at.get(proposal.npc_id, 0.0)
        cooldown_penalty = 0.35 if elapsed < self.cooldown_s else 0.0
        length_penalty = 0.15 if proposal.max_duration_ms > 6000 else 0.0

        return max(0.0, priority + addressee_bonus + urgency_bonus + rescue_bonus + challenge_bonus - cooldown_penalty - length_penalty)
