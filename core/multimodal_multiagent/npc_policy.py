"""Per-NPC lightweight policy for independent parallel proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .contracts import DirectorIntent, NPCProposal, UserState
from .speaker_selection import NPCProfile


@dataclass
class NPCMicroState:
    energy: float = 0.55
    assertiveness: float = 0.5
    cooperativeness: float = 0.5
    last_intent: str = "summary"
    turns_silent: int = 0


class NPCPolicyPlanner:
    ROLE_DEFAULTS = {
        "chief_interviewer": (0.72, 0.35),
        "tech_interviewer": (0.68, 0.30),
        "hr": (0.45, 0.70),
        "observer": (0.25, 0.60),
        "host": (0.55, 0.70),
        "elder": (0.62, 0.50),
        "friend": (0.45, 0.72),
        "guest": (0.40, 0.50),
    }

    def __init__(self):
        self.states: Dict[str, NPCMicroState] = {}

    def build_proposals(
        self,
        profiles: Dict[str, NPCProfile],
        user_state: UserState,
        director_intent: DirectorIntent,
        context: Dict,
    ) -> List[NPCProposal]:
        proposals: List[NPCProposal] = []

        for npc_id, profile in profiles.items():
            state = self.states.setdefault(npc_id, self._bootstrap(profile))
            proposal = self._proposal_for(npc_id, profile, state, user_state, director_intent, context)
            proposals.append(proposal)
            self._update_state(state, proposal)

        return proposals

    def _bootstrap(self, profile: NPCProfile) -> NPCMicroState:
        assertive, cooperative = self.ROLE_DEFAULTS.get(profile.role, self.ROLE_DEFAULTS["guest"])
        return NPCMicroState(assertiveness=assertive, cooperativeness=cooperative)

    def _proposal_for(
        self,
        npc_id: str,
        profile: NPCProfile,
        state: NPCMicroState,
        user_state: UserState,
        director_intent: DirectorIntent,
        context: Dict,
    ) -> NPCProposal:
        addressed = user_state.addressee_distribution.get(npc_id, 0.0)
        role_boost = 0.18 if profile.role in director_intent.preferred_roles else 0.0
        rescue_pressure = director_intent.rescue_bias * user_state.confusion
        challenge_pressure = director_intent.pressure_bias * (1.0 - user_state.confusion)

        urgency = (
            0.28 * state.assertiveness
            + 0.23 * addressed
            + 0.22 * state.energy
            + 0.15 * role_boost
            + 0.12 * (rescue_pressure if profile.role == "hr" else challenge_pressure)
        )
        urgency = max(0.0, min(1.0, urgency))

        intent = self._pick_intent(profile.role, director_intent, user_state, state)
        wants_to_speak = urgency > 0.42 or state.turns_silent >= 3
        interrupt_ok = intent in ("challenge", "probe", "agree") and user_state.wants_to_speak < 0.8

        max_duration_ms = 2400 if director_intent.scene == "dinner" else 3200
        if intent in ("summary", "rescue"):
            max_duration_ms += 500

        nonverbal = {
            "expression": "focused" if wants_to_speak else "listening",
            "gaze_target": "user" if addressed > 0.26 else "speaker",
            "micro_action": "note" if profile.role == "observer" else "nod",
        }

        return NPCProposal(
            npc_id=npc_id,
            wants_to_speak=wants_to_speak,
            urgency=urgency,
            intent=intent,
            interrupt_ok=interrupt_ok,
            max_duration_ms=max_duration_ms,
            nonverbal=nonverbal,
        )

    def _pick_intent(
        self,
        role: str,
        director_intent: DirectorIntent,
        user_state: UserState,
        state: NPCMicroState,
    ) -> str:
        if role == "hr" and user_state.confusion > 0.55:
            return "rescue"
        if role in ("chief_interviewer", "tech_interviewer") and director_intent.beat == "pressure_check":
            return "challenge" if state.assertiveness > 0.65 else "probe"
        if director_intent.scene == "dinner":
            if user_state.valence < 0.35:
                return "rescue"
            return "agree" if role == "friend" else "summary"
        return "probe"

    def _update_state(self, state: NPCMicroState, proposal: NPCProposal) -> None:
        if proposal.wants_to_speak:
            state.turns_silent = 0
            state.energy = max(0.25, state.energy - 0.05)
        else:
            state.turns_silent += 1
            state.energy = min(0.92, state.energy + 0.04)
        state.last_intent = proposal.intent
