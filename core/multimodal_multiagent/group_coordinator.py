"""Group-level coordination signal for nonverbal and pacing alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .contracts import DirectorIntent, FloorDecision, UserState


@dataclass
class GroupCoordinationSignal:
    phase: str
    should_soften: bool
    should_pressure: bool
    target_pace: str
    notes: str


class GroupCoordinator:
    def coordinate(
        self,
        user_state: UserState,
        director_intent: DirectorIntent,
        floor_decision: FloorDecision,
        context: Dict,
    ) -> GroupCoordinationSignal:
        turn = int(context.get("turn", 0))
        phase = self._phase(turn)

        should_soften = bool(user_state.confusion > 0.62 or director_intent.beat == "controlled_rescue")
        should_pressure = bool(
            director_intent.scene == "interview"
            and user_state.stress_arousal < 0.7
            and director_intent.beat == "pressure_check"
        )

        notes = [f"phase={phase}", f"beat={director_intent.beat}"]
        if floor_decision.speaker_id:
            notes.append(f"speaker={floor_decision.speaker_id}")
        if floor_decision.allow_interrupt:
            notes.append("runner_up_interrupt_enabled")

        return GroupCoordinationSignal(
            phase=phase,
            should_soften=should_soften,
            should_pressure=should_pressure,
            target_pace=user_state.pace_preference,
            notes=", ".join(notes),
        )

    def _phase(self, turn: int) -> str:
        if turn < 2:
            return "opening"
        if turn < 6:
            return "explore"
        return "resolve"
