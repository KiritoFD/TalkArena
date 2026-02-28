"""Scenario-level directing rules for interview and dinner modes."""

from __future__ import annotations

from typing import Dict

from .contracts import DirectorIntent, SceneType, UserState


class ScenarioDirector:
    def plan(self, scene: SceneType, user_state: UserState, context: Dict) -> DirectorIntent:
        turn = int(context.get("turn", 0))

        if scene == "interview":
            beat = "pressure_check" if user_state.stress_arousal < 0.7 else "controlled_rescue"
            preferred_roles = ["chief_interviewer", "tech_interviewer"]
            if user_state.confusion > 0.6:
                preferred_roles = ["hr", "chief_interviewer"]
            pressure_bias = 0.7 if turn % 3 != 0 else 0.5
            rescue_bias = 0.7 if user_state.confusion > 0.6 else 0.35
            return DirectorIntent(
                scene=scene,
                beat=beat,
                pressure_bias=pressure_bias,
                rescue_bias=rescue_bias,
                preferred_roles=preferred_roles,
            )

        beat = "table_banter" if user_state.valence > 0.4 else "polite_realign"
        return DirectorIntent(
            scene=scene,
            beat=beat,
            pressure_bias=0.25,
            rescue_bias=0.55 if user_state.stress_arousal > 0.55 else 0.3,
            preferred_roles=["host", "friend", "elder"],
        )
