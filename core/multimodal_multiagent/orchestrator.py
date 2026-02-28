"""Top-level orchestrator for multimodal multi-agent scenes."""

from __future__ import annotations

from typing import Dict, List, Optional

from .contracts import NPCProposal, NonverbalInstruction, RenderFrame, SceneType, SpeechInstruction
from .group_coordinator import GroupCoordinationSignal, GroupCoordinator
from .npc_policy import NPCPolicyPlanner
from .scenario_director import ScenarioDirector
from .speaker_selection import NPCProfile, SpeakerSelector
from .user_signal_fusion import UserSignalFusion


class MultimodalMultiAgentOrchestrator:
    def __init__(self):
        self.user_fusion = UserSignalFusion()
        self.director = ScenarioDirector()
        self.speaker_selector = SpeakerSelector()
        self.npc_policy = NPCPolicyPlanner()
        self.group_coordinator = GroupCoordinator()

    def run_tick(
        self,
        scene: SceneType,
        features: Dict,
        proposals: Optional[List[NPCProposal]],
        profiles: Dict[str, NPCProfile],
        context: Dict,
        ts_ms: int,
    ) -> Dict:
        user_state = self.user_fusion.fuse(features, list(profiles.keys()))
        director_intent = self.director.plan(scene, user_state, context)

        auto_proposals = proposals is None
        final_proposals = proposals or self.npc_policy.build_proposals(
            profiles=profiles,
            user_state=user_state,
            director_intent=director_intent,
            context=context,
        )

        floor_decision = self.speaker_selector.decide(final_proposals, profiles, user_state, director_intent)
        coordination = self.group_coordinator.coordinate(user_state, director_intent, floor_decision, context)

        speech_instruction = None
        if floor_decision.speaker_id:
            speech_instruction = SpeechInstruction(
                npc_id=floor_decision.speaker_id,
                text_prompt=(
                    f"scene={scene}; beat={director_intent.beat}; phase={coordination.phase}; "
                    f"pace={coordination.target_pace}; keep under 80 Chinese chars."
                ),
            )

        nonverbals = [
            self._nonverbal_for(
                npc_id=npc_id,
                speaker_id=floor_decision.speaker_id,
                runner_up=floor_decision.runner_up,
                allow_interrupt=floor_decision.allow_interrupt,
                coordination=coordination,
            )
            for npc_id in profiles
        ]

        return {
            "user_state": user_state,
            "director_intent": director_intent,
            "coordination": coordination,
            "proposals": final_proposals,
            "auto_proposals": auto_proposals,
            "floor_decision": floor_decision,
            "speech_instruction": speech_instruction,
            "render_frame": RenderFrame(timestamp_ms=ts_ms, nonverbals=nonverbals),
        }

    def _nonverbal_for(
        self,
        npc_id: str,
        speaker_id: Optional[str],
        runner_up: Optional[str],
        allow_interrupt: bool,
        coordination: GroupCoordinationSignal,
    ) -> NonverbalInstruction:
        if npc_id == speaker_id:
            return NonverbalInstruction(
                npc_id=npc_id,
                state="speaking",
                expression="focused" if not coordination.should_soften else "calm",
                intensity=0.68 if coordination.should_pressure else 0.56,
                gaze_target="user",
                backchannel=None,
            )

        base_state = "reacting" if coordination.phase != "opening" else "listening"
        expression = "supportive" if coordination.should_soften else "listening"
        backchannel = "我补一句" if allow_interrupt and npc_id == runner_up else None

        return NonverbalInstruction(
            npc_id=npc_id,
            state=base_state,
            expression=expression,
            intensity=0.42 if coordination.should_soften else 0.34,
            gaze_target="speaker",
            backchannel=backchannel,
        )
