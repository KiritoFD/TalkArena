"""Fuse multimodal front-end features into orchestration-ready user state."""

from __future__ import annotations

from typing import Dict

from .contracts import UserState


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class UserSignalFusion:
    def fuse(self, features: Dict, npc_ids: list[str]) -> UserState:
        audio = features.get("audio", {})
        face = features.get("face", {})
        text = features.get("text", {})

        pause_ms = float(audio.get("pause_ms", 0))
        speech_rate = float(audio.get("speech_rate", 4.0))
        pitch_jitter = float(audio.get("pitch_jitter", 0.2))
        volume_var = float(audio.get("volume_var", 0.2))
        pre_speech = float(audio.get("pre_speech_motion", 0.0))

        brow_furrow = float(face.get("brow_furrow", 0.0))
        blink_jump = float(face.get("blink_rate_jump", 0.0))
        facial_tension = float(face.get("facial_tension", 0.2))
        smile = float(face.get("smile", 0.2))
        head_pitch = float(face.get("head_forward", 0.0))
        gaze = face.get("gaze_target", "")

        mentions = text.get("mentions", [])
        self_repair = float(text.get("self_repair", 0.0))
        sentiment = float(text.get("sentiment", 0.0))

        wants_to_speak = _clamp(0.35 * pre_speech + 0.25 * head_pitch + 0.4 * (1.0 if pause_ms < 450 else 0.2))
        confusion = _clamp(0.28 * brow_furrow + 0.22 * blink_jump + 0.25 * self_repair + 0.25 * (1.0 if speech_rate < 3.2 else 0.2))
        stress_arousal = _clamp(0.34 * pitch_jitter + 0.28 * volume_var + 0.38 * facial_tension)
        valence = _clamp(0.55 * smile + 0.45 * ((sentiment + 1) / 2.0))

        addressee_distribution = self._infer_addressee(npc_ids, mentions, gaze)
        pace_preference = self._infer_pace(confusion, stress_arousal, pause_ms)

        return UserState(
            wants_to_speak=wants_to_speak,
            confusion=confusion,
            stress_arousal=stress_arousal,
            valence=valence,
            addressee_distribution=addressee_distribution,
            pace_preference=pace_preference,
        )

    def _infer_addressee(self, npc_ids: list[str], mentions: list[str], gaze: str) -> Dict[str, float]:
        if not npc_ids:
            return {}

        weights = {npc_id: 1.0 for npc_id in npc_ids}
        for mention in mentions:
            if mention in weights:
                weights[mention] += 2.0

        if gaze in weights:
            weights[gaze] += 1.5

        total = sum(weights.values()) or 1.0
        return {npc_id: weight / total for npc_id, weight in weights.items()}

    def _infer_pace(self, confusion: float, stress: float, pause_ms: float) -> str:
        load = (confusion + stress) / 2
        if load > 0.65 or pause_ms > 1200:
            return "slow"
        if load < 0.35 and pause_ms < 400:
            return "fast"
        return "normal"
