"""
情感状态机模块
负责用户情感状态的时序追踪、状态推断和平滑处理
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time


EMOTION_STATES = [
    "confident",
    "nervous",
    "angry",
    "happy",
    "sad",
    "surprised",
    "confused",
    "contemptuous",
    "neutral",
]


@dataclass
class MicroExpressionFeatures:
    """微表情特征 - 50维向量"""

    eye_openness_left: float = 0.5
    eye_openness_right: float = 0.5
    eye_aspect_ratio: float = 0.3
    brow_inner_up_left: float = 0.0
    brow_inner_up_right: float = 0.0
    brow_outer_up_left: float = 0.0
    brow_outer_up_right: float = 0.0
    brow_tension: float = 0.0
    gaze_direction_x: float = 0.0
    gaze_direction_y: float = 0.0
    blink_rate: float = 0.0

    mouth_open_ratio: float = 0.0
    mouth_width: float = 0.5
    lip_corner_up_left: float = 0.0
    lip_corner_up_right: float = 0.0
    lip_corner_down_left: float = 0.0
    lip_corner_down_right: float = 0.0
    lip_pressure: float = 0.0
    smile_asymmetry: float = 0.0
    smile_genuine_score: float = 0.0

    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    happiness: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    contempt: float = 0.0
    interest: float = 0.0
    confusion: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> "MicroExpressionFeatures":
        return cls(
            eye_openness_left=data.get("eyeOpennessLeft", 0.5),
            eye_openness_right=data.get("eyeOpennessRight", 0.5),
            eye_aspect_ratio=data.get("eyeAspectRatio", 0.3),
            brow_inner_up_left=data.get("browInnerUpLeft", 0.0),
            brow_inner_up_right=data.get("browInnerUpRight", 0.0),
            brow_outer_up_left=data.get("browOuterUpLeft", 0.0),
            brow_outer_up_right=data.get("browOuterUpRight", 0.0),
            brow_tension=data.get("browTension", 0.0),
            gaze_direction_x=data.get("gazeDirectionX", 0.0),
            gaze_direction_y=data.get("gazeDirectionY", 0.0),
            blink_rate=data.get("blinkRate", 0.0),
            mouth_open_ratio=data.get("mouthOpenRatio", 0.0),
            mouth_width=data.get("mouthWidth", 0.5),
            lip_corner_up_left=data.get("lipCornerUpLeft", 0.0),
            lip_corner_up_right=data.get("lipCornerUpRight", 0.0),
            lip_corner_down_left=data.get("lipCornerDownLeft", 0.0),
            lip_corner_down_right=data.get("lipCornerDownRight", 0.0),
            lip_pressure=data.get("lipPressure", 0.0),
            smile_asymmetry=data.get("smileAsymmetry", 0.0),
            smile_genuine_score=data.get("smileGenuineScore", 0.0),
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            happiness=data.get("happiness", 0.0),
            sadness=data.get("sadness", 0.0),
            anger=data.get("anger", 0.0),
            fear=data.get("fear", 0.0),
            surprise=data.get("surprise", 0.0),
            disgust=data.get("disgust", 0.0),
            contempt=data.get("contempt", 0.0),
            interest=data.get("interest", 0.0),
            confusion=data.get("confusion", 0.0),
        )

    def to_dict(self) -> Dict:
        return {
            "eyeOpennessLeft": self.eye_openness_left,
            "eyeOpennessRight": self.eye_openness_right,
            "eyeAspectRatio": self.eye_aspect_ratio,
            "browInnerUpLeft": self.brow_inner_up_left,
            "browInnerUpRight": self.brow_inner_up_right,
            "browOuterUpLeft": self.brow_outer_up_left,
            "browOuterUpRight": self.brow_outer_up_right,
            "browTension": self.brow_tension,
            "gazeDirectionX": self.gaze_direction_x,
            "gazeDirectionY": self.gaze_direction_y,
            "blinkRate": self.blink_rate,
            "mouthOpenRatio": self.mouth_open_ratio,
            "mouthWidth": self.mouth_width,
            "lipCornerUpLeft": self.lip_corner_up_left,
            "lipCornerUpRight": self.lip_corner_up_right,
            "lipCornerDownLeft": self.lip_corner_down_left,
            "lipCornerDownRight": self.lip_corner_down_right,
            "lipPressure": self.lip_pressure,
            "smileAsymmetry": self.smile_asymmetry,
            "smileGenuineScore": self.smile_genuine_score,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "happiness": self.happiness,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "disgust": self.disgust,
            "contempt": self.contempt,
            "interest": self.interest,
            "confusion": self.confusion,
        }


@dataclass
class VoiceEmotionFeatures:
    """语音情感特征"""

    loudness: float = 0.5
    speech_rate: float = 3.0
    pitch_mean: float = 150.0
    pitch_std: float = 20.0
    energy_variance: float = 0.1

    harmonic_ratio: float = 0.5
    noise_ratio: float = 0.1
    jitter: float = 0.01
    shimmer: float = 0.02

    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5

    emotion_scores: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceEmotionFeatures":
        return cls(
            loudness=data.get("loudness", 0.5),
            speech_rate=data.get("speechRate", 3.0),
            pitch_mean=data.get("pitchMean", 150.0),
            pitch_std=data.get("pitchStd", 20.0),
            energy_variance=data.get("energyVariance", 0.1),
            harmonic_ratio=data.get("harmonicRatio", 0.5),
            noise_ratio=data.get("noiseRatio", 0.1),
            jitter=data.get("jitter", 0.01),
            shimmer=data.get("shimmer", 0.02),
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            emotion_scores=data.get("emotionScores", {}),
        )

    def to_dict(self) -> Dict:
        return {
            "loudness": self.loudness,
            "speechRate": self.speech_rate,
            "pitchMean": self.pitch_mean,
            "pitchStd": self.pitch_std,
            "energyVariance": self.energy_variance,
            "harmonicRatio": self.harmonic_ratio,
            "noiseRatio": self.noise_ratio,
            "jitter": self.jitter,
            "shimmer": self.shimmer,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "emotionScores": self.emotion_scores,
        }


@dataclass
class MultimodalEmotionState:
    """融合后的多模态情感状态"""

    primary_emotion: str = "neutral"
    secondary_emotion: Optional[str] = None
    emotion_intensity: float = 0.5
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    confidence: float = 0.5

    face_confidence: float = 0.5
    voice_confidence: float = 0.5

    inconsistencies: List[Dict] = field(default_factory=list)
    hidden_sentiment: Optional[str] = None

    face_features: Optional[MicroExpressionFeatures] = None
    voice_features: Optional[VoiceEmotionFeatures] = None

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "primaryEmotion": self.primary_emotion,
            "secondaryEmotion": self.secondary_emotion,
            "emotionIntensity": self.emotion_intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "confidence": self.confidence,
            "faceConfidence": self.face_confidence,
            "voiceConfidence": self.voice_confidence,
            "inconsistencies": self.inconsistencies,
            "hiddenSentiment": self.hidden_sentiment,
            "timestamp": self.timestamp,
        }


class UserEmotionStateMachine:
    """用户情感状态机"""

    def __init__(self, smoothing_frames: int = 30):
        self.current_state = "neutral"
        self.state_confidence = 0.5
        self.state_history: List[Tuple[str, float, float]] = []
        self.frame_buffer: List[np.ndarray] = []
        self.smoothing_frames = smoothing_frames

        self.emotion_intensity = 0.5
        self.valence = 0.0
        self.arousal = 0.5
        self.dominance = 0.5

        self.inconsistencies: List[Dict] = []

    def update(
        self,
        face_features: Optional[MicroExpressionFeatures] = None,
        voice_features: Optional[VoiceEmotionFeatures] = None,
        text_sentiment: Optional[Dict] = None,
    ) -> MultimodalEmotionState:
        """更新情感状态，返回融合后的多模态情感"""

        emotion_vector = np.zeros(12, dtype=np.float32)
        face_confidence = 0.5
        voice_confidence = 0.5

        if face_features:
            face_vector = self._extract_face_emotion(face_features)
            emotion_vector = emotion_vector * 0.5 + face_vector * 0.5
            face_confidence = self._compute_face_confidence(face_features)

        if voice_features:
            voice_vector = self._extract_voice_emotion(voice_features)
            emotion_vector = emotion_vector * 0.5 + voice_vector * 0.5
            voice_confidence = self._compute_voice_confidence(voice_features)

        smoothed = self._temporal_smoothing(emotion_vector)

        self._update_state_variables(smoothed)

        self.inconsistencies = self._detect_inconsistencies(
            face_features, voice_features, text_sentiment
        )

        self._record_state_history()

        return MultimodalEmotionState(
            primary_emotion=self.current_state,
            emotion_intensity=self.emotion_intensity,
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            confidence=self.state_confidence,
            face_confidence=face_confidence,
            voice_confidence=voice_confidence,
            inconsistencies=self.inconsistencies,
            hidden_sentiment=self._infer_hidden_sentiment(),
            face_features=face_features,
            voice_features=voice_features,
            timestamp=time.time(),
        )

    def _extract_face_emotion(self, f: MicroExpressionFeatures) -> np.ndarray:
        """从表情特征提取情感向量"""
        valence = (
            f.happiness * 0.3
            + f.smile_genuine_score * 0.3
            + f.brow_tension * (-0.2)
            + f.lip_corner_up_left * 0.2
            + f.lip_corner_up_right * 0.2
            - f.sadness * 0.3
            - f.contempt * 0.2
        )

        arousal = (
            f.eye_openness_left * 0.15
            + f.eye_openness_right * 0.15
            + f.brow_tension * 0.3
            + f.mouth_open_ratio * 0.2
            + (1 - f.eye_aspect_ratio) * 0.2
        )

        dominance = (
            f.gaze_direction_x * 0.2
            + f.brow_outer_up_left * 0.2
            + f.brow_outer_up_right * 0.2
            - f.smile_asymmetry * 0.1
            + f.lip_pressure * 0.2
        )

        return np.array(
            [
                valence,
                arousal,
                dominance,
                f.happiness,
                f.sadness,
                f.anger,
                f.fear,
                f.surprise,
                f.disgust,
                f.contempt,
                f.interest,
                f.confusion,
            ],
            dtype=np.float32,
        )

    def _extract_voice_emotion(self, v: VoiceEmotionFeatures) -> np.ndarray:
        """从语音特征提取情感向量"""
        emotion_scores = v.emotion_scores or {}

        return np.array(
            [
                v.valence,
                v.arousal,
                v.dominance,
                emotion_scores.get("happy", 0.0),
                emotion_scores.get("sad", 0.0),
                emotion_scores.get("angry", 0.0),
                emotion_scores.get("nervous", 0.0),
                emotion_scores.get("surprised", 0.0),
                emotion_scores.get("disgust", 0.0),
                emotion_scores.get("confident", 0.0),
                emotion_scores.get("interested", 0.0),
                emotion_scores.get("hesitant", 0.0),
            ],
            dtype=np.float32,
        )

    def _temporal_smoothing(self, current: np.ndarray) -> np.ndarray:
        """时序平滑"""
        if not self.frame_buffer:
            smoothed = current
        else:
            alpha = 0.3
            smoothed = alpha * current + (1 - alpha) * self.frame_buffer[-1]

        self.frame_buffer.append(smoothed)
        if len(self.frame_buffer) > self.smoothing_frames:
            self.frame_buffer.pop(0)

        return smoothed

    def _update_state_variables(self, emotion_vector: np.ndarray):
        """更新状态变量"""
        self.valence = float(emotion_vector[0])
        self.arousal = float(emotion_vector[1])
        self.dominance = float(emotion_vector[2])

        self.emotion_intensity = float(np.max(emotion_vector[3:]))

        new_state, confidence = self._infer_state(emotion_vector)

        if new_state != self.current_state and confidence > 0.7:
            self.current_state = new_state
            self.state_confidence = confidence
        elif confidence > self.state_confidence:
            self.state_confidence = confidence

    def _infer_state(self, emotion_vector: np.ndarray) -> Tuple[str, float]:
        """从情感向量推断离散状态"""
        valence, arousal, dominance = (
            emotion_vector[0],
            emotion_vector[1],
            emotion_vector[2],
        )
        happiness = emotion_vector[3]
        sadness = emotion_vector[4]
        anger = emotion_vector[5]
        fear = emotion_vector[6]
        contempt = emotion_vector[9]
        confusion = emotion_vector[11]

        rules = []

        if valence > 0.3 and dominance > 0.5:
            rules.append(("confident", valence * dominance))
        if arousal > 0.6 and valence < 0.2:
            rules.append(("nervous", arousal * (1 - valence)))
        if valence < -0.2 and arousal > 0.5 and dominance < 0.5:
            rules.append(("angry", -valence * arousal))
        if happiness > 0.6:
            rules.append(("happy", happiness))
        if sadness > 0.5 and arousal < 0.4:
            rules.append(("sad", sadness))
        if contempt > 0.4 and dominance > 0.5:
            rules.append(("contemptuous", contempt * dominance))
        if confusion > 0.4:
            rules.append(("confused", confusion))

        if not rules:
            return "neutral", 0.5

        best_state, best_score = max(rules, key=lambda x: x[1])
        confidence = min(0.95, best_score)

        return best_state, confidence

    def _compute_face_confidence(self, f: MicroExpressionFeatures) -> float:
        """计算表情特征置信度"""
        confidence = 0.5
        confidence += 1 - f.brow_tension * 0.1
        confidence += f.eye_openness_left * 0.2
        confidence = max(0.1, min(0.95, confidence / 2))
        return confidence

    def _compute_voice_confidence(self, v: VoiceEmotionFeatures) -> float:
        """计算语音特征置信度"""
        confidence = 0.5
        confidence += 1 - v.noise_ratio * 3
        confidence += 1 - v.jitter * 10
        confidence = max(0.1, min(0.95, confidence / 2))
        return confidence

    def _detect_inconsistencies(
        self,
        face: Optional[MicroExpressionFeatures],
        voice: Optional[VoiceEmotionFeatures],
        text: Optional[Dict],
    ) -> List[Dict]:
        """检测多模态不一致性"""
        inconsistencies = []

        if not face or not voice:
            return inconsistencies

        face_happy = face.happiness > 0.6
        voice_sad = voice.emotion_scores.get("sad", 0) > 0.5
        if face_happy and voice_sad:
            inconsistencies.append(
                {
                    "type": "emotion_mismatch",
                    "description": "表情开心但语音透露悲伤",
                    "severity": "high",
                    "interpretation": "可能在强颜欢笑或言不由衷",
                }
            )

        face_calm = face.nervousness < 0.3 if hasattr(face, "nervousness") else True
        voice_nervous = voice.emotion_scores.get("nervous", 0) > 0.5
        if face_calm and voice_nervous:
            inconsistencies.append(
                {
                    "type": "emotion_mismatch",
                    "description": "表情平静但声音紧张",
                    "severity": "medium",
                    "interpretation": "内心紧张但努力保持镇定",
                }
            )

        face_angry = face.anger > 0.5
        voice_hesitant = voice.emotion_scores.get("hesitant", 0) > 0.4
        if face_angry and voice_hesitant:
            inconsistencies.append(
                {
                    "type": "emotion_mismatch",
                    "description": "表情愤怒但声音犹豫",
                    "severity": "medium",
                    "interpretation": "色厉内荏，外强中干",
                }
            )

        return inconsistencies

    def _infer_hidden_sentiment(self) -> Optional[str]:
        """推断隐藏情感"""
        if not self.inconsistencies:
            return None

        for inc in self.inconsistencies:
            if "interpretation" in inc:
                return inc.get("interpretation")

        return None

    def _record_state_history(self):
        """记录状态历史"""
        self.state_history.append(
            (self.current_state, self.state_confidence, time.time())
        )
        if len(self.state_history) > 100:
            self.state_history.pop(0)

    def get_history(self, last_n: int = 10) -> List[Dict]:
        """获取最近N个状态"""
        history = self.state_history[-last_n:]
        return [{"state": s, "confidence": c, "timestamp": t} for s, c, t in history]

    def get_trend(self) -> str:
        """获取情感趋势"""
        if len(self.state_history) < 3:
            return "stable"

        recent = [s for s, _, _ in self.state_history[-5:]]

        stress_count = sum(1 for s in recent if s in ["nervous", "angry"])
        happy_count = sum(1 for s in recent if s in ["happy", "confident"])

        if stress_count >= 3:
            return "increasing_stress"
        elif happy_count >= 3:
            return "improving"
        else:
            return "stable"


class EmotionMemory:
    """情感记忆模块 - 存储用户的历史情感状态"""

    def __init__(self, max_memory: int = 50):
        self.max_memory = max_memory
        self.memories: List[Dict] = []

    def store(
        self,
        user_input: str,
        multimodal_state: MultimodalEmotionState,
        npc_response: str,
        outcome: Optional[str] = None,
    ):
        """存储交互记忆"""
        memory = {
            "user_input": user_input,
            "emotion_state": multimodal_state.to_dict(),
            "npc_response": npc_response,
            "outcome": outcome,
            "timestamp": time.time(),
        }

        self.memories.append(memory)

        if len(self.memories) > self.max_memory:
            self.memories.pop(0)

    def retrieve(
        self,
        current_input: str,
        current_emotion: MultimodalEmotionState,
        top_k: int = 3,
    ) -> List[Dict]:
        """检索相关记忆"""
        if not self.memories:
            return []

        current_state = current_emotion.primary_emotion

        scored_memories = []
        for mem in self.memories:
            mem_emotion = mem.get("emotion_state", {}).get("primaryEmotion", "neutral")

            score = 0.0

            if mem_emotion == current_state:
                score += 1.0

            if current_emotion.primary_emotion in ["nervous", "angry"]:
                if mem_emotion in ["nervous", "angry"]:
                    score += 0.5

            if current_emotion.inconsistencies and mem.get("emotion_state", {}).get(
                "inconsistencies"
            ):
                score += 0.3

            scored_memories.append((mem, score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return [mem for mem, score in scored_memories[:top_k] if score > 0]

    def get_emotion_patterns(self) -> Dict:
        """获取用户情感模式"""
        if not self.memories:
            return {"pattern": "unknown", "stability": 0.0}

        emotions = [
            m.get("emotion_state", {}).get("primaryEmotion", "neutral")
            for m in self.memories
        ]

        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        stability = most_common[1] / len(emotions)

        return {
            "pattern": most_common[0],
            "stability": stability,
            "counts": emotion_counts,
        }
