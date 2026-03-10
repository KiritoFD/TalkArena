"""
升级版多模态融合分析模块
支持微表情分析和语音情感分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from core.emotion_state import (
    MicroExpressionFeatures,
    VoiceEmotionFeatures,
    MultimodalEmotionState,
    UserEmotionStateMachine,
    EmotionMemory,
)

logger = logging.getLogger("TalkArena")


class VoiceEmotionAnalyzer:
    """语音情感分析器"""

    def __init__(self):
        self.sample_rate = 16000

    def analyze(self, audio_data: np.ndarray) -> VoiceEmotionFeatures:
        """分析语音情感特征"""

        features = VoiceEmotionFeatures()

        features.loudness = float(np.mean(np.abs(audio_data)))

        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
        speech_rate = (zero_crossings / self.sample_rate) * 60
        features.speech_rate = min(6.0, max(1.0, speech_rate / 100))

        features.energy_variance = float(np.var(np.abs(audio_data)))

        features.emotion_scores = self._infer_emotion(features)

        features.valence = self._compute_valence(features)
        features.arousal = self._compute_arousal(features)
        features.dominance = self._compute_dominance(features)

        return features

    def _infer_emotion(self, features: VoiceEmotionFeatures) -> Dict[str, float]:
        """推断语音情感"""
        scores = {}

        loudness = features.loudness
        speech_rate = features.speech_rate
        energy_var = features.energy_variance

        scores["happy"] = (
            min(1.0, loudness * 1.2) * 0.3 + min(1.0, speech_rate / 4.0) * 0.3
        )
        scores["sad"] = (1 - loudness) * 0.4 + (1 - min(1.0, speech_rate / 3.0)) * 0.3
        scores["angry"] = min(1.0, loudness) * 0.4 + min(1.0, energy_var * 10) * 0.3
        scores["nervous"] = (
            min(1.0, energy_var * 15) * 0.5 + (1 - min(1.0, speech_rate / 2.5)) * 0.3
        )
        scores["confident"] = (1 - energy_var * 8) * 0.4 + min(
            1.0, speech_rate / 3.5
        ) * 0.3
        scores["hesitant"] = (1 - loudness) * 0.3 + min(1.0, energy_var * 8) * 0.4

        for k in scores:
            scores[k] = max(0.0, min(1.0, scores[k]))

        return scores

    def _compute_valence(self, f: VoiceEmotionFeatures) -> float:
        scores = f.emotion_scores or {}
        return (scores.get("happy", 0) - scores.get("sad", 0)) * 0.5

    def _compute_arousal(self, f: VoiceEmotionFeatures) -> float:
        scores = f.emotion_scores or {}
        return max(
            scores.get("angry", 0),
            scores.get("nervous", 0),
            scores.get("happy", 0) * 0.8,
        )

    def _compute_dominance(self, f: VoiceEmotionFeatures) -> float:
        scores = f.emotion_scores or {}
        return (
            scores.get("confident", 0.5) * 0.7 + (1 - scores.get("hesitant", 0)) * 0.3
        )


class MultimodalFusionEngine:
    """多模态融合引擎"""

    def __init__(self):
        self.state_machine = UserEmotionStateMachine()
        self.emotion_memory = EmotionMemory()
        self.voice_analyzer = VoiceEmotionAnalyzer()

    def process_face_features(self, face_data: Dict) -> MultimodalEmotionState:
        """处理表情特征"""
        face_features = MicroExpressionFeatures.from_dict(face_data)

        state = self.state_machine.update(face_features=face_features)

        return state

    def process_voice_features(self, voice_data: Dict) -> MultimodalEmotionState:
        """处理语音特征"""
        voice_features = VoiceEmotionFeatures.from_dict(voice_data)

        state = self.state_machine.update(voice_features=voice_features)

        return state

    def fuse(
        self,
        face_data: Optional[Dict] = None,
        voice_data: Optional[Dict] = None,
        text: str = "",
    ) -> MultimodalEmotionState:
        """融合多模态特征"""

        face_features = None
        voice_features = None

        if face_data:
            face_features = MicroExpressionFeatures.from_dict(face_data)

        if voice_data:
            voice_features = VoiceEmotionFeatures.from_dict(voice_data)

        text_sentiment = self._analyze_text_sentiment(text) if text else None

        state = self.state_machine.update(
            face_features=face_features,
            voice_features=voice_features,
            text_sentiment=text_sentiment,
        )

        return state

    def _analyze_text_sentiment(self, text: str) -> Dict:
        """简单的文本情感分析"""
        text_lower = text.lower()

        positive_words = ["开心", "高兴", "谢谢", "好的", "可以", "没问题"]
        negative_words = ["不行", "不要", "生气", "愤怒", "难过", "紧张"]

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        if positive_count > negative_count:
            return {"sentiment": "positive", "confidence": 0.6}
        elif negative_count > positive_count:
            return {"sentiment": "negative", "confidence": 0.6}

        return {"sentiment": "neutral", "confidence": 0.3}

    def store_interaction(
        self,
        user_input: str,
        multimodal_state: MultimodalEmotionState,
        npc_response: str,
    ):
        """存储交互到记忆"""
        self.emotion_memory.store(
            user_input=user_input,
            multimodal_state=multimodal_state,
            npc_response=npc_response,
        )

    def get_related_memories(
        self, current_state: MultimodalEmotionState, top_k: int = 3
    ) -> List[Dict]:
        """获取相关记忆"""
        return self.emotion_memory.retrieve(
            current_input="", current_emotion=current_state, top_k=top_k
        )

    def get_emotion_patterns(self) -> Dict:
        """获取用户情感模式"""
        return self.emotion_memory.get_emotion_patterns()

    def get_trend(self) -> str:
        """获取情感趋势"""
        return self.state_machine.get_trend()

    def get_history(self, last_n: int = 10) -> List[Dict]:
        """获取情感历史"""
        return self.state_machine.get_history(last_n)


class EmotionDrivenResponseGenerator:
    """情感驱动响应生成器 - 为NPC生成行为指令"""

    def __init__(self):
        self.behavior_templates = self._load_templates()

    def _load_templates(self) -> Dict:
        return {
            "confident": {
                "facial": "严肃直视",
                "eye_contact": "坚定",
                "body_posture": "前倾",
                "voice_tone": "有力",
                "gesture": "指点",
            },
            "nervous": {
                "facial": "关切",
                "eye_contact": "温和",
                "body_posture": "放松",
                "voice_tone": "缓和",
                "gesture": "安抚",
            },
            "angry": {
                "facial": "惊讶",
                "eye_contact": "关切",
                "body_posture": "暂停",
                "voice_tone": "缓和",
                "gesture": "举手",
            },
            "happy": {
                "facial": "开心",
                "eye_contact": "明亮",
                "body_posture": "放松",
                "voice_tone": "轻快",
                "gesture": "点头",
            },
            "sad": {
                "facial": "同情",
                "eye_contact": "温柔",
                "body_posture": "前倾",
                "voice_tone": "温和",
                "gesture": "轻拍",
            },
            "neutral": {
                "facial": "自然",
                "eye_contact": "正常",
                "body_posture": "自然",
                "voice_tone": "正常",
                "gesture": "无",
            },
        }

    def generate_behavior_cues(
        self, emotion_state: MultimodalEmotionState, npc_personality: str = "default"
    ) -> Dict:
        """生成行为提示"""

        primary = emotion_state.primary_emotion
        template = self.behavior_templates.get(
            primary, self.behavior_templates["neutral"]
        )

        intensity = emotion_state.emotion_intensity

        cues = {
            "facial_expression": template["facial"],
            "eye_contact": template["eye_contact"],
            "body_language": template["body_posture"],
            "voice_tone": template["voice_tone"],
            "hand_gesture": template["gesture"],
            "intensity": intensity,
            "emotion": primary,
            "hidden_sentiment": emotion_state.hidden_sentiment,
            "confidence": emotion_state.confidence,
        }

        if emotion_state.inconsistencies:
            cues["inconsistencies"] = emotion_state.inconsistencies

        return cues

    def get_npc_strategy(
        self, user_emotion: MultimodalEmotionState, npc_role: str = "aggressor"
    ) -> Dict:
        """获取NPC策略建议"""

        emotion = user_emotion.primary_emotion

        strategies = {
            "confident": {
                "aggressor": {
                    "tactic": "defensive_counter",
                    "description": "用户自信，要更加谨慎，增加难度",
                    "tone": "认真",
                    "emotional_tone": "严肃",
                },
                "supporter": {
                    "tactic": "observation",
                    "description": "观察学习",
                    "tone": "温和",
                    "emotional_tone": "中性",
                },
            },
            "nervous": {
                "aggressor": {
                    "tactic": "continual_attack",
                    "description": "用户紧张，继续施压",
                    "tone": "严厉",
                    "emotional_tone": "强势",
                },
                "supporter": {
                    "tactic": "give_way",
                    "description": "适当给台阶",
                    "tone": "温和",
                    "emotional_tone": "关切",
                },
            },
            "angry": {
                "aggressor": {
                    "tactic": "de_escalation",
                    "description": "用户愤怒，适当收敛",
                    "tone": "缓和",
                    "emotional_tone": "收敛",
                },
                "supporter": {
                    "tactic": "support",
                    "description": "支持安抚",
                    "tone": "温和",
                    "emotional_tone": "关心",
                },
            },
        }

        default = {
            "tactic": "normal",
            "description": "正常应对",
            "tone": "自然",
            "emotional_tone": "中性",
        }

        return strategies.get(emotion, {}).get(npc_role, default)


class MultimodalAnalyzer:
    """主多模态分析器 - 兼容旧接口"""

    def __init__(self):
        self.fusion_engine = MultimodalFusionEngine()
        self.response_generator = EmotionDrivenResponseGenerator()

    def analyze_multimodal(
        self,
        text: str = "",
        emotion_features: Optional[Dict] = None,
        voice_features: Optional[Dict] = None,
    ) -> Dict:
        """分析多模态输入"""

        state = self.fusion_engine.fuse(
            face_data=emotion_features, voice_data=voice_features, text=text
        )

        behavior_cues = self.response_generator.generate_behavior_cues(state)

        patterns = self.fusion_engine.get_emotion_patterns()
        trend = self.fusion_engine.get_trend()

        return {
            "emotion_state": state.to_dict(),
            "behavior_cues": behavior_cues,
            "patterns": patterns,
            "trend": trend,
            "inconsistencies": state.inconsistencies,
        }

    def process_turn(self, user_input: str, multimodal_data: Dict) -> Dict:
        """处理一轮交互"""

        emotion_data = multimodal_data.get("emotion", {})
        voice_level = multimodal_data.get("voice_level", 0)
        voice_features_dict = multimodal_data.get("voice_features")

        face_data = None
        if emotion_data:
            face_data = {
                "confidence": emotion_data.get("confidence", 0.5),
                "nervousness": emotion_data.get("nervous", 0.5),
                "calm": emotion_data.get("calm", 0.5),
                "focus": emotion_data.get("focus", 0.5),
                "happiness": emotion_data.get("confidence", 0.5) * 0.3,
                "sadness": emotion_data.get("nervous", 0.5) * 0.3,
                "anger": 0.0,
                "valence": emotion_data.get("confidence", 0.5) - 0.5,
                "arousal": emotion_data.get("nervous", 0.5),
                "dominance": emotion_data.get("confidence", 0.5),
                "smileGenuineScore": emotion_data.get("confidence", 0.5) * 0.3,
                "browTension": emotion_data.get("nervous", 0.5) * 0.3,
            }

        voice_data = None
        if voice_features_dict:
            voice_data = voice_features_dict
        elif voice_level > 0:
            voice_data = {
                "loudness": voice_level / 100.0,
                "speechRate": 3.0,
                "pitchMean": 150.0,
                "pitchStd": 20.0,
                "energyVariance": 0.1,
                "emotionScores": {
                    "nervous": (100 - voice_level) / 100.0 * 0.3,
                    "confident": voice_level / 100.0 * 0.5,
                },
            }

        result = self.analyze_multimodal(
            text=user_input, emotion_features=face_data, voice_features=voice_data
        )

        return result

    def store_memory(self, user_input: str, multimodal_data: Dict, npc_response: str):
        """存储交互记忆"""

        state = self.fusion_engine.state_machine.current_state

        from core.emotion_state import MultimodalEmotionState

        temp_state = MultimodalEmotionState(
            primary_emotion=state,
            emotion_intensity=0.5,
            valence=0.0,
            arousal=0.5,
            dominance=0.5,
        )

        self.fusion_engine.store_interaction(user_input, temp_state, npc_response)

    def get_status_icons(
        self, emotion_features=None, voice_features=None
    ) -> Dict[str, str]:
        """获取状态图标"""

        state = self.fusion_engine.state_machine

        emotion_icons = {
            "confident": "😎",
            "nervous": "😰",
            "angry": "😠",
            "happy": "😊",
            "sad": "😢",
            "surprised": "😲",
            "confused": "😕",
            "contemptuous": "🙄",
            "neutral": "😐",
        }

        return {
            "emotion_icon": emotion_icons.get(state.current_state, "😐"),
            "emotion_status": state.current_state,
            "confidence": f"{state.state_confidence:.0%}",
        }
