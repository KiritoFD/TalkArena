"""
多模态融合分析模块
处理表情特征和语音特征的分析，支持实时融合
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger("TalkArena")


class EmotionType(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    TIRED = "tired"


class VoiceState(Enum):
    CALM = "calm"
    EXCITED = "excited"
    HESITANT = "hesitant"
    AGITATED = "agitated"
    NEUTRAL = "neutral"


@dataclass
class EmotionFeatures:
    """表情特征数据结构"""

    eye_openness: float = 0.0
    smile_score: float = 0.0
    brow_raise: float = 0.0
    symmetry: float = 1.0
    looking_at_camera: float = 0.0
    confidence: float = 50.0
    nervousness: float = 50.0
    dominant_emotion: str = "neutral"
    head_pose: Optional[Dict] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "EmotionFeatures":
        """从字典创建"""
        return cls(
            eye_openness=data.get("eyeOpenness", 0.0),
            smile_score=data.get("smileScore", 0.0),
            brow_raise=data.get("browRaise", 0.0),
            symmetry=data.get("symmetry", 1.0),
            looking_at_camera=data.get("lookingAtCamera", 0.0),
            confidence=data.get("confidence", 50.0),
            nervousness=data.get("nervousness", 50.0),
            dominant_emotion=data.get("dominantEmotion", "neutral"),
            head_pose=data.get("headPose"),
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "eyeOpenness": self.eye_openness,
            "smileScore": self.smile_score,
            "browRaise": self.brow_raise,
            "symmetry": self.symmetry,
            "lookingAtCamera": self.looking_at_camera,
            "confidence": self.confidence,
            "nervousness": self.nervousness,
            "dominantEmotion": self.dominant_emotion,
            "headPose": self.head_pose,
        }


@dataclass
class VoiceFeatures:
    """语音特征数据结构"""

    speech_rate: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    volume_variance: float = 0.0
    pause_frequency: float = 0.0
    energy_pattern: str = "stable"
    voice_confidence: float = 50.0
    voice_nervousness: float = 50.0
    emotion_label: str = "neutral"

    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceFeatures":
        """从字典创建"""
        return cls(
            speech_rate=data.get("speechRate", 0.0),
            pitch_mean=data.get("pitchMean", 0.0),
            pitch_std=data.get("pitchStd", 0.0),
            volume_variance=data.get("volumeVariance", 0.0),
            pause_frequency=data.get("pauseFrequency", 0.0),
            energy_pattern=data.get("energyPattern", "stable"),
            voice_confidence=data.get("voiceConfidence", 50.0),
            voice_nervousness=data.get("voiceNervousness", 50.0),
            emotion_label=data.get("emotionLabel", "neutral"),
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "speechRate": self.speech_rate,
            "pitchMean": self.pitch_mean,
            "pitchStd": self.pitch_std,
            "volumeVariance": self.volume_variance,
            "pauseFrequency": self.pause_frequency,
            "energyPattern": self.energy_pattern,
            "voiceConfidence": self.voice_confidence,
            "voiceNervousness": self.voice_nervousness,
            "emotionLabel": self.emotion_label,
        }


class MultimodalAnalyzer:
    """多模态分析器 - 分析表情和语音特征"""

    def __init__(self):
        self.emotion_icons = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "surprised": "😲",
            "nervous": "😰",
            "tired": "😴",
            "neutral": "😐",
            "confident": "😎",
            "calm": "😌",
            "excited": "🤩",
            "hesitant": "🤔",
            "agitated": "😤",
        }

        self.voice_icons = {
            "calm": "🎵",
            "excited": "🎸",
            "hesitant": "📢",
            "agitated": "🥁",
            "neutral": "🎤",
        }

    def analyze_emotion(self, features: EmotionFeatures) -> Dict:
        """分析表情特征，返回评估结果"""
        # 计算综合得分
        emotion_score = 50.0

        # 自信度贡献
        emotion_score += (features.confidence - 50) * 0.3

        # 紧张度惩罚
        emotion_score -= (features.nervousness - 50) * 0.3

        # 表情自然度
        if features.symmetry > 0.7:
            emotion_score += 10

        # 眼神交流
        if features.looking_at_camera > 0.6:
            emotion_score += 10

        emotion_score = max(0, min(100, emotion_score))

        # 生成反馈
        feedback = self._generate_emotion_feedback(features)

        # 获取表情图标
        emotion_icon = self.emotion_icons.get(
            features.dominant_emotion, self.emotion_icons["neutral"]
        )

        return {
            "score": round(emotion_score, 1),
            "dominant_emotion": features.dominant_emotion,
            "emotion_icon": emotion_icon,
            "confidence": round(features.confidence, 1),
            "nervousness": round(features.nervousness, 1),
            "feedback": feedback,
            "raw_features": features.to_dict(),
        }

    def analyze_voice(self, features: VoiceFeatures) -> Dict:
        """分析语音特征，返回评估结果"""
        # 计算语音得分
        voice_score = 50.0

        # 自信度贡献
        voice_score += (features.voice_confidence - 50) * 0.4

        # 紧张度惩罚
        voice_score -= (features.voice_nervousness - 50) * 0.4

        # 语速适中加分
        if 2.0 <= features.speech_rate <= 4.0:
            voice_score += 10

        # 音调稳定加分
        if features.pitch_std < 25:
            voice_score += 10

        voice_score = max(0, min(100, voice_score))

        # 生成反馈
        feedback = self._generate_voice_feedback(features)

        # 获取语音图标
        voice_icon = self.voice_icons.get(
            features.emotion_label, self.voice_icons["neutral"]
        )

        return {
            "score": round(voice_score, 1),
            "emotion_label": features.emotion_label,
            "voice_icon": voice_icon,
            "voice_confidence": round(features.voice_confidence, 1),
            "voice_nervousness": round(features.voice_nervousness, 1),
            "speech_rate": round(features.speech_rate, 1),
            "feedback": feedback,
            "raw_features": features.to_dict(),
        }

    def analyze_multimodal(
        self,
        text: str,
        emotion_features: Optional[EmotionFeatures] = None,
        voice_features: Optional[VoiceFeatures] = None,
    ) -> Dict:
        """
        综合分析文本、表情和语音
        返回完整的评估结果
        """
        results = {
            "text_score": 50.0,  # 这里可以调用LLM评估文本
            "emotion_analysis": None,
            "voice_analysis": None,
            "overall_score": 50.0,
            "feedback": "",
            "inconsistencies": [],
            "suggestions": [],
        }

        weights = {"text": 0.5, "emotion": 0.25, "voice": 0.25}
        scores = {"text": 50.0, "emotion": 50.0, "voice": 50.0}

        # 分析表情
        if emotion_features:
            emotion_result = self.analyze_emotion(emotion_features)
            results["emotion_analysis"] = emotion_result
            scores["emotion"] = emotion_result["score"]
        else:
            # 如果没有表情数据，降低表情权重
            weights["text"] += weights["emotion"] * 0.5
            weights["voice"] += weights["emotion"] * 0.5
            weights["emotion"] = 0

        # 分析语音
        if voice_features:
            voice_result = self.analyze_voice(voice_features)
            results["voice_analysis"] = voice_result
            scores["voice"] = voice_result["score"]
        else:
            # 如果没有语音数据，降低语音权重
            weights["text"] += weights["voice"]
            weights["voice"] = 0

        # 计算综合得分
        overall_score = sum(scores[k] * weights[k] for k in scores)
        results["overall_score"] = round(overall_score, 1)
        results["breakdown"] = {
            "text": round(scores["text"], 1),
            "emotion": round(scores["emotion"], 1),
            "voice": round(scores["voice"], 1),
        }

        # 检测不一致性
        inconsistencies = self._detect_inconsistencies(
            text, emotion_features, voice_features
        )
        results["inconsistencies"] = inconsistencies

        # 生成综合反馈
        results["feedback"] = self._generate_overall_feedback(results, inconsistencies)

        # 生成建议
        results["suggestions"] = self._generate_suggestions(
            emotion_features, voice_features
        )

        return results

    def _generate_emotion_feedback(self, features: EmotionFeatures) -> str:
        """生成表情反馈"""
        feedbacks = []

        if features.nervousness > 70:
            feedbacks.append("表情略显紧张")
        elif features.nervousness < 30:
            feedbacks.append("神态自若")

        if features.confidence > 70:
            feedbacks.append("自信满满")
        elif features.confidence < 40:
            feedbacks.append("可以更自信些")

        if features.looking_at_camera > 0.7:
            feedbacks.append("眼神交流充分")
        elif features.looking_at_camera < 0.3:
            feedbacks.append("建议多进行眼神交流")

        if features.symmetry < 0.6:
            feedbacks.append("面部表情自然度可以提升")

        return " | ".join(feedbacks) if feedbacks else "表情管理到位"

    def _generate_voice_feedback(self, features: VoiceFeatures) -> str:
        """生成语音反馈"""
        feedbacks = []

        if features.voice_nervousness > 70:
            feedbacks.append("声音略显紧张")
        elif features.voice_nervousness < 30:
            feedbacks.append("声音沉稳")

        if features.voice_confidence > 70:
            feedbacks.append("语气坚定有力")

        if features.speech_rate > 4.5:
            feedbacks.append("语速偏快")
        elif features.speech_rate < 2.0:
            feedbacks.append("语速偏慢")
        else:
            feedbacks.append("语速适中")

        if features.pitch_std > 30:
            feedbacks.append("音调波动较大")
        elif features.pitch_std < 15:
            feedbacks.append("音调平稳")

        if features.pause_frequency > 0.2:
            feedbacks.append("停顿较多")

        return " | ".join(feedbacks) if feedbacks else "语气得当"

    def _detect_inconsistencies(
        self,
        text: str,
        emotion_features: Optional[EmotionFeatures],
        voice_features: Optional[VoiceFeatures],
    ) -> list:
        """检测多模态不一致性"""
        inconsistencies = []

        # 检测1: 文本开心但表情不笑
        if emotion_features and ("开心" in text or "高兴" in text or "谢谢" in text):
            if emotion_features.smile_score < 0.2:
                inconsistencies.append("嘴上说开心，但表情未见笑容")

        # 检测2: 文本强硬但声音颤抖
        if voice_features and any(
            word in text for word in ["必须", "一定", "肯定", "没错"]
        ):
            if voice_features.voice_nervousness > 60:
                inconsistencies.append("话语强硬但声音略显紧张")

        # 检测3: 文本谦虚但表情傲慢
        if emotion_features and any(
            word in text for word in ["不敢", "惭愧", "过奖", "哪里"]
        ):
            if emotion_features.brow_raise > 0.7:
                inconsistencies.append("嘴上谦虚但表情显得高傲")

        # 检测4: 表情紧张但声音自信
        if emotion_features and voice_features:
            if (
                emotion_features.nervousness > 60
                and voice_features.voice_confidence > 70
            ):
                inconsistencies.append("表情紧张但声音很沉稳")

        return inconsistencies

    def _generate_overall_feedback(self, results: Dict, inconsistencies: list) -> str:
        """生成综合反馈"""
        score = results["overall_score"]

        if score >= 80:
            base_feedback = "表现出色！气场很足"
        elif score >= 60:
            base_feedback = "表现不错，还有提升空间"
        elif score >= 40:
            base_feedback = "表现一般，需要多加练习"
        else:
            base_feedback = "建议调整心态，放松一些"

        # 添加不一致性警告
        if inconsistencies:
            base_feedback += f"（注意：{inconsistencies[0]}）"

        return base_feedback

    def _generate_suggestions(
        self,
        emotion_features: Optional[EmotionFeatures],
        voice_features: Optional[VoiceFeatures],
    ) -> list:
        """生成改进建议"""
        suggestions = []

        if emotion_features:
            if emotion_features.nervousness > 60:
                suggestions.append("🎯 建议：对着镜子练习微笑，深呼吸放松")
            if emotion_features.looking_at_camera < 0.4:
                suggestions.append("👀 建议：说话时要看着对方，增强眼神交流")
            if emotion_features.symmetry < 0.7:
                suggestions.append("😊 建议：让表情更自然，放松面部肌肉")

        if voice_features:
            if voice_features.voice_nervousness > 60:
                suggestions.append("🗣️ 建议：放慢语速，用腹式呼吸稳定声音")
            if voice_features.pause_frequency > 0.2:
                suggestions.append('💬 建议：减少"嗯""啊"等口头禅')
            if voice_features.speech_rate > 4.5:
                suggestions.append("⏱️ 建议：放慢语速，给自己思考的时间")

        return suggestions

    def get_status_icons(
        self,
        emotion_features: Optional[EmotionFeatures] = None,
        voice_features: Optional[VoiceFeatures] = None,
    ) -> Dict[str, str]:
        """
        获取状态图标，用于前端展示
        返回表情图标和语音图标
        """
        icons = {
            "emotion_icon": "❓",
            "emotion_status": "未检测",
            "voice_icon": "❓",
            "voice_status": "未检测",
        }

        if emotion_features:
            emotion_icon = self.emotion_icons.get(
                emotion_features.dominant_emotion, "😐"
            )
            icons["emotion_icon"] = emotion_icon
            icons["emotion_status"] = self._get_emotion_status_text(emotion_features)

        if voice_features:
            voice_icon = self.voice_icons.get(voice_features.emotion_label, "🎤")
            icons["voice_icon"] = voice_icon
            icons["voice_status"] = self._get_voice_status_text(voice_features)

        return icons

    def _get_emotion_status_text(self, features: EmotionFeatures) -> str:
        """获取表情状态文本"""
        emotion_map = {
            "happy": "开心",
            "sad": "难过",
            "angry": "生气",
            "surprised": "惊讶",
            "nervous": "紧张",
            "tired": "疲惫",
            "neutral": "平静",
            "confident": "自信",
        }

        emotion_text = emotion_map.get(
            features.dominant_emotion, features.dominant_emotion
        )

        # 添加自信度/紧张度提示
        if features.confidence > 70:
            return f"{emotion_text}·自信"
        elif features.nervousness > 60:
            return f"{emotion_text}·紧张"

        return emotion_text

    def _get_voice_status_text(self, features: VoiceFeatures) -> str:
        """获取语音状态文本"""
        emotion_map = {
            "calm": "沉稳",
            "excited": "激动",
            "hesitant": "犹豫",
            "agitated": "焦躁",
            "neutral": "平和",
        }

        voice_text = emotion_map.get(features.emotion_label, features.emotion_label)

        # 添加语速提示
        if features.speech_rate > 4.5:
            return f"{voice_text}·偏快"
        elif features.speech_rate < 2.0:
            return f"{voice_text}·偏慢"

        return voice_text
