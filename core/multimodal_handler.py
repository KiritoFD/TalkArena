"""
多模态输入处理模块
处理包含表情和语音特征的游戏回合
"""

from typing import Dict, Optional, List, Generator
import json
from core.multimodal_analyzer import MultimodalAnalyzer
from core.emotion_state import MicroExpressionFeatures, VoiceEmotionFeatures


class MultimodalInputHandler:
    """多模态输入处理器"""

    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.analyzer = MultimodalAnalyzer()

    def process_multimodal_turn(
        self,
        session_id: str,
        text: str,
        emotion_features_dict: Optional[Dict] = None,
        voice_features_dict: Optional[Dict] = None,
    ) -> Generator:
        """
        处理包含表情和语音特征的游戏回合

        Args:
            session_id: 会话ID
            text: 用户输入文本
            emotion_features_dict: 表情特征字典（可选）
            voice_features_dict: 语音特征字典（可选）

        Yields:
            包含多模态分析结果的更新
        """
        # 解析特征
        emotion_features = None
        voice_features = None

        if emotion_features_dict:
            try:
                emotion_features = MicroExpressionFeatures.from_dict(
                    emotion_features_dict
                )
            except Exception as e:
                print(f"[Multimodal] 解析表情特征失败: {e}")

        if voice_features_dict:
            try:
                voice_features = VoiceEmotionFeatures.from_dict(voice_features_dict)
            except Exception as e:
                print(f"[Multimodal] 解析语音特征失败: {e}")

        # 多模态分析
        emotion_dict = emotion_features.to_dict() if emotion_features else None
        voice_dict = voice_features.to_dict() if voice_features else None
        multimodal_result = self.analyzer.analyze_multimodal(
            text=text, emotion_features=emotion_dict, voice_features=voice_dict
        )

        # 获取状态图标
        status_icons = self.analyzer.get_status_icons()

        # 构建结果摘要
        result_summary = {
            "overall_score": multimodal_result["overall_score"],
            "breakdown": multimodal_result["breakdown"],
            "feedback": multimodal_result["feedback"],
            "inconsistencies": multimodal_result["inconsistencies"],
            "suggestions": multimodal_result["suggestions"],
            "emotion_analysis": multimodal_result["emotion_analysis"],
            "voice_analysis": multimodal_result["voice_analysis"],
            "status_icons": status_icons,
        }

        yield {"stage": "multimodal_analysis", "result": result_summary}

        # 返回分析结果字符串（用于显示）
        judgment = self._format_judgment(multimodal_result, status_icons)

        yield {
            "stage": "multimodal_complete",
            "judgment": judgment,
            "result": result_summary,
        }

    def _format_judgment(self, result: Dict, status_icons: Dict) -> str:
        """格式化评估结果为字符串"""
        lines = []

        # 总体评分
        lines.append(f"📊 综合评分: {result['overall_score']:.0f}/100")

        # 分项得分
        breakdown = result["breakdown"]
        lines.append(
            f"   文本: {breakdown['text']:.0f} | 表情: {breakdown['emotion']:.0f} | 语音: {breakdown['voice']:.0f}"
        )

        # 状态图标
        lines.append(
            f"   表情状态: {status_icons['emotion_icon']} {status_icons['emotion_status']}"
        )
        lines.append(
            f"   语音状态: {status_icons['voice_icon']} {status_icons['voice_status']}"
        )

        # 反馈
        lines.append(f"   💡 {result['feedback']}")

        # 不一致性警告
        if result["inconsistencies"]:
            lines.append(f"   ⚠️ {' | '.join(result['inconsistencies'])}")

        # 建议
        if result["suggestions"]:
            lines.append(f"   📝 {result['suggestions'][0]}")

        return "\n".join(lines)

    def quick_analyze(
        self,
        emotion_features_dict: Optional[Dict] = None,
        voice_features_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        快速分析，返回状态图标（用于实时展示）

        Returns:
            包含状态图标的字典
        """
        emotion_features = None
        voice_features = None

        if emotion_features_dict:
            try:
                emotion_features = EmotionFeatures.from_dict(emotion_features_dict)
            except:
                pass

        if voice_features_dict:
            try:
                voice_features = VoiceFeatures.from_dict(voice_features_dict)
            except:
                pass

        return self.analyzer.get_status_icons(
            emotion_features=emotion_features, voice_features=voice_features
        )
