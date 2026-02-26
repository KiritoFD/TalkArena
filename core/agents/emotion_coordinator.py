"""
情感协调Agent
负责全局情感状态管理、多NPC协调、行为策略规划
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("TalkArena")


@dataclass
class UserSentiment:
    """用户情感态势"""

    emotion_type: str
    pressure_strategy: str
    sympathy_level: float
    hidden_sentiment: Optional[str] = None
    intensity: float = 0.5
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5


@dataclass
class EmotionTarget:
    """NPC情感目标"""

    target_emotion: str
    intensity: float
    is_active: bool
    speaking_priority: float


@dataclass
class CoordinationStrategy:
    """协调策略"""

    tactic: str
    description: str
    roles: Dict[str, str]
    timing: str


@dataclass
class BehaviorInstruction:
    """行为指令"""

    target_role: str
    tactic: str
    description: str
    facial_expression: str = "自然"
    eye_contact: str = "正常"
    voice_tone: str = "自然"
    gesture: str = "无"
    thought_bubble: Optional[str] = None


class EmotionCoordinatorAgent:
    """情感协调Agent"""

    def __init__(self, llm=None):
        self.llm = llm
        self.npc_emotion_models: Dict[str, Dict] = {}
        self.coordination_history: List[Dict] = []

    def coordinate(
        self,
        user_multimodal_state: Dict,
        dialogue_context: Dict,
        npc_configs: List[Dict],
    ) -> Dict:
        """
        协调多个NPC的情感与行为
        """

        user_sentiment = self._analyze_user_sentiment(user_multimodal_state)

        npc_states = self._get_npc_states(npc_configs)

        emotion_targets = self._assign_emotion_targets(
            user_sentiment, npc_states, dialogue_context
        )

        coordination = self._plan_coordination(
            user_sentiment, npc_states, emotion_targets
        )

        behavior_instructions = self._generate_instructions(
            user_sentiment, coordination, npc_states
        )

        return {
            "user_sentiment": {
                "type": user_sentiment.emotion_type,
                "strategy": user_sentiment.pressure_strategy,
                "sympathy": user_sentiment.sympathy_level,
                "hidden": user_sentiment.hidden_sentiment,
                "intensity": user_sentiment.intensity,
            },
            "coordination": {
                "tactic": coordination.tactic,
                "description": coordination.description,
                "timing": coordination.timing,
            },
            "behavior_instructions": {
                inst.target_role: {
                    "emotion": inst.target_emotion,
                    "facial": inst.facial_expression,
                    "eye": inst.eye_contact,
                    "tone": inst.voice_tone,
                    "gesture": inst.gesture,
                    "thought": inst.thought_bubble,
                    "priority": emotion_targets.get(
                        inst.target_role, EmotionTarget("", 0, False, 0)
                    ).speaking_priority,
                }
                for inst in behavior_instructions
            },
            "npc_understanding": self._generate_npc_understanding(
                user_sentiment, npc_configs
            ),
        }

    def _analyze_user_sentiment(self, state: Dict) -> UserSentiment:
        """分析用户情感态势"""

        primary = state.get("primaryEmotion", "neutral")
        intensity = state.get("emotionIntensity", 0.5)
        arousal = state.get("arousal", 0.5)
        valence = state.get("valence", 0.0)
        dominance = state.get("dominance", 0.5)
        hidden = state.get("hiddenSentiment")

        if primary == "nervous":
            if arousal > 0.7:
                emotion_type = "highly_stressed"
                pressure_strategy = "continual_attack"
                sympathy_level = 0.2
            elif arousal > 0.4:
                emotion_type = "moderately_stressed"
                pressure_strategy = "gradual_escalation"
                sympathy_level = 0.4
            else:
                emotion_type = "calmly_nervous"
                pressure_strategy = "observation"
                sympathy_level = 0.6

        elif primary == "confident":
            if arousal > 0.6:
                emotion_type = "aggressively_confident"
                pressure_strategy = "defensive_counter"
                sympathy_level = 0.3
            else:
                emotion_type = "calmly_confident"
                pressure_strategy = "measured_approach"
                sympathy_level = 0.5

        elif primary == "angry":
            emotion_type = "hostile"
            pressure_strategy = "de_escalation"
            sympathy_level = 0.8

        elif primary == "happy":
            emotion_type = "positive_mood"
            pressure_strategy = "maintain_momentum"
            sympathy_level = 0.7

        elif primary == "sad":
            emotion_type = "down"
            pressure_strategy = "supportive_approach"
            sympathy_level = 0.9

        else:
            emotion_type = "neutral"
            pressure_strategy = "probing"
            sympathy_level = 0.5

        return UserSentiment(
            emotion_type=emotion_type,
            pressure_strategy=pressure_strategy,
            sympathy_level=sympathy_level,
            hidden_sentiment=hidden,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )

    def _get_npc_states(self, npc_configs: List[Dict]) -> Dict[str, Dict]:
        """获取NPC状态"""
        states = {}
        for npc in npc_configs:
            name = npc.get("name", "NPC")
            role = npc.get("role", "default")
            states[name] = {
                "role": role,
                "default_emotion": npc.get("defaultEmotion", "neutral"),
                "aggression": npc.get("aggression", 0.5),
            }
        return states

    def _assign_emotion_targets(
        self,
        user_sentiment: UserSentiment,
        npc_states: Dict[str, Dict],
        dialogue_context: Dict,
    ) -> Dict[str, EmotionTarget]:
        """为NPC分配情感目标"""

        targets = {}
        current_speaker = dialogue_context.get("current_speaker", "")

        npc_list = list(npc_states.items())

        for i, (npc_name, npc_state) in enumerate(npc_list):
            is_active = (npc_name == current_speaker) or (i == 0)

            target_emotion = "neutral"
            intensity = 0.5
            priority = 0.5

            if user_sentiment.emotion_type == "highly_stressed":
                if npc_state.get("role") == "aggressor":
                    target_emotion = "dominant"
                    intensity = 0.9
                elif npc_state.get("role") == "supporter":
                    target_emotion = "sympathetic"
                    intensity = 0.6
                else:
                    target_emotion = "observing"
                    intensity = 0.4

            elif user_sentiment.emotion_type == "aggressively_confident":
                if is_active:
                    target_emotion = "defensive"
                    intensity = 0.8
                else:
                    target_emotion = "alert"
                    intensity = 0.5

            elif user_sentiment.emotion_type == "hostile":
                target_emotion = "conciliatory"
                intensity = 0.7

            elif user_sentiment.emotion_type == "down":
                target_emotion = "encouraging"
                intensity = 0.6
            else:
                target_emotion = npc_state.get("default_emotion", "neutral")
                intensity = 0.5

            if is_active:
                priority = 1.0
            else:
                priority = 0.5

            targets[npc_name] = EmotionTarget(
                target_emotion=target_emotion,
                intensity=intensity,
                is_active=is_active,
                speaking_priority=priority,
            )

        return targets

    def _plan_coordination(
        self,
        user_sentiment: UserSentiment,
        npc_states: Dict[str, Dict],
        emotion_targets: Dict[str, EmotionTarget],
    ) -> CoordinationStrategy:
        """规划协调策略"""

        strategy = user_sentiment.pressure_strategy

        if strategy == "continual_attack":
            return CoordinationStrategy(
                tactic="overwhelming_force",
                description="多NPC轮番施压，不给喘息机会",
                roles={
                    "aggressor": "主攻手",
                    "supporter": "侧面补刀",
                    "observer": "制造尴尬",
                },
                timing="快速轮转",
            )
        elif strategy == "gradual_escalation":
            return CoordinationStrategy(
                tactic="gradual_escalation",
                description="逐步提升压力，温水煮青蛙",
                roles={
                    "aggressor": "先礼后兵",
                    "supporter": "表面劝和",
                    "observer": "适时加入",
                },
                timing="每个角色2-3轮",
            )
        elif strategy == "de_escalation":
            return CoordinationStrategy(
                tactic="de_escalation",
                description="适当让步，避免冲突升级",
                roles={
                    "aggressor": "暂时收声",
                    "supporter": "打圆场",
                    "observer": "转移话题",
                },
                timing="放缓节奏",
            )
        elif strategy == "supportive_approach":
            return CoordinationStrategy(
                tactic="supportive_approach",
                description="给鼓励和认可",
                roles={
                    "aggressor": "减少施压",
                    "supporter": "正面鼓励",
                    "observer": "适时夸奖",
                },
                timing="温和互动",
            )
        else:
            return CoordinationStrategy(
                tactic="probing",
                description="试探用户底线",
                roles={
                    "aggressor": "轻量级试探",
                    "supporter": "补充问题",
                    "observer": "记录反应",
                },
                timing="慢节奏",
            )

    def _generate_instructions(
        self,
        user_sentiment: UserSentiment,
        coordination: CoordinationStrategy,
        npc_states: Dict[str, Dict],
    ) -> List[BehaviorInstruction]:
        """生成行为指令"""

        instructions = []

        thought_bubbles = {
            "confident": ["口气不小嘛", "有意思", "让我试试你的深浅"],
            "nervous": ["这娃子紧张了", "再加把劲", "哈哈，撑不住了吧"],
            "angry": ["呦，急了", "踩到痛点了", "这就受不了了？"],
            "neutral": ["在想着什么呢", "下一句会说什么", "嗯..."],
        }

        for npc_name, npc_state in npc_states.items():
            role = npc_state.get("role", "default")

            thought = ""
            if user_sentiment.emotion_type in thought_bubbles:
                thought = thought_bubbles[user_sentiment.emotion_type][
                    hash(npc_name) % len(thought_bubbles[user_sentiment.emotion_type])
                ]

            instruction = BehaviorInstruction(
                target_role=npc_name,
                tactic=coordination.tactic,
                description=coordination.roles.get(role, "正常应对"),
                facial_expression=self._get_facial(role, user_sentiment),
                eye_contact="直视" if role == "aggressor" else "温和",
                voice_tone=self._get_tone(role, user_sentiment),
                gesture="指点" if role == "aggressor" else "无",
                thought_bubble=thought,
            )

            instructions.append(instruction)

        return instructions

    def _get_facial(self, role: str, sentiment: UserSentiment) -> str:
        if role == "aggressor":
            if sentiment.emotion_type == "nervous":
                return "轻蔑"
            elif sentiment.emotion_type == "angry":
                return "收敛"
            else:
                return "严肃"
        elif role == "supporter":
            if sentiment.emotion_type == "nervous":
                return "关切"
            else:
                return "假笑"
        return "自然"

    def _get_tone(self, role: str, sentiment: UserSentiment) -> str:
        if sentiment.pressure_strategy == "de_escalation":
            return "缓和"
        elif sentiment.pressure_strategy == "continual_attack":
            return "严厉"
        return "自然"

    def _generate_npc_understanding(
        self, user_sentiment: UserSentiment, npc_configs: List[Dict]
    ) -> Dict[str, Dict]:
        """生成NPC对用户的理解"""

        understanding = {}

        detection_map = {
            "highly_stressed": "检测到高度紧张",
            "moderately_stressed": "检测到中等紧张",
            "aggressively_confident": "检测到过度自信",
            "hostile": "检测到愤怒情绪",
            "positive_mood": "情绪积极正面",
            "down": "情绪低落",
            "neutral": "情绪平稳",
        }

        detected = detection_map.get(user_sentiment.emotion_type, "状态未知")

        strategy_map = {
            "continual_attack": "持续施压策略",
            "de_escalation": "缓和策略",
            "supportive_approach": "支持策略",
            "probing": "试探策略",
        }

        strategy = strategy_map.get(user_sentiment.pressure_strategy, "正常应对")

        for npc in npc_configs:
            name = npc.get("name", "NPC")
            understanding[name] = {
                "detected": detected,
                "strategy": strategy,
                "hidden": user_sentiment.hidden_sentiment or "无",
            }

        return understanding
